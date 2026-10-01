"""OpenAI가 만든 질문으로 코드 실행 흐름을 학습하는 Streamlit 앱."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from copy import deepcopy
from html import escape
from typing import Any, Callable, Literal
from urllib.parse import urlencode

import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

from storage import (
    ArchiveStore,
    KeyUnavailable,
    StorageConflict,
    StorageError,
    check_progress,
    fresh_progress,
    google_owner,
    new_tutorial_record,
)


LANGUAGES = {"C": "c", "C++": "cpp", "Java": "java", "Python": "python"}
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_CONTACT_EMAIL = "be0128st@gmail.com"
LOGGER = logging.getLogger("execution_trace_tutor")


class GeneratedTraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line_number: int
    question: str
    choices: list[str]
    correct_choice_number: Literal[1, 2, 3]
    hint: str
    explanation: str


class TraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line_number: int
    question: str
    choices: list[str]
    answer: str
    hint: str
    explanation: str
    step_number: int
    code_line: str


class GeneratedTutorial(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str
    steps: list[GeneratedTraceStep] = Field(min_length=1, max_length=7)
    annotated_code: str


class Tutorial(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str
    steps: list[TraceStep]
    annotated_code: str


class TutorialError(Exception):
    """사용자에게 안전하게 표시할 수 있는 생성 실패 사유."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics


class ChoiceValidationError(TutorialError):
    """선택지만 잘못 생성된 경우 재시도할 수 있도록 구분한다."""

    def __init__(self, step_number: int, reason: str):
        super().__init__(
            f"{step_number}번 문항의 선택지에 문제가 있습니다. 다시 생성해 주세요.",
            {"step_number": step_number, "reason": reason},
        )


def build_prompt(language: str, problem: str, source: str) -> str:
    """물리적 줄 번호와 실제 실행 순서를 모델에 분명히 전달한다."""
    numbered_source = "\n".join(
        f"{number}: {line}" for number, line in enumerate(source.split("\n"), start=1)
    )
    return f"""당신은 프로그래밍 실행 추적을 가르치는 한국어 튜터입니다.
선택 언어: {language}
문제 설명: {problem}

아래 원본 소스를 처음부터 끝까지 실제 실행 순서대로 추적해 JSON 스키마에 맞춰 답하세요.
소스를 실행했다고 주장하지 말고 언어 규칙에 근거해 분석하세요.
전체 실행을 분석한 뒤 학습 가치가 높은 실행 지점만 1~7개 고르세요.
짧고 단순한 프로그램에서 7개를 억지로 채우지 마세요. 질문 순서는 실제 실행 순서입니다.
최종 출력과 문제에서 요구한 결과, 실행 오류 또는 정의되지 않은 동작의 원인을 우선하세요.
포인터 역참조, 참조 공유, 배열 접근, 함수의 부수 효과, 결과를 바꾸는 분기,
반복 중 핵심 값 변화와 종료 조건, 슬라이싱·가변 객체의 의미를 우선하세요.
단순 상수 초기화나 같은 의미의 반복 질문은 낮은 우선순위입니다.
다만 형 변환·별칭 관계 등 중요한 개념이 있다면 초기화도 질문으로 고를 수 있습니다.
질문하지 않는 줄과 반복 회차도 실제로 실행된 것으로 계산해 이후 상태에 반영하세요.
빈 줄, 주석, 단독 중괄호처럼 실행할 동작이 없는 줄은 질문하지 마세요.
같은 줄을 여러 번 고르면 각 질문에 이번 실행의 반복 변수 값과 관련 상태를 명시하세요.
출력문은 이번 실행에서 출력되는 값과 지금까지 누적된 출력 결과를 구분해 물으세요.
line_number는 아래의 1-based 물리적 줄 번호입니다. 단계를 실행 순서대로 나열하세요.
step_number와 code_line은 앱이 배열 순서와 원본 코드에서 채우므로 작성하지 마세요.
질문은 해당 줄 실행 전/후 중 어느 시점인지 명시하세요.
choices에는 서로 다른 짧은 선택지 정확히 3개를 넣고, 선택지 안에는 번호를 쓰지 마세요.
correct_choice_number는 정답 선택지의 1부터 3까지의 번호입니다. 정답 문구를 별도로 쓰지 마세요.
오답 두 개도 그럴듯하게 작성하고 정답 위치가 항상 같지 않게 하세요.
질문, 힌트, 설명은 간결하게 작성하세요.
힌트는 정답을 직접 말하지 않고 단서를 주세요. 설명은 왜 그 답인지 알려 주세요.
C/C++의 포인터·메모리·정수 규칙, Java의 참조·배열, Python의 렉시컬 스코프(LEGB)·
가변 객체·슬라이싱 등 해당 코드에 실제로 필요한 언어 규칙만 적용하세요.
입력값이 없거나 C/C++의 정의되지 않은 동작, 실행 오류가 생기면 결과를 지어내지 말고
그 지점에서 무엇이 확정되지 않는지 또는 어떤 오류가 생기는지 묻는 진단 단계를 만드세요.
annotated_code는 원본 소스의 동작을 바꾸지 않으면서 각 줄을 복습할 수 있도록
해당 언어의 주석을 추가한 전체 코드입니다. 문자열, 전처리기, 줄 이어쓰기 안에는
주석을 끼워 넣지 마세요. 모든 질문, 힌트, 설명, 주석은 한국어로 작성하세요.

원본 소스 (번호는 설명용이며 코드에 포함되지 않음):
{numbered_source}"""


def parse_json_object(text: str) -> dict[str, Any]:
    """정상 JSON을 우선 처리하고, 코드 펜스와 주변 문구만 허용한다."""
    if not text or not text.strip():
        raise TutorialError("모델이 빈 응답을 보냈습니다. 다시 생성해 주세요.")

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, re.DOTALL | re.I)
        if fenced:
            try:
                value = json.loads(fenced.group(1))
            except json.JSONDecodeError as exc:
                raise TutorialError("모델 응답의 JSON을 읽을 수 없습니다. 다시 생성해 주세요.") from exc
        else:
            decoder = json.JSONDecoder()
            value = None
            for match in re.finditer(r"\{", text):
                try:
                    candidate, end = decoder.raw_decode(text, match.start())
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict) and not re.search(r"\{", text[end:]):
                    value = candidate
                    break
            if value is None:
                raise TutorialError("모델 응답의 JSON을 읽을 수 없습니다. 다시 생성해 주세요.")

    if not isinstance(value, dict):
        raise TutorialError("모델 응답이 JSON 객체가 아닙니다. 다시 생성해 주세요.")
    return value


def check_choices(choices: list[str], step_number: int, answer: str | None = None) -> None:
    """선택지 자체와 보관함에 저장된 정답의 일치 여부를 확인한다."""
    normalized = [normalize_answer(choice) for choice in choices]
    if len(normalized) != 3:
        raise ChoiceValidationError(step_number, "선택지 개수가 3개가 아님")
    if any(not choice for choice in normalized):
        raise ChoiceValidationError(step_number, "빈 선택지 있음")
    if len(set(normalized)) != 3:
        raise ChoiceValidationError(step_number, "중복 선택지 있음")
    if answer is not None and normalized.count(normalize_answer(answer)) != 1:
        raise ChoiceValidationError(step_number, "정답이 선택지와 일치하지 않음")


def validate_tutorial(payload: dict[str, Any], language: str, source: str) -> Tutorial:
    """구조뿐 아니라 화면에 표시할 소스 위치도 검사한다."""
    try:
        tutorial = Tutorial.model_validate(payload)
    except ValidationError as exc:
        raise TutorialError("모델 응답에 필요한 단계 정보가 없습니다. 다시 생성해 주세요.") from exc

    lines = source.split("\n")
    if tutorial.language != language or not tutorial.steps or not tutorial.annotated_code.strip():
        raise TutorialError("모델 응답의 언어 또는 단계가 올바르지 않습니다. 다시 생성해 주세요.")
    for expected_number, step in enumerate(tutorial.steps, start=1):
        if step.step_number != expected_number:
            raise TutorialError("내부 단계 번호 검증에 실패했습니다. 앱을 다시 실행해 주세요.")
        if not 1 <= step.line_number <= len(lines):
            raise TutorialError("모델 응답의 줄 번호가 원본 코드 밖에 있습니다. 다시 생성해 주세요.")
        if not step.code_line.strip() or step.code_line != lines[step.line_number - 1]:
            raise TutorialError("모델 응답의 코드 줄이 원본과 다릅니다. 다시 생성해 주세요.")
        if not all(
            value.strip()
            for value in (step.question, step.answer, step.hint, step.explanation)
        ):
            raise TutorialError("모델 응답에 빈 질문이나 답변이 있습니다. 다시 생성해 주세요.")
        check_choices(step.choices, expected_number, step.answer)
    return tutorial


def finalize_generated_tutorial(payload: dict[str, Any], language: str, source: str) -> Tutorial:
    """모델이 판단한 줄 번호를 검증하고, 순서와 원문은 앱에서 확정한다."""
    if isinstance(payload.get("steps"), list) and len(payload["steps"]) > 7:
        raise TutorialError("문항이 7개를 초과했습니다. 다시 생성해 주세요.")
    try:
        generated = GeneratedTutorial.model_validate(payload)
    except ValidationError as exc:
        raise TutorialError("모델 응답에 필요한 단계 정보가 없습니다. 다시 생성해 주세요.") from exc
    lines = source.split("\n")
    steps = []
    for number, step in enumerate(generated.steps, start=1):
        if not 1 <= step.line_number <= len(lines):
            raise TutorialError("모델 응답의 줄 번호가 원본 코드 밖에 있습니다. 다시 생성해 주세요.")
        check_choices(step.choices, number)
        steps.append(
            TraceStep(
                **step.model_dump(exclude={"correct_choice_number"}),
                answer=step.choices[step.correct_choice_number - 1],
                step_number=number,
                code_line=lines[step.line_number - 1],
            )
        )
    return validate_tutorial(
        Tutorial(
            language=generated.language,
            steps=steps,
            annotated_code=generated.annotated_code,
        ).model_dump(),
        language,
        source,
    )


def describe_api_error(exc: Exception) -> str:
    """키나 원시 서버 응답을 화면에 노출하지 않고 오류를 분류한다."""
    code = getattr(exc, "status_code", None)
    try:
        code = int(code)
    except (ValueError, TypeError):
        code = None
    name = type(exc).__name__.lower()
    if code in (401, 403):
        return "OpenAI API 키 또는 접근 권한을 확인해 주세요."
    if code == 429:
        return "OpenAI 호출 한도 또는 사용 가능 잔액을 확인해 주세요."
    if code == 503:
        return "OpenAI 서버가 혼잡합니다(503). 아래 오류 상세를 확인해 주세요."
    if code == 500:
        return "OpenAI 서버에서 오류가 발생했습니다(500). 아래 오류 상세를 확인해 주세요."
    if code in (408, 504) or "timeout" in name:
        return "OpenAI 응답 시간이 초과됐습니다. 다시 시도해 주세요."
    if code == 400 or code == 404:
        return "OpenAI 요청 또는 모델 이름을 확인해 주세요."
    if code is not None and code >= 500:
        return f"OpenAI 서버 오류({code})가 발생했습니다. 아래 오류 상세를 확인해 주세요."
    if "connect" in name or "network" in name:
        return "OpenAI에 연결할 수 없습니다. 네트워크를 확인해 주세요."
    return "튜토리얼을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요."


def api_error_diagnostics(exc: Exception, api_key: str, model: str) -> dict[str, Any]:
    """키와 소스 코드는 빼고 API가 준 진단 정보만 남긴다."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    if not isinstance(error, dict):
        error = {}
    message = error.get("message") or getattr(exc, "message", None) or str(exc)
    if not isinstance(message, str):
        message = ""
    if api_key:
        message = message.replace(api_key, "[가림]")
    message = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[가림]", message)
    message = re.sub(r"(?i)(key=)[^&\s]+", r"\1[가림]", message)
    message = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._-]+", r"\1[가림]", message)
    return {
        "model": model,
        "http_status": getattr(exc, "status_code", None),
        "api_status": str(error.get("code") or error.get("type") or ""),
        "error_type": type(exc).__name__,
        "request_id": getattr(exc, "request_id", None) or headers.get("x-request-id", ""),
        "message": message[:500],
    }


def generate_tutorial(
    api_key: str,
    model: str,
    language: str,
    problem: str,
    source: str,
    on_retry: Callable[[], None] | None = None,
) -> Tutorial:
    """선택지 오류에만 한 번 더 요청하고 검증된 튜토리얼을 반환한다."""
    started = time.perf_counter()
    try:
        # 긴 자동 재시도로 사용자가 기다리지 않도록 제한한다.
        client = OpenAI(api_key=api_key, timeout=60.0, max_retries=0)
        prompt = build_prompt(language, problem, source)
        for attempt in range(2):
            response = client.responses.parse(
                model=model,
                input=prompt,
                text_format=GeneratedTutorial,
                store=False,
            )
            if response.status == "incomplete":
                raise TutorialError("OpenAI 응답이 중간에 끊겼습니다. 코드를 줄여 다시 생성해 주세요.")
            try:
                payload = (
                    response.output_parsed.model_dump()
                    if response.output_parsed is not None
                    else parse_json_object(response.output_text)
                )
                return finalize_generated_tutorial(payload, language, source)
            except ChoiceValidationError as exc:
                if attempt:
                    exc.diagnostics["retry_count"] = 1
                    LOGGER.warning("선택지 재생성 실패: %s", json.dumps(exc.diagnostics, ensure_ascii=False))
                    raise
                if on_retry:
                    on_retry()
                prompt += (
                    "\n\n앞선 응답의 선택지에 오류가 있어 폐기했습니다. "
                    "선택지를 정확히 3개 만들고, 빈 문구나 중복 문구가 없는지 확인한 뒤 "
                    "정답의 번호만 correct_choice_number에 적으세요."
                )
        raise AssertionError("재시도 횟수를 초과했습니다.")
    except TutorialError:
        raise
    except Exception as exc:
        diagnostics = api_error_diagnostics(exc, api_key, model)
        diagnostics["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        LOGGER.error("OpenAI 요청 실패: %s", json.dumps(diagnostics, ensure_ascii=False))
        raise TutorialError(describe_api_error(exc), diagnostics) from None


def normalize_answer(answer: str) -> str:
    """앞뒤 공백과 줄바꿈 형식만 정규화한다."""
    return answer.replace("\r\n", "\n").replace("\r", "\n").strip()


def current_progress(state: Any) -> dict[str, Any]:
    """화면 상태에서 저장할 최소한의 풀이 상태를 구성한다."""
    steps = len(state["quiz_data"]["steps"])
    outcomes = state.get("outcomes")
    if outcomes is None:
        outcomes = fresh_progress(steps)["outcomes"]
        for number in range(state["current_step_idx"]):
            outcomes[number]["status"] = "correct"
    return {
        "current_step_idx": state["current_step_idx"],
        "hint_opened": state["hint_opened"],
        "outcomes": deepcopy(outcomes),
    }


def commit_progress(state: Any, progress: dict[str, Any]) -> None:
    """원격 저장이 성공한 후에만 화면 상태를 변경한다."""
    check_progress(progress, len(state["quiz_data"]["steps"]))
    if state.get("active_record_id"):
        version = make_store().update_progress(
            state["owner_id"], state["active_record_id"], state["active_version"], progress
        )
        state["active_version"] = version
    state["current_step_idx"] = progress["current_step_idx"]
    state["hint_opened"] = progress["hint_opened"]
    state["outcomes"] = progress["outcomes"]
    state["needs_reload"] = False


def open_hint(state: Any) -> None:
    """힌트를 저장해 다른 기기에서도 패스 조건을 유지한다."""
    progress = current_progress(state)
    if progress["current_step_idx"] >= len(progress["outcomes"]) or progress["hint_opened"]:
        return
    progress["hint_opened"] = True
    commit_progress(state, progress)


def advance_step(state: Any, *, passed: bool = False) -> bool:
    """힌트를 열었을 때만 패스하고 한 번만 다음 단계로 이동한다."""
    steps = state["quiz_data"]["steps"]
    index = state["current_step_idx"]
    if index >= len(steps) or (passed and not state["hint_opened"]):
        return False
    progress = current_progress(state)
    progress["outcomes"][index]["status"] = "passed" if passed else "correct"
    progress["current_step_idx"] = index + 1
    progress["hint_opened"] = False
    commit_progress(state, progress)
    step = steps[index]
    state["last_result"] = {
        "step_number": index + 1,
        "line_number": step["line_number"],
        "answer": step["answer"],
        "explanation": step["explanation"],
    }
    state["feedback"] = None
    return True


def submit_answer(state: Any, answer: str) -> bool:
    """오답에서는 현재 단계와 힌트 상태를 유지한다."""
    index = state["current_step_idx"]
    steps = state["quiz_data"]["steps"]
    if index >= len(steps):
        return False
    if not answer.strip():
        state["feedback"] = "답을 입력해 주세요."
        return False
    if normalize_answer(answer) != normalize_answer(steps[index]["answer"]):
        progress = current_progress(state)
        progress["outcomes"][index]["wrong_count"] += 1
        commit_progress(state, progress)
        state["feedback"] = "아직 정답이 아닙니다. 다시 생각해 보세요."
        return False
    return advance_step(state)


def highlight_source(source: str, language: str, line_number: int) -> str:
    """원본 전체를 한 번 렉싱해 여러 줄 문자열과 현재 줄을 함께 표시한다."""
    lexer = get_lexer_by_name(LANGUAGES[language], stripnl=False, ensurenl=False)
    formatter = HtmlFormatter(linenos="inline", hl_lines=[line_number], cssclass="trace-code")
    code_html = highlight(source, lexer, formatter)
    css = formatter.get_style_defs(".trace-code")
    return f"<style>{css}\n.trace-code {{overflow-x:auto; padding:0.75rem;}}</style>{code_html}"


def setting(name: str) -> str:
    """서버 설정만 읽는다. 개인 OpenAI 키의 환경변수 대체는 사용하지 않는다."""
    try:
        secret = st.secrets.get(name, "")
    except Exception:
        secret = ""
    return str(secret or os.getenv(name, "")).strip()


def make_store() -> ArchiveStore:
    return ArchiveStore(
        setting("SUPABASE_URL"),
        setting("SUPABASE_SECRET_KEY"),
        setting("OPENAI_KEY_ENCRYPTION_KEY"),
    )


def contact_html(email: str) -> str:
    """자바스크립트 없이 키보드로도 여는 문의 버튼을 만든다."""
    subject = "[실행 추적 튜터] 문의"
    gmail = "https://mail.google.com/mail/?" + urlencode(
        {"view": "cm", "fs": "1", "to": email, "su": subject}
    )
    mailto = "mailto:" + email + "?" + urlencode({"subject": subject})
    return f"""
<style>
.contact-float {{position:fixed;right:max(16px,env(safe-area-inset-right));
bottom:max(16px,env(safe-area-inset-bottom));z-index:9999;font-family:Arial,sans-serif}}
.contact-float summary {{display:flex;align-items:center;justify-content:center;width:56px;
height:56px;float:right;border-radius:50%;background:#2554b8;color:white;
box-shadow:0 3px 12px #0004;cursor:pointer;list-style:none;font-size:25px}}
.contact-float summary::-webkit-details-marker {{display:none}}
.contact-float summary:focus-visible,.contact-float a:focus-visible {{outline:3px solid #ffb000;outline-offset:2px}}
.contact-panel {{clear:both;position:absolute;right:0;bottom:68px;width:min(290px,calc(100vw - 32px));
padding:16px;border-radius:12px;background:white;color:#202432;box-shadow:0 4px 22px #0004}}
.contact-panel a {{display:block;margin-top:10px;padding:10px;border-radius:7px;background:#eef3ff;
color:#123f96;text-align:center;text-decoration:none;font-weight:600}}
.contact-panel small {{display:block;margin-top:10px;overflow-wrap:anywhere}}
@media(max-width:600px) {{.contact-float {{bottom:max(20px,env(safe-area-inset-bottom))}}}}
</style>
<details class="contact-float"><summary aria-label="개발자에게 문의하기" title="개발자에게 문의하기">✉️</summary>
<div class="contact-panel"><strong>개발자에게 문의하기</strong>
<a href="{escape(gmail, quote=True)}" target="_blank" rel="noopener noreferrer">Gmail로 문의하기</a>
<a href="{escape(mailto, quote=True)}">기본 메일 앱으로 열기</a>
<small>받는 사람: {escape(email)}</small></div></details>
<style>.stMainBlockContainer {{padding-bottom:100px}}</style>
"""


def initialize_state() -> None:
    defaults = {
        "quiz_data": None,
        "current_step_idx": 0,
        "hint_opened": False,
        "generation_count": 0,
        "feedback": None,
        "last_result": None,
        "completed_celebrated": False,
        "study_problem": "",
        "study_language": "",
        "study_source": "",
        "generation_seconds": None,
        "outcomes": None,
        "active_record_id": None,
        "active_version": 0,
        "owner_id": None,
        "view": "학습",
        "pending_generation": None,
        "delete_candidate": None,
        "needs_reload": False,
        "archive_limit": 100,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_study(state: Any) -> None:
    """계정 변경이나 현재 기록 삭제 때 이전 사용자 내용을 지운다."""
    for key, value in {
        "quiz_data": None,
        "current_step_idx": 0,
        "hint_opened": False,
        "outcomes": None,
        "feedback": None,
        "last_result": None,
        "completed_celebrated": False,
        "study_problem": "",
        "study_language": "",
        "study_source": "",
        "generation_seconds": None,
        "active_record_id": None,
        "active_version": 0,
        "pending_generation": None,
        "delete_candidate": None,
        "needs_reload": False,
    }.items():
        state[key] = value
    state["generation_count"] += 1
    state["archive_limit"] = 100


def apply_record(state: Any, record: dict[str, Any]) -> None:
    """소유자·원본 줄 번호·진행 상태를 검사한 뒤 기록을 연다."""
    if record.get("owner_id") != state["owner_id"]:
        raise StorageError("이 문제 기록을 열 수 없습니다.")
    try:
        problem, language, source = record["problem"], record["language"], record["source"]
        tutorial = validate_tutorial(record["quiz"], language, source)
        progress = check_progress(record["progress"], len(tutorial.steps))
        record_id, version = record["id"], record["version"]
    except (KeyError, TypeError, ValueError, TutorialError) as exc:
        raise StorageError("저장된 문제 기록을 읽을 수 없습니다.") from None
    state["quiz_data"] = tutorial.model_dump()
    state["current_step_idx"] = progress["current_step_idx"]
    state["hint_opened"] = progress["hint_opened"]
    state["outcomes"] = deepcopy(progress["outcomes"])
    state["active_record_id"] = record_id
    state["active_version"] = version
    state["study_problem"] = problem
    state["study_language"] = language
    state["study_source"] = source
    state["feedback"] = None
    state["completed_celebrated"] = progress["current_step_idx"] >= len(tutorial.steps)
    state["generation_seconds"] = None
    state["generation_count"] += 1
    state["needs_reload"] = False
    index = progress["current_step_idx"]
    if index:
        step = tutorial.steps[index - 1]
        state["last_result"] = {
            "step_number": index,
            "line_number": step.line_number,
            "answer": step.answer,
            "explanation": step.explanation,
        }
    else:
        state["last_result"] = None


def show_review_history(steps: list[dict[str, Any]], completed: int, language: str) -> None:
    """완료한 문항만 다시 보여주고, 현재 문항의 정답은 숨긴다."""
    if completed == 0:
        return
    with st.expander(f"지난 단계 다시 보기 ({completed}개 완료)"):
        st.dataframe(
            [
                {
                    "단계": number + 1,
                    "행": step["line_number"],
                    "질문": step["question"],
                    "정답": step["answer"],
                    "진행": (
                        "패스" if st.session_state.get("outcomes")
                        and st.session_state["outcomes"][number]["status"] == "passed" else "정답"
                    ),
                }
                for number, step in enumerate(steps[:completed])
            ],
            hide_index=True,
            width="stretch",
        )
        selected = st.selectbox(
            "해설을 볼 단계",
            range(completed),
            format_func=lambda number: f"{number + 1}단계 · {steps[number]['line_number']}행",
            key=f"review_step_{st.session_state['generation_count']}",
        )
        step = steps[selected]
        st.code(step["code_line"], language=LANGUAGES[language])
        st.write(f"**질문:** {step['question']}")
        st.write("**선택지:** " + " · ".join(step["choices"]))
        st.success(f"정답: {step['answer']}")
        st.write(f"**해설:** {step['explanation']}")


def show_study_view() -> None:
    quiz = st.session_state["quiz_data"]
    if quiz is None:
        return
    index = st.session_state["current_step_idx"]
    steps = quiz["steps"]
    language = st.session_state["study_language"]

    st.divider()
    st.subheader("실행 추적 학습")
    st.write(f"**문제:** {st.session_state['study_problem']}")
    st.write(f"**언어:** {language}")
    if st.session_state["generation_seconds"] is not None:
        st.caption(f"생성 소요 시간: {st.session_state['generation_seconds']:.1f}초")
    if st.session_state["last_result"]:
        result = st.session_state["last_result"]
        previous = (
            f"직전 {result['step_number']}단계 ({result['line_number']}행)"
            if "step_number" in result else "직전 단계"
        )
        st.success(f"{previous}의 정답: {result['answer']}")
        st.write(result["explanation"])
    show_review_history(steps, min(index, len(steps)), language)

    # 인덱스 접근보다 완료 조건을 먼저 처리한다.
    if index >= len(steps):
        if not st.session_state["completed_celebrated"]:
            st.balloons()
            st.session_state["completed_celebrated"] = True
        st.success("모든 단계를 완료했습니다! 아래 주석 코드를 복습해 보세요.")
        st.code(quiz["annotated_code"], language=LANGUAGES[language], line_numbers=True)
        return

    step = steps[index]
    st.progress((index + 1) / len(steps), text=f"{index + 1} / {len(steps)}단계")
    st.html(highlight_source(st.session_state["study_source"], language, step["line_number"]))
    st.write(f"**현재 {step['line_number']}행:** {step['question']}")

    # 휴대전화에서도 한 번 탭하면 채점되도록 선택지를 버튼으로 표시한다.
    for choice_number, choice in enumerate(step["choices"], start=1):
        key = f"choice_{st.session_state['generation_count']}_{index}_{choice_number}"
        if st.button(f"{choice_number}. {choice}", key=key, width="stretch"):
            try:
                if submit_answer(st.session_state, choice):
                    st.rerun()
            except StorageError as exc:
                st.error(str(exc))
                if isinstance(exc, StorageConflict):
                    st.session_state["needs_reload"] = True

    if st.session_state["feedback"]:
        st.warning(st.session_state["feedback"])

    if st.button("💡 Show Hint", key=f"hint_{st.session_state['generation_count']}_{index}"):
        try:
            open_hint(st.session_state)
        except StorageError as exc:
            st.error(str(exc))
            if isinstance(exc, StorageConflict):
                st.session_state["needs_reload"] = True
    if st.session_state["hint_opened"]:
        st.info(step["hint"])
        if st.button("➡️ Next Line (Pass)", key=f"pass_{st.session_state['generation_count']}_{index}"):
            try:
                if advance_step(st.session_state, passed=True):
                    st.rerun()
            except StorageError as exc:
                st.error(str(exc))
                if isinstance(exc, StorageConflict):
                    st.session_state["needs_reload"] = True

    if st.session_state.get("needs_reload") and st.button("최신 풀이 상태 불러오기"):
        try:
            record = make_store().get_tutorial(st.session_state["owner_id"], st.session_state["active_record_id"])
            apply_record(st.session_state, record)
            st.session_state["needs_reload"] = False
            st.rerun()
        except StorageError as exc:
            st.error(str(exc))


def show_key_settings(store: ArchiveStore, owner: str) -> str:
    """개인 키를 한 번 등록하고 이후에는 상태만 보여준다."""
    st.subheader("OpenAI 설정")
    try:
        registered = store.has_key(owner)
    except StorageError as exc:
        st.error(str(exc))
        registered = False
    if registered:
        st.success("개인 API 키 등록됨")
    with st.form("key_form", clear_on_submit=True):
        new_key = st.text_input(
            "API 키 등록·변경", type="password", autocomplete="off",
            help="한 번 등록하면 같은 Google 계정으로 다른 기기에서도 사용할 수 있습니다.",
            key=f"api_key_input_{owner}",
        )
        save_key = st.form_submit_button("API 키 저장")
    if save_key:
        try:
            store.save_key(owner, new_key)
        except StorageError as exc:
            st.error(str(exc))
        else:
            st.rerun()
    if registered and st.button("저장된 API 키 삭제"):
        try:
            store.delete_key(owner)
        except StorageError as exc:
            st.error(str(exc))
        else:
            st.rerun()
    return st.text_input("모델", value=DEFAULT_MODEL, key="openai_model")


def save_pending_generation(store: ArchiveStore, owner: str) -> None:
    """API 재호출 없이 생성된 결과를 저장한다."""
    pending = st.session_state["pending_generation"]
    if not pending:
        return
    st.warning("생성된 문제를 아직 보관하지 못했습니다. 저장을 다시 시도할 수 있습니다.")
    if st.button("생성된 문제 저장 다시 시도"):
        try:
            record = store.insert_tutorial(owner, pending)
            apply_record(st.session_state, record)
        except (StorageError, TutorialError) as exc:
            st.error(str(exc))
        else:
            st.session_state["pending_generation"] = None
            st.rerun()
    if st.button("저장하지 않고 버리기"):
        st.session_state["pending_generation"] = None
        st.rerun()


def show_generate_view(store: ArchiveStore, owner: str, model: str) -> None:
    """생성 결과가 보관된 뒤에만 현재 학습을 교체한다."""
    with st.form("generate_form"):
        language = st.selectbox("프로그래밍 언어", list(LANGUAGES), key=f"language_input_{owner}")
        problem = st.text_area("문제 설명", placeholder="예: 다음 프로그램의 실행 결과는?", key=f"problem_input_{owner}")
        source = st.text_area("소스 코드", height=260, placeholder="코드를 여기에 붙여 넣으세요.", key=f"source_input_{owner}")
        requested = st.form_submit_button("핵심 문제 생성")

    if requested:
        if st.session_state["pending_generation"]:
            st.error("먼저 생성된 문제를 저장하거나 버려 주세요.")
        elif not problem.strip() or not source.strip():
            st.error("문제 설명과 소스 코드를 모두 입력해 주세요.")
        elif not model.strip():
            st.error("모델 이름을 입력해 주세요.")
        else:
            started = time.perf_counter()
            retry_notice = st.empty()
            try:
                api_key = store.get_key(owner)
                with st.spinner("전체 실행을 분석하고 핵심 문항을 고르는 중입니다..."):
                    tutorial = generate_tutorial(
                        api_key, model.strip(), language, problem, source,
                        on_retry=lambda: retry_notice.info("선택지 오류가 있어 한 번 더 생성하고 있습니다..."),
                    )
            except StorageError as exc:
                st.error(str(exc))
            except TutorialError as exc:
                st.error(str(exc))
                st.caption(f"요청 경과 시간: {time.perf_counter() - started:.1f}초")
                if exc.diagnostics:
                    with st.expander("생성 오류 상세 (키 제외)"):
                        st.json(exc.diagnostics)
            else:
                record = new_tutorial_record(problem, language, source, model.strip(), tutorial.model_dump())
                st.session_state["pending_generation"] = record
                try:
                    saved = store.insert_tutorial(owner, record)
                    apply_record(st.session_state, saved)
                except (StorageError, TutorialError) as exc:
                    st.error(str(exc))
                else:
                    st.session_state["pending_generation"] = None
                    st.session_state["generation_seconds"] = time.perf_counter() - started
                    st.rerun()
            finally:
                retry_notice.empty()

    save_pending_generation(store, owner)
    show_study_view()


def show_archive_view(store: ArchiveStore, owner: str) -> None:
    st.subheader("내 문제 보관함")
    try:
        records = store.list_tutorials(owner, st.session_state["archive_limit"])
    except StorageError as exc:
        st.error(str(exc))
        return
    language_filter = st.selectbox("언어 필터", ["전체", *LANGUAGES], key="archive_language")
    status_filter = st.selectbox("진행 상태", ["전체", "진행 중", "완료"], key="archive_status")
    shown = 0
    for summary in records:
        progress = summary.get("progress") or {}
        outcomes = progress.get("outcomes") or []
        completed = progress.get("current_step_idx") == len(outcomes) and bool(outcomes)
        if language_filter != "전체" and summary["language"] != language_filter:
            continue
        if (status_filter == "완료" and not completed) or (status_filter == "진행 중" and completed):
            continue
        shown += 1
        record_id = summary["id"]
        state_label = "완료" if completed else f"진행 중 ({progress.get('current_step_idx', 0)}/{len(outcomes)})"
        title = summary["problem"].strip().splitlines()[0][:55]
        with st.expander(f"{summary['language']} · {state_label} · {title}"):
            st.caption(f"마지막 학습: {summary.get('updated_at', '')[:16]}")
            if st.button("열기·복습" if completed else "이어 풀기", key=f"open_{record_id}"):
                try:
                    record = store.get_tutorial(owner, record_id)
                    apply_record(st.session_state, record)
                except (StorageError, TutorialError) as exc:
                    st.error(str(exc))
                else:
                    st.session_state["view"] = "학습"
                    st.rerun()
            if st.button("다시 풀기", key=f"replay_{record_id}"):
                try:
                    original = store.get_tutorial(owner, record_id)
                    validate_tutorial(original["quiz"], original["language"], original["source"])
                    replay = new_tutorial_record(
                        original["problem"], original["language"], original["source"],
                        original["model"], original["quiz"],
                    )
                    st.session_state["pending_generation"] = replay
                    saved = store.insert_tutorial(owner, replay)
                    apply_record(st.session_state, saved)
                except (StorageError, TutorialError) as exc:
                    st.error(str(exc))
                else:
                    st.session_state["pending_generation"] = None
                    st.session_state["view"] = "학습"
                    st.rerun()
            if st.button("삭제", key=f"delete_{record_id}"):
                st.session_state["delete_candidate"] = record_id
            if st.session_state["delete_candidate"] == record_id:
                st.warning("이 문제와 풀이 기록을 삭제할까요?")
                if st.button("삭제 확인", key=f"confirm_{record_id}"):
                    try:
                        store.delete_tutorial(owner, record_id)
                    except StorageError as exc:
                        st.error(str(exc))
                    else:
                        if st.session_state["active_record_id"] == record_id:
                            reset_study(st.session_state)
                        st.session_state["delete_candidate"] = None
                        st.rerun()
    if not shown:
        st.info("표시할 문제 기록이 없습니다.")
    if len(records) == st.session_state["archive_limit"] and st.button("더 보기"):
        st.session_state["archive_limit"] += 100
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="실행 추적 튜터", page_icon="🧭", layout="wide")
    initialize_state()
    st.html(contact_html(setting("CONTACT_EMAIL") or DEFAULT_CONTACT_EMAIL))
    st.title("다국어 실행 추적 튜터")
    st.caption("C · C++ · Java · Python")

    owner = google_owner(st.user)
    if not owner:
        if getattr(st.user, "is_logged_in", False):
            st.error("Google 계정 정보를 확인할 수 없습니다. 다시 로그인해 주세요.")
            if st.button("로그아웃"):
                st.logout()
        else:
            st.write("Google 계정으로 로그인해 본인 문제와 API 키를 불러오세요.")
            if st.button("Google 로그인"):
                try:
                    st.login()
                except Exception:
                    st.error("Google 로그인 설정을 확인해 주세요.")
        st.stop()

    if st.session_state["owner_id"] != owner:
        previous_owner = st.session_state["owner_id"]
        reset_study(st.session_state)
        if previous_owner:
            for prefix in ("api_key_input", "language_input", "problem_input", "source_input"):
                st.session_state.pop(f"{prefix}_{previous_owner}", None)
        st.session_state["owner_id"] = owner
        st.session_state["view"] = "학습"
    try:
        store = make_store()
    except StorageError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.write(st.user.get("email", "Google 계정"))
        if st.button("로그아웃"):
            reset_study(st.session_state)
            st.logout()
        model = show_key_settings(store, owner)

    left, right = st.columns(2)
    with left:
        if st.button("학습", width="stretch"):
            st.session_state["view"] = "학습"
    with right:
        if st.button("내 문제 보관함", width="stretch"):
            st.session_state["view"] = "보관함"

    if st.session_state["view"] == "보관함":
        show_archive_view(store, owner)
    else:
        show_generate_view(store, owner, model)


if __name__ == "__main__":
    main()
