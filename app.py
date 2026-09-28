"""OpenAI가 만든 질문으로 코드 실행 흐름을 학습하는 Streamlit 앱."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, ValidationError
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name


LANGUAGES = {"C": "c", "C++": "cpp", "Java": "java", "Python": "python"}
DEFAULT_MODEL = "gpt-4.1-mini"
LOGGER = logging.getLogger("execution_trace_tutor")


class GeneratedTraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line_number: int
    question: str
    choices: list[str]
    answer: str
    hint: str
    explanation: str


class TraceStep(GeneratedTraceStep):
    step_number: int
    code_line: str


class GeneratedTutorial(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str
    steps: list[GeneratedTraceStep]
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


def build_prompt(language: str, problem: str, source: str) -> str:
    """물리적 줄 번호와 실제 실행 순서를 모델에 분명히 전달한다."""
    numbered_source = "\n".join(
        f"{number}: {line}" for number, line in enumerate(source.split("\n"), start=1)
    )
    return f"""당신은 프로그래밍 실행 추적을 가르치는 한국어 튜터입니다.
선택 언어: {language}
문제 설명: {problem}

아래 원본 소스를 실제 실행 순서대로 추적해 JSON 스키마에 맞춰 답하세요.
소스를 실행했다고 주장하지 말고 언어 규칙에 근거해 분석하세요.
실제로 실행되는 각 문장과 조건 판단 줄을 빠뜨리지 말고 한 단계씩 만드세요.
선언, 대입, 분기 조건, 반복 조건과 갱신, 함수 호출, 출력, 반환도 포함하세요.
빈 줄, 주석, 단독 중괄호처럼 실행할 동작이 없는 줄은 제외하세요.
분기, 반복, 함수 호출은 실제 실행 순서로 배치하고 반복 방문한 줄은 별도 단계로 넣으세요.
같은 줄을 여러 번 방문하면 질문에 이번 실행의 반복 변수 값이나 관련 상태를 명시하세요.
출력문은 이번 실행에서 출력되는 값과 지금까지 누적된 출력 결과를 구분해 물으세요.
같은 줄이 다시 실행되더라도 반복 조건 검사와 갱신 등 사이에 실행되는 단계를 빠뜨리지 마세요.
line_number는 아래의 1-based 물리적 줄 번호입니다. 단계를 실행 순서대로 나열하세요.
step_number와 code_line은 앱이 배열 순서와 원본 코드에서 채우므로 작성하지 마세요.
질문은 해당 줄 실행 전/후 중 어느 시점인지 명시하세요.
choices에는 서로 다른 짧은 선택지 정확히 3개를 넣고, 그중 한 개만 정답이 되게 하세요.
answer는 정답 선택지의 문구를 그대로 복사하세요. 오답 두 개도 그럴듯하게 작성하고
정답 위치가 항상 같지 않게 하세요. 질문, 힌트, 설명은 간결하게 작성하세요.
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
        choices = [normalize_answer(choice) for choice in step.choices]
        if (
            len(choices) != 3
            or any(not choice for choice in choices)
            or len(set(choices)) != 3
            or choices.count(normalize_answer(step.answer)) != 1
        ):
            raise TutorialError("모델 응답의 3지선다 선택지가 올바르지 않습니다. 다시 생성해 주세요.")
    return tutorial


def finalize_generated_tutorial(payload: dict[str, Any], language: str, source: str) -> Tutorial:
    """모델이 판단한 줄 번호를 검증하고, 순서와 원문은 앱에서 확정한다."""
    try:
        generated = GeneratedTutorial.model_validate(payload)
    except ValidationError as exc:
        raise TutorialError("모델 응답에 필요한 단계 정보가 없습니다. 다시 생성해 주세요.") from exc
    lines = source.split("\n")
    steps = []
    for number, step in enumerate(generated.steps, start=1):
        if not 1 <= step.line_number <= len(lines):
            raise TutorialError("모델 응답의 줄 번호가 원본 코드 밖에 있습니다. 다시 생성해 주세요.")
        steps.append(
            TraceStep(
                **step.model_dump(),
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


def generate_tutorial(api_key: str, model: str, language: str, problem: str, source: str) -> Tutorial:
    """구조화 응답을 받은 뒤 검증된 튜토리얼만 반환한다."""
    started = time.perf_counter()
    try:
        # 긴 자동 재시도로 사용자가 기다리지 않도록 제한한다.
        client = OpenAI(api_key=api_key, timeout=60.0, max_retries=0)
        response = client.responses.parse(
            model=model,
            input=build_prompt(language, problem, source),
            text_format=GeneratedTutorial,
            store=False,
        )
    except Exception as exc:
        diagnostics = api_error_diagnostics(exc, api_key, model)
        diagnostics["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        LOGGER.error("OpenAI 요청 실패: %s", json.dumps(diagnostics, ensure_ascii=False))
        raise TutorialError(describe_api_error(exc), diagnostics) from None
    if response.status == "incomplete":
        raise TutorialError("OpenAI 응답이 중간에 끊겼습니다. 코드를 줄여 다시 생성해 주세요.")
    if response.output_parsed is not None:
        return finalize_generated_tutorial(response.output_parsed.model_dump(), language, source)
    return finalize_generated_tutorial(parse_json_object(response.output_text), language, source)


def normalize_answer(answer: str) -> str:
    """앞뒤 공백과 줄바꿈 형식만 정규화한다."""
    return answer.replace("\r\n", "\n").replace("\r", "\n").strip()


def advance_step(state: Any, *, passed: bool = False) -> bool:
    """힌트를 열었을 때만 패스하고 한 번만 다음 단계로 이동한다."""
    steps = state["quiz_data"]["steps"]
    index = state["current_step_idx"]
    if index >= len(steps) or (passed and not state["hint_opened"]):
        return False
    step = steps[index]
    state["last_result"] = {
        "step_number": index + 1,
        "line_number": step["line_number"],
        "answer": step["answer"],
        "explanation": step["explanation"],
    }
    state["current_step_idx"] = index + 1
    state["hint_opened"] = False
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


def resolve_api_key(typed_key: str) -> str:
    """세션 입력, secrets, 환경변수 순으로 키를 찾는다."""
    if typed_key.strip():
        return typed_key.strip()
    try:
        secret = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:  # secrets 파일이 설정되지 않은 로컬 실행도 지원한다.
        secret = ""
    return str(secret or os.getenv("OPENAI_API_KEY", "")).strip()


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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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
    visit_count = sum(item["line_number"] == step["line_number"] for item in steps)
    visit_number = sum(item["line_number"] == step["line_number"] for item in steps[:index + 1])
    current = f"현재 {step['line_number']}행"
    if visit_count > 1:
        current += f" (총 {visit_count}회 중 {visit_number}번째 실행)"
    st.write(f"**{current}:** {step['question']}")

    # 휴대전화에서도 한 번 탭하면 채점되도록 선택지를 버튼으로 표시한다.
    for choice_number, choice in enumerate(step["choices"], start=1):
        key = f"choice_{st.session_state['generation_count']}_{index}_{choice_number}"
        if st.button(f"{choice_number}. {choice}", key=key, width="stretch"):
            if submit_answer(st.session_state, choice):
                st.rerun()

    if st.session_state["feedback"]:
        st.warning(st.session_state["feedback"])

    if st.button("💡 Show Hint", key=f"hint_{st.session_state['generation_count']}_{index}"):
        st.session_state["hint_opened"] = True
    if st.session_state["hint_opened"]:
        st.info(step["hint"])
        if st.button("➡️ Next Line (Pass)", key=f"pass_{st.session_state['generation_count']}_{index}"):
            if advance_step(st.session_state, passed=True):
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="실행 추적 튜터", page_icon="🧭", layout="wide")
    initialize_state()
    st.title("다국어 실행 추적 튜터")
    st.caption("C · C++ · Java · Python")

    with st.sidebar:
        st.subheader("OpenAI 설정")
        typed_key = st.text_input("OpenAI API 키", type="password", key="openai_api_key", help="입력한 키는 현재 세션에서만 사용합니다.")
        model = st.text_input("모델", value=DEFAULT_MODEL, key="openai_model")

    with st.form("generate_form"):
        language = st.selectbox("프로그래밍 언어", list(LANGUAGES))
        problem = st.text_area("문제 설명", placeholder="예: 다음 프로그램의 실행 결과는?")
        source = st.text_area("소스 코드", height=260, placeholder="코드를 여기에 붙여 넣으세요.")
        requested = st.form_submit_button("Generate Interactive Tutorial")

    if requested:
        api_key = resolve_api_key(typed_key)
        if not problem.strip() or not source.strip():
            st.error("문제 설명과 소스 코드를 모두 입력해 주세요.")
        elif not model.strip():
            st.error("모델 이름을 입력해 주세요.")
        elif not api_key:
            st.error("OpenAI API 키를 입력하거나 OPENAI_API_KEY를 설정해 주세요.")
        else:
            started = time.perf_counter()
            try:
                with st.spinner("실행 줄마다 3지선다와 주석 코드를 만드는 중입니다. 반복문이 많으면 시간이 걸릴 수 있습니다..."):
                    tutorial = generate_tutorial(api_key, model.strip(), language, problem, source)
            except TutorialError as exc:
                st.error(str(exc))
                st.caption(f"요청 경과 시간: {time.perf_counter() - started:.1f}초")
                if exc.diagnostics:
                    with st.expander("API 오류 상세 (키 제외)"):
                        st.json(exc.diagnostics)
            else:
                # 검증이 끝나야 기존 학습을 새 내용으로 교체한다.
                st.session_state["quiz_data"] = tutorial.model_dump()
                st.session_state["current_step_idx"] = 0
                st.session_state["hint_opened"] = False
                st.session_state["generation_count"] += 1
                st.session_state["feedback"] = None
                st.session_state["last_result"] = None
                st.session_state["completed_celebrated"] = False
                st.session_state["study_problem"] = problem
                st.session_state["study_language"] = language
                st.session_state["study_source"] = source
                st.session_state["generation_seconds"] = time.perf_counter() - started

    show_study_view()


if __name__ == "__main__":
    main()
