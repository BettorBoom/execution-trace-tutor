"""OpenAI가 만든 질문으로 코드 실행 흐름을 학습하는 Streamlit 앱."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal, InvalidOperation
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
QUIZ_SCHEMA_VERSION = 2
LOGGER = logging.getLogger("execution_trace_tutor")


class Calculation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: int
    operator: Literal["+", "-", "*", "%"]
    right: int
    result: int


class StateChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    before: str
    after: str
    calculations: list[Calculation]


class LineNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line_number: int
    note: str


class GeneratedTraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line_number: int
    question_kind: Literal["value_after", "output_this_step", "output_so_far", "diagnostic"]
    target: str
    context: str
    answer: str
    distractors: list[str]
    hint: str
    explanation: str
    changes: list[StateChange]


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
    question_kind: str = ""
    target: str = ""
    context: str = ""
    changes: list[StateChange] = Field(default_factory=list)


class GeneratedTutorial(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str
    # OpenAI의 엄격한 JSON Schema에서 배열 길이 제약이 400을 유발할 수 있어 앱에서 검사한다.
    steps: list[GeneratedTraceStep]
    line_notes: list[LineNote]


class Tutorial(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str
    steps: list[TraceStep]
    annotated_code: str
    schema_version: int = 1
    line_notes: list[LineNote] = Field(default_factory=list)
    excluded_steps: int = 0


class TutorialError(Exception):
    """사용자에게 안전하게 표시할 수 있는 생성 실패 사유."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics


class GenerationValidationError(TutorialError):
    """생성 내용의 일관성 오류는 한 번 재생성할 수 있다."""

    def __init__(self, step_number: int, reason: str):
        super().__init__(
            (
                f"{step_number}번 문항의 실행 추적 내용에 문제가 있습니다. 다시 생성해 주세요."
                if step_number else "모델 응답의 실행 추적 내용에 문제가 있습니다. 다시 생성해 주세요."
            ),
            {"step_number": step_number, "reason": reason},
        )


ChoiceValidationError = GenerationValidationError


def build_prompt(language: str, problem: str, source: str) -> str:
    """정답과 수치 근거를 한 번만 생성하고 화면 문장은 앱이 구성한다."""
    numbered_source = "\n".join(
        f"{number}: {line}" for number, line in enumerate(source.split("\n"), start=1)
    )
    return f"""당신은 초보자에게 코드 실행을 설명하는 한국어 튜터입니다.
선택 언어: {language}
문제 설명: {problem}

아래 원본 소스의 전체 실행을 분석하세요. 실행했다고 주장하지 말고 언어 규칙에 따라 계산하세요.
실제 실행 순서에서 학습 가치가 높은 지점만 1~7개 고르세요. 짧은 코드는 억지로 채우지 마세요.
최종 출력·문제에서 요구한 결과·오류 원인, 포인터/참조/배열/슬라이싱, 함수 부수 효과와
반복문의 핵심 값 변화를 우선하세요. 단순 상수 초기화와 같은 의미의 반복 질문은 피하세요.
문제가 출력값을 요구하고 코드에 출력문이 있으면, 마지막 출력문 또는 그 결과를 확정할 수
없는 오류 지점을 반드시 문항으로 포함하세요. 출력값 문항의 line_number는 실제 출력문 행입니다.
질문하지 않는 줄과 반복 회차도 실행한 것으로 계산하세요. 같은 줄의 반복 출제는 context에
반복 변수와 이번 회차를 명시하세요. context와 hint에 정답 값을 미리 쓰지 마세요.
line_number는 아래 1-based 물리적 줄 번호이며 steps는 실제 실행 순서입니다.
빈 줄·주석·단독 중괄호는 문항으로 만들지 마세요.

question_kind: 값은 value_after, 이번 출력은 output_this_step, 누적 출력은 output_so_far,
입력 미지정/정의되지 않은 동작/실행 오류는 diagnostic입니다. 앱이 질문 문장을 만드므로
질문 문장이나 번호를 생성하지 마세요. target에는 묻는 변수 또는 표현식만 적으세요.
answer에는 실제 정답 값 하나만 적고, distractors에는 서로 다른 오답 정확히 2개를 적으세요.
숫자를 묻는 문항의 오답도 숫자 형식이어야 합니다. 정답 또는 같은 뜻의 값은 오답에 넣지 마세요.
value_after에서는 target을 첫 번째 changes 항목의 target과 글자까지 똑같이 적으세요.
첫 번째 changes 항목에는 그 target의 직전 값 before, 직후 값 after를 넣고,
answer를 첫 번째 after와 글자까지 똑같이 적으세요. 관련된 다른 변수 변화는 그 뒤에 적으세요.
포인터의 주소 자체처럼 숫자로 확정할 수 없는 값은 묻지 말고 *p, arr[i] 등의 실제 값이나
참조 관계의 효과를 물으세요.
계산 근거가 단순 정수 +, -, *, 양수 %이면 calculations에 피연산자·결과를 순서대로 적으세요.
예: arr[2]=(4+2)%5이면 (4,+,2,6), (6,%,5,1), after='1', answer='1'.
지원하지 않는 연산의 calculations는 비워 두고 explanation에 언어 규칙과 계산을 설명하세요.
explanation은 초보자가 숫자를 따라갈 수 있게 직전 값→연산→직후 값을 구체적으로 적으세요.
hint는 정답을 직접 밝히지 않는 단서로 작성하세요.
line_notes에는 출제하지 않은 핵심 실행 행과 반복문의 대표·마지막 회차를 행 번호·값 변화로
설명하세요. 메모는 최대 20개로 요약하고, 반복이 길면 모든 회차를 나열하지 마세요.
중괄호 자체나 단순 선언에는 메모를 억지로 만들지 마세요. 원본 코드를 다시 출력하지 마세요.
C/C++ 포인터·정수 규칙, Java 참조·배열, Python LEGB·가변 객체·슬라이싱 중 실제 코드에
필요한 규칙을 적용하세요. 모든 설명과 메모는 간단하고 쉬운 한국어로 작성하세요.

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


def choice_identity(value: str, question_kind: str) -> str | Decimal:
    """값 질문에서 1, 01, 1.0처럼 같은 수인 선택지를 구분하지 않는다."""
    normalized = normalize_answer(value)
    if question_kind == "value_after" and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", normalized):
        try:
            return Decimal(normalized)
        except InvalidOperation:
            pass
    return normalized


def check_choices(
    choices: list[str], step_number: int, answer: str | None = None, question_kind: str = ""
) -> None:
    """선택지 자체와 보관함에 저장된 정답의 일치 여부를 확인한다."""
    normalized = [normalize_answer(choice) for choice in choices]
    if len(normalized) != 3:
        raise ChoiceValidationError(step_number, "선택지 개수가 3개가 아님")
    if any(not choice for choice in normalized):
        raise ChoiceValidationError(step_number, "빈 선택지 있음")
    if len({choice_identity(choice, question_kind) for choice in normalized}) != 3:
        raise ChoiceValidationError(step_number, "중복 선택지 있음")
    if answer is not None and sum(
        choice_identity(choice, question_kind) == choice_identity(answer, question_kind)
        for choice in normalized
    ) != 1:
        raise ChoiceValidationError(step_number, "정답이 선택지와 일치하지 않음")
    if answer is not None and isinstance(choice_identity(answer, question_kind), Decimal) and any(
        not isinstance(choice_identity(choice, question_kind), Decimal) for choice in normalized
    ):
        raise ChoiceValidationError(step_number, "숫자 질문에 숫자가 아닌 선택지 있음")


def question_for(kind: str, line_number: int, target: str, context: str) -> str:
    """답의 형식이 질문 문장과 어긋나지 않도록 앱에서 작성한다."""
    prefix = f"{context.strip()} — " if context.strip() else ""
    if kind == "value_after":
        return f"{prefix}{line_number}행 실행 직후 `{target}`의 값은 무엇인가요? 선택지에서 값 하나를 고르세요."
    if kind == "output_this_step":
        return f"{prefix}{line_number}행의 `{target}`가 이번에 출력하는 내용은 무엇인가요?"
    if kind == "output_so_far":
        return f"{prefix}{line_number}행의 `{target}` 실행 직후까지 누적된 출력은 무엇인가요?"
    return f"{prefix}{line_number}행의 `{target}` 결과를 확정할 수 없는 이유는 무엇인가요?"


def check_calculations(changes: list[StateChange], step_number: int) -> None:
    """모델이 제시한 작은 정수 계산만 검사한다. 코드 전체를 실행하지 않는다."""
    for change in changes:
        if not change.target.strip() or not change.before.strip() or not change.after.strip():
            raise GenerationValidationError(step_number, "값 변화의 대상 또는 전후 값이 비어 있음")
        previous_result = None
        for calculation in change.calculations:
            a, b = calculation.left, calculation.right
            if previous_result is not None and previous_result not in (a, b):
                raise GenerationValidationError(step_number, "연속 계산의 중간 값이 이어지지 않음")
            if calculation.operator == "+":
                expected = a + b
            elif calculation.operator == "-":
                expected = a - b
            elif calculation.operator == "*":
                expected = a * b
            else:
                if a < 0 or b <= 0:
                    raise GenerationValidationError(step_number, "로컬에서 검증할 수 없는 나머지 계산")
                expected = a % b
            if expected != calculation.result:
                raise GenerationValidationError(step_number, "기재된 정수 계산의 결과가 다름")
            previous_result = calculation.result
        if change.calculations and change.after.strip() != str(change.calculations[-1].result):
            raise GenerationValidationError(step_number, "계산 결과와 실행 후 값이 다름")


def review_code(source: str, language: str, notes: list[LineNote], steps: list[TraceStep]) -> str:
    """원본 뒤에 행별 메모를 붙여 문자열·전처리기 내용을 보존한다."""
    marker = "#" if language == "Python" else "//"
    all_notes = [(item.line_number, item.note) for item in notes]
    for step in steps:
        for change in step.changes:
            all_notes.append((
                step.line_number,
                f"{step.context}: {change.target} {change.before} → {change.after}",
            ))
    comments = [
        f"{marker} {line_number}행: {note.replace(chr(10), ' ').replace(chr(13), ' ')}"
        for line_number, note in all_notes
    ]
    return source + ("\n\n" + "\n".join(comments) if comments else "")


def validate_tutorial(payload: dict[str, Any], language: str, source: str) -> Tutorial:
    """구조뿐 아니라 화면에 표시할 소스 위치도 검사한다."""
    try:
        tutorial = Tutorial.model_validate(payload)
    except ValidationError as exc:
        raise TutorialError("모델 응답에 필요한 단계 정보가 없습니다. 다시 생성해 주세요.") from exc

    lines = source.split("\n")
    if tutorial.language != language or not tutorial.steps or not tutorial.annotated_code.strip():
        raise TutorialError("모델 응답의 언어 또는 단계가 올바르지 않습니다. 다시 생성해 주세요.")
    if len(tutorial.steps) > 7 and tutorial.schema_version >= QUIZ_SCHEMA_VERSION:
        raise TutorialError("문항이 7개를 초과했습니다. 다시 생성해 주세요.")
    if type(tutorial.excluded_steps) is not int or not 0 <= tutorial.excluded_steps <= 6:
        raise TutorialError("제외된 문항 수가 올바르지 않습니다.")
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
        check_choices(step.choices, expected_number, step.answer, step.question_kind)
        if tutorial.schema_version >= QUIZ_SCHEMA_VERSION:
            if not step.target.strip() or (step.question_kind == "value_after" and not step.changes):
                raise TutorialError("실행 시점 또는 값 변화 설명이 빠졌습니다. 다시 생성해 주세요.")
            if step.question != question_for(step.question_kind, step.line_number, step.target, step.context):
                raise TutorialError("저장된 질문 문장이 실행 정보와 다릅니다.")
            check_calculations(step.changes, expected_number)
            if step.question_kind == "value_after" and not any(
                item.target.strip() == step.target.strip()
                and normalize_answer(item.after) == normalize_answer(step.answer)
                for item in step.changes
            ):
                raise GenerationValidationError(expected_number, "정답과 대상 변수의 실행 후 값이 다름")
            if step.question_kind == "value_after":
                change = next(item for item in step.changes if item.target.strip() == step.target.strip())
                if (
                    change.before.lstrip("-").isdigit()
                    and change.after.lstrip("-").isdigit()
                    and change.before != change.after
                    and (change.before not in step.explanation or change.after not in step.explanation)
                ):
                    raise GenerationValidationError(expected_number, "해설에 값 변화가 빠짐")
    if tutorial.schema_version >= QUIZ_SCHEMA_VERSION:
        for note in tutorial.line_notes:
            if not 1 <= note.line_number <= len(lines) or not note.note.strip():
                raise TutorialError("행별 주석의 위치나 설명이 올바르지 않습니다. 다시 생성해 주세요.")
        if tutorial.annotated_code != review_code(source, language, tutorial.line_notes, tutorial.steps):
            raise TutorialError("저장된 주석 코드가 원본 실행 정보와 다릅니다.")
    return tutorial


def output_lines(source: str, language: str) -> set[int]:
    """출력 결과를 묻는 문제에서 핵심 문항의 물리적 행을 확인한다."""
    pattern = {
        "C": r"\b(?:printf|puts|putchar)\s*\(",
        "C++": r"\b(?:printf|puts|putchar)\s*\(|\b(?:std::)?cout\s*<<",
        "Java": r"\bSystem\.out\.(?:print|println|printf)\s*\(",
        "Python": r"\bprint\s*\(",
    }[language]
    return {
        number for number, line in enumerate(source.split("\n"), start=1)
        if not line.lstrip().startswith("#" if language == "Python" else ("//", "/*", "*"))
        and re.search(pattern, line)
    }


def build_trace_step(step: GeneratedTraceStep, lines: list[str], number: int) -> TraceStep:
    """한 문항의 정답·상태·선택지를 대조하고 실제 질문 대상을 확정한다."""
    if not 1 <= step.line_number <= len(lines):
        raise GenerationValidationError(number, "줄 번호가 원본 코드 밖에 있음")
    if not all(value.strip() for value in (step.target, step.answer, step.hint, step.explanation)):
        raise GenerationValidationError(number, "질문 맥락·정답·설명에 빈 값이 있음")
    target = step.target.strip()
    if step.question_kind == "value_after":
        matching = [
            change for change in step.changes
            if normalize_answer(change.after) == normalize_answer(step.answer)
        ]
        if not matching:
            raise GenerationValidationError(number, "정답과 모든 실행 후 값이 다름")
        # 모델의 자유 형식 target보다, 같은 정답을 가진 구조화된 값 변화의 대상을 사용한다.
        selected = next((change for change in matching if change.target.strip() == target), matching[0])
        target = selected.target.strip()
        if (
            selected.before.lstrip("-").isdigit()
            and selected.after.lstrip("-").isdigit()
            and selected.before != selected.after
            and (selected.before not in step.explanation or selected.after not in step.explanation)
        ):
            raise GenerationValidationError(number, "해설에 값 변화가 빠짐")
    check_calculations(step.changes, number)
    if len(step.distractors) != 2:
        raise ChoiceValidationError(number, "오답 후보 개수가 2개가 아님")
    choices = [step.answer, *step.distractors]
    check_choices(choices, number, step.answer, step.question_kind)
    random.SystemRandom().shuffle(choices)
    return TraceStep(
        line_number=step.line_number,
        question=question_for(step.question_kind, step.line_number, target, step.context),
        choices=choices,
        answer=step.answer,
        hint=step.hint,
        explanation=step.explanation,
        step_number=number,
        code_line=lines[step.line_number - 1],
        question_kind=step.question_kind,
        target=target,
        context=step.context,
        changes=step.changes,
    )


def finalize_generated_tutorial(
    payload: dict[str, Any], language: str, source: str, problem: str = ""
) -> Tutorial:
    """모델이 판단한 줄 번호를 검증하고, 순서와 원문은 앱에서 확정한다."""
    if isinstance(payload.get("steps"), list) and not 1 <= len(payload["steps"]) <= 7:
        raise GenerationValidationError(0, "문항 수가 1~7개가 아님")
    try:
        generated = GeneratedTutorial.model_validate(payload)
    except ValidationError as exc:
        raise GenerationValidationError(0, "필수 단계 정보 또는 자료형이 올바르지 않음") from exc
    lines = source.split("\n")
    if len(generated.line_notes) > 20:
        raise GenerationValidationError(0, "행별 메모가 20개를 초과함")
    if sum(bool(line.strip()) for line in lines) > 4 and not generated.line_notes:
        raise GenerationValidationError(1, "핵심 행별 메모가 없음")
    steps = []
    prints = output_lines(source, language) if re.search(r"출력|output", problem, re.I) else set()
    excluded = 0
    seen_questions: set[tuple[int, str, str, str]] = set()
    for number, step in enumerate(generated.steps, start=1):
        try:
            candidate = build_trace_step(step, lines, number)
            identity = (
                candidate.line_number, candidate.question_kind,
                candidate.target, candidate.context.strip(),
            )
            if identity in seen_questions:
                raise GenerationValidationError(number, "같은 시점의 질문이 중복됨")
        except GenerationValidationError as exc:
            if not prints or step.question_kind != "value_after":
                raise
            excluded += 1
            LOGGER.warning("불일치 보조 문항 제외: %s", json.dumps(exc.diagnostics, ensure_ascii=False))
            continue
        seen_questions.add(identity)
        candidate.step_number = len(steps) + 1
        steps.append(candidate)
    if not steps:
        raise GenerationValidationError(0, "검증을 통과한 문항이 없음")
    if prints:
        if prints and not any(
            step.question_kind == "diagnostic"
            or (step.question_kind in ("output_this_step", "output_so_far") and step.line_number in prints)
            for step in steps
        ):
            raise GenerationValidationError(0, "요구된 출력 또는 실행 오류에 대한 핵심 문항이 없음")
    return validate_tutorial(
        Tutorial(
            language=generated.language,
            steps=steps,
            annotated_code=review_code(source, language, generated.line_notes, steps),
            schema_version=QUIZ_SCHEMA_VERSION,
            line_notes=generated.line_notes,
            excluded_steps=excluded,
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
    if code == 400:
        body = getattr(exc, "body", None)
        error = body.get("error", body) if isinstance(body, dict) else {}
        if isinstance(error, dict):
            detail = str(error.get("message") or "").lower()
            parameter = str(error.get("param") or "").lower()
            if "schema" in detail or "schema" in parameter:
                return "앱의 OpenAI 응답 형식에 문제가 있습니다. 아래 오류 상세를 확인해 주세요."
            if error.get("code") == "model_not_found" or parameter == "model":
                return "OpenAI 모델 이름 또는 접근 권한을 확인해 주세요."
        return "OpenAI가 요청을 거부했습니다(400). 아래 오류 상세를 확인해 주세요."
    if code == 404:
        return "OpenAI 모델 이름 또는 접근 권한을 확인해 주세요."
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
    """값·선택지 불일치에만 한 번 더 요청하고 검증된 튜토리얼을 반환한다."""
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
                    else parse_json_object(getattr(response, "output_text", ""))
                )
                return finalize_generated_tutorial(payload, language, source, problem)
            except TutorialError as original_error:
                exc = (
                    original_error if isinstance(original_error, GenerationValidationError)
                    else GenerationValidationError(0, str(original_error))
                )
                diagnostics = {
                    **exc.diagnostics,
                    "model": model,
                    "attempt": attempt + 1,
                    "response_id": getattr(response, "id", None) or "",
                    "elapsed_seconds": round(time.perf_counter() - started, 1),
                }
                LOGGER.warning("생성 내용 검증 실패: %s", json.dumps(diagnostics, ensure_ascii=False))
                if attempt:
                    exc.diagnostics["retry_count"] = 1
                    raise
                if on_retry:
                    on_retry()
                prompt += (
                    f"\n\n앞선 응답의 {exc.diagnostics['step_number']}번 문항 또는 전체 결과에서 "
                    f"{exc.diagnostics['reason']} 오류가 있어 폐기했습니다. "
                    "원본 소스를 다시 추적하고, 정답·값 변화·계산·서로 다른 오답 2개가 "
                    "모두 일치하는 전체 결과를 새로 작성하세요."
                )
        raise AssertionError("재시도 횟수를 초과했습니다.")
    except TutorialError:
        raise
    except Exception as exc:
        diagnostics = api_error_diagnostics(exc, api_key, model)
        diagnostics["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        LOGGER.error("OpenAI 요청 실패: %s", json.dumps(diagnostics, ensure_ascii=False))
        raise TutorialError(describe_api_error(exc), diagnostics) from None


def run_generation_job(api_key: str, model: str, language: str, problem: str, source: str) -> dict[str, Any]:
    """스레드에서 오류를 자료로 바꿔 Streamlit 재실행 뒤에도 정확한 사유를 보존한다."""
    try:
        tutorial = generate_tutorial(api_key, model, language, problem, source)
    except TutorialError as exc:
        return {"tutorial": None, "error": str(exc), "diagnostics": exc.diagnostics}
    return {"tutorial": tutorial.model_dump(), "error": None, "diagnostics": None}


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
        "awaiting_next": state.get("awaiting_next", False),
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
    state["awaiting_next"] = progress["awaiting_next"]
    state["outcomes"] = progress["outcomes"]
    state["needs_reload"] = False


def open_hint(state: Any) -> None:
    """힌트를 저장해 다른 기기에서도 패스 조건을 유지한다."""
    progress = current_progress(state)
    if progress["current_step_idx"] >= len(progress["outcomes"]) or progress["hint_opened"] or progress["awaiting_next"]:
        return
    progress["hint_opened"] = True
    commit_progress(state, progress)


def advance_step(state: Any, *, passed: bool = False) -> bool:
    """힌트를 열었을 때만 패스하고 한 번만 다음 단계로 이동한다."""
    steps = state["quiz_data"]["steps"]
    index = state["current_step_idx"]
    if index >= len(steps) or state.get("awaiting_next") or (passed and not state["hint_opened"]):
        return False
    progress = current_progress(state)
    progress["outcomes"][index]["status"] = "passed" if passed else "correct"
    progress["current_step_idx"] = index + 1
    progress["hint_opened"] = False
    progress["awaiting_next"] = True
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


def acknowledge_result(state: Any) -> bool:
    """해설을 확인한 후에만 다음 문항이나 완료 화면을 연다."""
    if not state.get("awaiting_next"):
        return False
    progress = current_progress(state)
    progress["awaiting_next"] = False
    commit_progress(state, progress)
    state["last_result"] = None
    return True


def submit_answer(state: Any, answer: str) -> bool:
    """오답에서는 현재 단계와 힌트 상태를 유지한다."""
    index = state["current_step_idx"]
    steps = state["quiz_data"]["steps"]
    if index >= len(steps) or state.get("awaiting_next"):
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
        "awaiting_next": False,
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
        "generation_job": None,
        "generation_error": None,
        "prefill_generation": None,
        "delete_candidate": None,
        "needs_reload": False,
        "archive_limit": 100,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_study(state: Any) -> None:
    """계정 변경이나 현재 기록 삭제 때 이전 사용자 내용을 지운다."""
    job = state.get("generation_job")
    if job:
        job["future"].cancel()
        job["executor"].shutdown(wait=False, cancel_futures=True)
    for key, value in {
        "quiz_data": None,
        "current_step_idx": 0,
        "hint_opened": False,
        "awaiting_next": False,
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
        "generation_job": None,
        "generation_error": None,
        "prefill_generation": None,
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
    state["awaiting_next"] = progress.get("awaiting_next", False)
    state["outcomes"] = deepcopy(progress["outcomes"])
    state["active_record_id"] = record_id
    state["active_version"] = version
    state["study_problem"] = problem
    state["study_language"] = language
    state["study_source"] = source
    state["feedback"] = None
    state["completed_celebrated"] = (
        progress["current_step_idx"] >= len(tutorial.steps)
        and not state["awaiting_next"]
    )
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
        show_changes(step)


def show_changes(step: dict[str, Any]) -> None:
    """생성 때 저장한 같은 값 변화 근거를 모든 복습 화면에 재사용한다."""
    for change in step.get("changes", []):
        st.write(f"**{change['target']}:** {change['before']} → {change['after']}")
        if change.get("calculations"):
            st.code(
                " → ".join(
                    f"{item['left']} {item['operator']} {item['right']} = {item['result']}"
                    for item in change["calculations"]
                ),
                language="text",
            )


def show_reload_control() -> None:
    if st.session_state.get("needs_reload") and st.button("최신 풀이 상태 불러오기"):
        try:
            record = make_store().get_tutorial(st.session_state["owner_id"], st.session_state["active_record_id"])
            apply_record(st.session_state, record)
            st.session_state["needs_reload"] = False
            st.rerun()
        except StorageError as exc:
            st.error(str(exc))


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
    st.caption(f"학습 형식 v{quiz.get('schema_version', 1)}")
    if quiz.get("excluded_steps"):
        st.caption(f"값·정답이 맞지 않는 보조 문항 {quiz['excluded_steps']}개를 제외했습니다.")
    if quiz.get("schema_version", 1) < QUIZ_SCHEMA_VERSION:
        st.warning("기존 방식으로 만든 문제입니다. 새 값 변화 설명을 적용하려면 아래 버튼으로 새 문제를 준비하세요.")
        if st.button("이 코드로 새 형식 생성 준비"):
            st.session_state["prefill_generation"] = {
                "language": language,
                "problem": st.session_state["study_problem"],
                "source": st.session_state["study_source"],
            }
            st.rerun()
    if st.session_state["generation_seconds"] is not None:
        st.caption(f"생성 소요 시간: {st.session_state['generation_seconds']:.1f}초")

    # 마지막 정답도 해설을 먼저 보고 직접 완료 화면으로 간다.
    awaiting_next = st.session_state.get("awaiting_next", False)
    if index >= len(steps) and not awaiting_next:
        if not st.session_state["completed_celebrated"]:
            st.balloons()
            st.session_state["completed_celebrated"] = True
        st.success("모든 단계를 완료했습니다! 아래 주석 코드를 복습해 보세요.")
        st.code(quiz["annotated_code"], language=LANGUAGES[language], line_numbers=True)
        show_review_history(steps, len(steps), language)
        show_reload_control()
        return

    shown_index = index - 1 if awaiting_next else index
    step = steps[shown_index]
    st.progress((shown_index + 1) / len(steps), text=f"{shown_index + 1} / {len(steps)}단계")
    st.html(highlight_source(st.session_state["study_source"], language, step["line_number"]))
    st.write(f"**현재 {step['line_number']}행:** {step['question']}")

    if awaiting_next:
        passed = st.session_state["outcomes"][shown_index]["status"] == "passed"
        st.success("패스했습니다. 정답과 해설을 확인하세요." if passed else "맞았습니다!")
        st.write(f"**정답:** {step['answer']}")
        st.write(f"**해설:** {step['explanation']}")
        show_changes(step)
        for note in quiz.get("line_notes", []):
            if note["line_number"] == step["line_number"]:
                st.caption(f"{note['line_number']}행 실행 메모: {note['note']}")
        label = "학습 마치기" if index == len(steps) else "다음 문제"
        if st.button(label, key=f"next_{st.session_state['generation_count']}_{shown_index}", width="stretch"):
            try:
                if acknowledge_result(st.session_state):
                    st.rerun()
            except StorageError as exc:
                st.error(str(exc))
                if isinstance(exc, StorageConflict):
                    st.session_state["needs_reload"] = True
        show_review_history(steps, shown_index, language)
        show_reload_control()
        return

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

    show_review_history(steps, index, language)
    show_reload_control()


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


@st.fragment(run_every="1s")
def show_generation_status(store: ArchiveStore, owner: str) -> None:
    """백그라운드 API 응답을 확인하고 검증된 결과만 보관한다."""
    job = st.session_state.get("generation_job")
    if job:
        if job["owner"] != owner:
            return
        if not job["future"].done():
            st.info("실행 흐름을 분석하고 핵심 문제를 만들고 있습니다. 다른 화면을 사용해도 됩니다.")
            return
        st.session_state["generation_job"] = None
        job["executor"].shutdown(wait=False)
        try:
            result = job["future"].result()
        except Exception:
            LOGGER.error("생성 작업 처리 실패: %s", job["model"])
            st.session_state["generation_error"] = {
                "message": "문제 생성 작업에 실패했습니다. 다시 시도해 주세요.",
                "diagnostics": None,
                "elapsed_seconds": round(time.perf_counter() - job["started"], 1),
            }
            st.rerun()
        else:
            if result["error"]:
                st.session_state["generation_error"] = {
                    "message": result["error"],
                    "diagnostics": result["diagnostics"],
                    "elapsed_seconds": round(time.perf_counter() - job["started"], 1),
                }
                st.rerun()
            record = new_tutorial_record(
                job["problem"], job["language"], job["source"], job["model"], result["tutorial"]
            )
            st.session_state["pending_generation"] = record
            try:
                saved = store.insert_tutorial(owner, record)
                apply_record(st.session_state, saved)
            except (StorageError, TutorialError) as exc:
                st.error(str(exc))
            else:
                st.session_state["pending_generation"] = None
                st.session_state["generation_seconds"] = time.perf_counter() - job["started"]
                st.session_state["view"] = "학습"
            st.rerun()

    error = st.session_state.get("generation_error")
    if error:
        st.error(error["message"])
        st.caption(f"요청 경과 시간: {error['elapsed_seconds']:.1f}초")
        if error["diagnostics"]:
            with st.expander("생성 오류 상세 (키 제외)"):
                st.json(error["diagnostics"])


def show_generate_view(store: ArchiveStore, owner: str, model: str) -> None:
    """생성 결과가 보관된 뒤에만 현재 학습을 교체한다."""
    prefill = st.session_state.get("prefill_generation")
    if prefill:
        st.session_state[f"language_input_{owner}"] = prefill["language"]
        st.session_state[f"problem_input_{owner}"] = prefill["problem"]
        st.session_state[f"source_input_{owner}"] = prefill["source"]
        st.session_state["prefill_generation"] = None
        st.info("원본 문제와 코드를 입력칸에 채웠습니다. 새 문제를 만들려면 ‘핵심 문제 생성’을 누르세요.")
    with st.form("generate_form"):
        language = st.selectbox("프로그래밍 언어", list(LANGUAGES), key=f"language_input_{owner}")
        problem = st.text_area("문제 설명", placeholder="예: 다음 프로그램의 실행 결과는?", key=f"problem_input_{owner}")
        source = st.text_area("소스 코드", height=260, placeholder="코드를 여기에 붙여 넣으세요.", key=f"source_input_{owner}")
        requested = st.form_submit_button(
            "핵심 문제 생성", disabled=st.session_state["generation_job"] is not None
        )

    if requested:
        if st.session_state["generation_job"]:
            st.info("이미 문제를 생성하고 있습니다.")
        elif st.session_state["pending_generation"]:
            st.error("먼저 생성된 문제를 저장하거나 버려 주세요.")
        elif not problem.strip() or not source.strip():
            st.error("문제 설명과 소스 코드를 모두 입력해 주세요.")
        elif not model.strip():
            st.error("모델 이름을 입력해 주세요.")
        else:
            try:
                api_key = store.get_key(owner)
            except StorageError as exc:
                st.error(str(exc))
            else:
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trace-generation")
                future = executor.submit(run_generation_job, api_key, model.strip(), language, problem, source)
                st.session_state["generation_job"] = {
                    "future": future,
                    "executor": executor,
                    "owner": owner,
                    "language": language,
                    "problem": problem,
                    "source": source,
                    "model": model.strip(),
                    "started": time.perf_counter(),
                }
                st.session_state["generation_error"] = None
                st.rerun()

    show_generation_status(store, owner)
    save_pending_generation(store, owner)
    show_study_view()


def show_archive_view(store: ArchiveStore, owner: str) -> None:
    # 보관함을 보는 동안에도 실행 중인 생성 작업의 완료·실패를 수거한다.
    show_generation_status(store, owner)
    save_pending_generation(store, owner)
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
    st.caption("C · C++ · Java · Python | 앱 버전 2")

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
