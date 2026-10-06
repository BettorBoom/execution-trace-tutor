"""격리 실행으로 확인한 값으로 코드 흐름을 학습하는 Streamlit 앱."""

import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from html import escape
from typing import Any, Literal
from urllib.parse import urlencode

import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from trace_worker import _inline_if_assignment
from verified_trace import (
    MAX_SOURCE_BYTES, ProbePlan, TraceError, plan_prompt, run_isolated_trace,
    validate_probe_plan,
)

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
VERIFIED_SCHEMA_VERSION = 4
MIN_NEW_STEPS = 3
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
    verified_fact: dict[str, Any] | None = None


class Tutorial(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str
    steps: list[TraceStep]
    annotated_code: str
    schema_version: int = 1
    line_notes: list[LineNote] = Field(default_factory=list)
    excluded_steps: int = 0
    execution_verified: bool = False
    stdout: str = ""


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
    if question_kind in ("value_before", "value_after") and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", normalized):
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
    if kind == "value_before":
        return f"{prefix}{line_number}행 실행 직전 `{target}`의 값은 무엇인가요? 선택지에서 값 하나를 고르세요."
    if kind == "output_character":
        return (f"{prefix}{line_number}행에서 `{target}`의 출력 문자는 무엇인가요?"
                if context else f"최종 표준 출력의 {target}는 무엇인가요?")
    if kind == "output_length":
        return "프로그램이 출력한 글자는 모두 몇 개인가요? 줄바꿈도 한 글자로 셉니다."
    if kind == "normal_exit":
        return "격리 실행에서 이 프로그램은 어떻게 종료했나요?"
    if kind == "output_this_step":
        return f"{prefix}{line_number}행의 `{target}`가 이번에 출력하는 내용은 무엇인가요?"
    if kind == "output_so_far":
        return f"{prefix}{line_number}행의 `{target}` 실행 직후까지 누적된 출력은 무엇인가요?"
    if kind == "program_output":
        return "이 프로그램이 정상 종료할 때의 최종 표준 출력은 무엇인가요?"
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
    lines = source.split("\n")
    all_notes = [(item.line_number, item.note) for item in notes]
    for step in steps:
        for change in step.changes:
            detail = f"{step.context}: {change.target} {change.before} → {change.after}"
            if step.verified_fact and _inline_if_assignment(lines[step.line_number - 1]):
                detail += "; 이 회차에는 if 조건이 참이어서 대입 실행"
            if step.verified_fact:
                context_values = step.verified_fact.get("context_values", {})
                if context_values:
                    detail += "; " + ", ".join(
                        f"{expression}={value}" for expression, value in sorted(context_values.items())
                    )
            all_notes.append((
                step.line_number,
                detail,
            ))
        if step.question_kind == "program_output" and step.verified_fact:
            all_notes.append((step.line_number, f"검증된 최종 표준 출력: {step.answer}"))
        if step.question_kind == "output_character" and step.verified_fact:
            label = step.target if step.context else f"전체 표준 출력의 {step.target}"
            all_notes.append((step.line_number, f"{label}: {step.answer} 출력"))
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
            if not step.target.strip() or (step.question_kind in ("value_before", "value_after") and not step.changes):
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
        reviewed = review_code(source, language, tutorial.line_notes, tutorial.steps)
        if tutorial.execution_verified:
            # JSONB는 객체 키 순서를 바꿀 수 있다. 주석은 검증된 사실에서 다시 만든다.
            tutorial.annotated_code = reviewed
        elif tutorial.annotated_code != reviewed:
            raise TutorialError("저장된 주석 코드가 원본 실행 정보와 다릅니다.")
    if tutorial.schema_version >= VERIFIED_SCHEMA_VERSION:
        if not tutorial.execution_verified:
            raise TutorialError("실행 검증 정보가 없는 문제는 채점할 수 없습니다.")
        for step in tutorial.steps:
            fact = step.verified_fact or {}
            if step.question_kind in ("value_before", "value_after"):
                if (fact.get("source") != "probe" or fact.get("line_number") != step.line_number
                    or fact.get("target") != step.target
                    or fact.get("before" if step.question_kind == "value_before" else "after") != step.answer):
                    raise TutorialError("정답과 실행 관측값이 다릅니다.")
                change = next(item for item in step.changes if item.target.strip() == step.target.strip())
                if fact.get("before") != change.before or fact.get("after") != change.after:
                    raise TutorialError("값 변화 설명과 실행 관측값이 다릅니다.")
            elif step.question_kind == "output_character":
                index = fact.get("index")
                if (fact.get("source") != "stdout_char" or type(index) is not int
                    or not 0 <= index < len(tutorial.stdout)
                    or fact.get("line_number") != step.line_number
                    or tutorial.stdout[index] != step.answer):
                    raise TutorialError("출력 문자와 실행 관측값이 다릅니다.")
            elif step.question_kind == "output_length":
                if (fact.get("source") != "stdout_length" or fact.get("line_number") != step.line_number
                    or step.answer != str(len(tutorial.stdout))):
                    raise TutorialError("출력 길이와 실행 관측값이 다릅니다.")
            elif step.question_kind == "normal_exit":
                if fact.get("source") != "execution" or fact.get("status") != "ok" or step.answer != "정상 종료":
                    raise TutorialError("실행 종료 상태가 올바르지 않습니다.")
            elif step.question_kind == "program_output":
                if fact.get("source") != "stdout" or step.answer != display_output(tutorial.stdout):
                    raise TutorialError("정답과 실제 출력이 다릅니다.")
            else:
                raise TutorialError("실행으로 확인되지 않은 문항 종류입니다.")
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


def display_output(stdout: str) -> str:
    """터미널의 관례적인 마지막 개행만 화면에서 생략한다."""
    if not stdout:
        return "(출력 없음)"
    if stdout.endswith("\n") and stdout.count("\n") == 1 and stdout[:-1].strip() == stdout[:-1]:
        return stdout[:-1]
    if stdout and "\n" not in stdout and stdout.strip() == stdout:
        return stdout
    return json.dumps(stdout, ensure_ascii=False)


def archive_time(value: str) -> str:
    """서버의 UTC 저장 시각을 한국 사용자에게 익숙한 시각으로 표시한다."""
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if moment.tzinfo is None:
            return str(value)[:16]
        return moment.astimezone(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")
    except (AttributeError, ValueError):
        return str(value)[:16]


def verified_choices(answer: str, numeric: bool) -> list[str]:
    """모델이 정답 선택지를 정하지 못하게 관측값에서 오답을 만든다."""
    if numeric and re.fullmatch(r"-?\d+", answer):
        number = int(answer)
        choices = [str(number - 1), answer, str(number + 1)]
    else:
        try:
            literal = json.loads(answer)
        except (ValueError, TypeError):
            literal = None
        text_value = literal if isinstance(literal, str) else answer
        if text_value:
            match = re.search(r"\d(?!.*\d)", text_value, re.DOTALL)
            if len(text_value) == 1 and text_value.isascii() and text_value.isalpha():
                first = ord("A" if text_value.isupper() else "a")
                offset = ord(text_value) - first
                shifts = (1, 2) if offset == 0 else ((-2, -1) if offset == 25 else (-1, 1))
                variants = [chr(ord(text_value) + shift) for shift in shifts]
            elif match:
                digit = int(match.group())
                variants = [text_value[:match.start()] + str((digit + offset) % 10) + text_value[match.end():]
                            for offset in (-1, 1)]
            elif text_value.endswith("\n"):
                variants = [text_value[:-1], text_value + "\n"]
            else:
                variants = ([text_value[:-1], text_value + text_value[-1]]
                            if len(text_value) > 1 else [text_value * 2, text_value * 3])
            encode = (lambda value: json.dumps(value, ensure_ascii=False)) if isinstance(literal, str) else str
            choices = [answer, *(encode(value) for value in variants)]
        else:
            choices = [answer, answer + "?", answer + "!"]
    random.SystemRandom().shuffle(choices)
    return choices


def build_verified_tutorial(
    language: str, source: str, probes: list[dict[str, Any]], observations: list[dict[str, Any]], stdout: str
) -> Tutorial:
    """모델의 숫자는 받지 않고, 실행 결과만 정답·주석의 근거로 쓴다."""
    lines = source.split("\n")
    selected: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for probe_id, probe in enumerate(probes):
        events = [item for item in observations if item.get("id") == probe_id]
        if not events:
            continue
        changed = [item for item in events if item.get("before") != item.get("after")]
        changed = [item for item in changed if re.fullmatch(r"-?\d+", str(item.get("after", "")))]
        if not changed:
            continue
        candidates.extend(changed)
        def importance(event: dict[str, Any]) -> tuple[int, int]:
            try:
                delta = abs(int(event["after"]) - int(event["before"]))
            except (TypeError, ValueError):
                delta = 1
            return (delta, -int(event["occurrence"]))
        selected.append(max(changed, key=importance))
    # 같은 실행 행의 같은 회차를 별칭·공백만 바꿔 다시 묻지 않는다.
    unique_events: dict[tuple[int, int], dict[str, Any]] = {}
    for event in sorted(selected, key=lambda item: len(item.get("context_values", {})), reverse=True):
        unique_events.setdefault((int(event["line_number"]), int(event["occurrence"])), event)
    # 한 줄이 여러 번 실행됐다면 서로 다른 값 변화·실행 조건을 추가로 묻는다.
    # 같은 전후 값과 조건을 되풀이하는 회차는 문항 수를 채우려고 복제하지 않는다.
    # 조건이 참일 때만 갱신되는 값은 각 갱신이 학습의 핵심일 수 있다.
    conditional_changes = {
        (item["line_number"], item["target"], item["before"], item["after"])
        for item in candidates
        if re.match(r"\s*if\s*\(", lines[int(item["line_number"]) - 1])
    }
    needed_values = min(6, max(MIN_NEW_STEPS - bool(stdout), min(4, len(conditional_changes))))
    ordered_candidates = sorted(candidates, key=lambda item: int(item.get("event_index", item["line_number"])))
    for require_new_change in (True, False):
        for event in ordered_candidates:
            if len(unique_events) >= needed_values:
                break
            identity = (int(event["line_number"]), int(event["occurrence"]))
            similar = [
                item for item in unique_events.values()
                if item["line_number"] == event["line_number"]
                and item["target"] == event["target"]
                and item["before"] == event["before"]
                and item["after"] == event["after"]
            ]
            if identity in unique_events or (similar and (
                require_new_change or any(item.get("context_values", {}) == event.get("context_values", {})
                                          for item in similar)
            )):
                continue
            unique_events[identity] = event
    selected = sorted(unique_events.values(), key=lambda item: int(item.get("event_index", item["line_number"])))

    steps: list[TraceStep] = []
    for event in selected[:6]:
        before, after = str(event["before"]), str(event["after"])
        line_number, target = int(event["line_number"]), str(event["target"])
        occurrence = int(event["occurrence"])
        # 반복 변수 등 실제 관측한 실행 직전 조건을 질문에도 보여 준다.
        conditions = [
            f"{expression}={value}"
            for expression, value in event.get("context_values", {}).items()
            if expression != target and re.fullmatch(r"[A-Za-z_]\w*", expression)
            and re.fullmatch(r"-?\d+", str(value))
        ]
        conditional_assignment = _inline_if_assignment(lines[line_number - 1]) is not None
        context = f"조건이 참이 된 {occurrence}번째 대입 실행" if conditional_assignment else f"{occurrence}번째 실행"
        if conditions:
            context += " (실행 직전 " + ", ".join(conditions) + ")"
        change = StateChange(target=target, before=before, after=after, calculations=[])
        explanation = (
            f"{line_number}행에서 `if` 조건이 참이 된 {occurrence}번째 회차에 "
            if conditional_assignment else f"{line_number}행의 {occurrence}번째 실행에서 "
        ) + f"`{target}`의 값이 {before} → {after}로 바뀝니다."
        context_values = [
            f"`{expression}` = {value}"
            for expression, value in event.get("context_values", {}).items()
            if re.fullmatch(r"-?\d+", str(value))
        ]
        if context_values:
            explanation += " 실행 직전에 확인한 계산값: " + ", ".join(context_values) + "."
        steps.append(TraceStep(
            line_number=line_number,
            question=question_for("value_after", line_number, target, context),
            choices=verified_choices(after, numeric=True), answer=after,
            hint=f"실행 직전 `{target}`의 값은 {before}입니다. 대입식의 오른쪽을 계산해 보세요.",
            explanation=explanation, step_number=len(steps) + 1,
            code_line=lines[line_number - 1], question_kind="value_after",
            target=target, context=context, changes=[change],
            verified_fact={"source": "probe", **event},
        ))

    printed_lines = output_lines(source, language)
    line_number = max(printed_lines) if printed_lines else next(
        (number for number in range(len(lines), 0, -1) if lines[number - 1].strip()), 1
    )
    answer = display_output(stdout)
    if stdout or len(steps) < MIN_NEW_STEPS:
        steps.append(TraceStep(
            line_number=line_number,
            question=question_for("program_output", line_number, "표준 출력", ""),
            choices=(verified_choices(answer, numeric=bool(re.fullmatch(r"-?\d+", answer)))
                     if stdout else ["(출력 없음)", "0", "빈 줄 1개"]),
            answer=answer,
            hint="이전 행의 값 변화를 반영해 최종 출력을 확인해 보세요.",
            explanation=f"격리 환경에서 원본 코드를 실행해 확인한 표준 출력은 {answer}입니다.",
            step_number=len(steps) + 1, code_line=lines[line_number - 1],
            question_kind="program_output", target="표준 출력", context="",
            verified_fact={"source": "stdout", "line_number": line_number},
        ))
    # 문항이 부족하면 같은 답을 복제하지 않고 실행 직전 값과 출력 문자를 묻는다.
    for step in list(steps):
        if len(steps) >= MIN_NEW_STEPS:
            break
        if step.question_kind != "value_after":
            continue
        fact = step.verified_fact or {}
        before = str(fact.get("before", ""))
        if not re.fullmatch(r"-?\d+", before):
            continue
        steps.insert(steps.index(step), TraceStep(
            line_number=step.line_number,
            question=question_for("value_before", step.line_number, step.target, step.context),
            choices=verified_choices(before, numeric=True), answer=before,
            hint=f"이 행을 실행하면 `{step.target}`의 값은 {step.answer}가 됩니다. 대입 전 값을 찾아보세요.",
            explanation=f"{step.line_number}행의 {step.context} 전에 `{step.target}`은 {before}이고, 실행 후 {step.answer}로 바뀝니다.",
            step_number=0, code_line=step.code_line, question_kind="value_before",
            target=step.target, context=step.context, changes=step.changes,
            verified_fact=fact,
        ))
    if len(steps) < MIN_NEW_STEPS and stdout:
        # 정확히 두 번의 %c 출력이면 화면에 실제 포인터 표현식을 보여 준다.
        char_args = None
        if language == "C" and len(stdout) == 2 and printed_lines == {line_number}:
            char_args = re.fullmatch(
                r'\s*printf\s*\(\s*"%c%c"\s*,\s*([^,]+?)\s*,\s*([^,]+?)\s*\)\s*;\s*',
                lines[line_number - 1],
            )
        for index, character in enumerate(stdout):
            if len(steps) >= MIN_NEW_STEPS or not character.isprintable() or character.isspace():
                continue
            target = char_args.group(index + 1).strip() if char_args and index < 2 else f"{index + 1}번째 문자"
            context = f"{index + 1}번째 %c" if char_args and index < 2 else ""
            pointer_move = re.fullmatch(r"\*\(\s*([A-Za-z_]\w*)\s*([+-])\s*(\d+)\s*\)", target)
            explanation = f"격리 실행에서 {index + 1}번째로 출력된 문자는 {character}입니다."
            if pointer_move:
                direction = "오른쪽" if pointer_move.group(2) == "+" else "왼쪽"
                explanation = (
                    f"`{pointer_move.group(1)}`가 가리키는 위치에서 {pointer_move.group(3)}칸 "
                    f"{direction}의 문자를 읽습니다. 실제 출력 문자는 {character}입니다."
                )
            steps.insert(-1, TraceStep(
                line_number=line_number,
                question=question_for("output_character", line_number, target, context),
                choices=verified_choices(character, numeric=False), answer=character,
                hint="이 출력문의 인자를 순서대로 따라가 보세요." if char_args else "실제 출력의 문자 순서를 확인해 보세요.",
                explanation=explanation,
                step_number=0, code_line=lines[line_number - 1],
                question_kind="output_character", target=target, context=context,
                verified_fact={"source": "stdout_char", "index": index, "line_number": line_number},
            ))
    if len(steps) < MIN_NEW_STEPS:
        count = len(stdout)
        choices = [str(count), str(count + 1), str(count + 2)]
        random.SystemRandom().shuffle(choices)
        steps.insert(-1, TraceStep(
            line_number=line_number, question=question_for("output_length", line_number, "표준 출력", ""),
            choices=choices, answer=str(count),
            hint="화면에 보이지 않는 줄바꿈도 출력된 글자에 포함합니다.",
            explanation=f"격리 실행의 표준 출력은 줄바꿈을 포함해 {count}글자입니다.",
            step_number=0, code_line=lines[line_number - 1], question_kind="output_length",
            target="표준 출력", verified_fact={"source": "stdout_length", "line_number": line_number},
        ))
    if len(steps) < MIN_NEW_STEPS:
        steps.insert(-1, TraceStep(
            line_number=line_number, question=question_for("normal_exit", line_number, "실행", ""),
            choices=["정상 종료", "실행 오류", "컴파일 오류"], answer="정상 종료",
            hint="원본 코드의 격리 실행 결과를 확인해 보세요.",
            explanation="격리 실행에서 원본 코드가 오류 없이 정상 종료했습니다.",
            step_number=0, code_line=lines[line_number - 1], question_kind="normal_exit",
            target="실행", verified_fact={"source": "execution", "status": "ok"},
        ))
    for number, step in enumerate(steps, 1):
        step.step_number = number
    tutorial = Tutorial(
        language=language, steps=steps, annotated_code=review_code(source, language, [], steps),
        schema_version=VERIFIED_SCHEMA_VERSION,
        execution_verified=True, stdout=stdout,
    )
    return validate_tutorial(tutorial.model_dump(), language, source)


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
    sandbox_settings: dict[str, str] | None = None,
) -> Tutorial:
    """모델은 관측 위치만 고르고, 정답은 격리 실행 결과에서 만든다."""
    if not sandbox_settings or not all(
        sandbox_settings.get(name, "").strip() for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")
    ):
        raise TutorialError("실행 검증용 샌드박스가 아직 설정되지 않았습니다. 관리자에게 문의해 주세요.")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise TutorialError("코드가 너무 깁니다. 24KB 이하로 줄여 주세요.")
    started = time.perf_counter()
    try:
        client = OpenAI(api_key=api_key, timeout=60.0, max_retries=0)
        response = client.responses.parse(
            model=model, input=plan_prompt(language, problem, source),
            text_format=ProbePlan, store=False,
        )
    except Exception as exc:
        diagnostics = api_error_diagnostics(exc, api_key, model)
        diagnostics["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        LOGGER.error("OpenAI 요청 실패: %s", json.dumps(diagnostics, ensure_ascii=False))
        raise TutorialError(describe_api_error(exc), diagnostics) from None

    try:
        if response.status == "incomplete":
            raise TutorialError("OpenAI 응답이 중간에 끊겼습니다. 코드를 줄여 다시 생성해 주세요.")
        try:
            payload = (
                response.output_parsed.model_dump() if response.output_parsed is not None
                else parse_json_object(getattr(response, "output_text", ""))
            )
            plan = ProbePlan.model_validate(payload)
        except ValidationError as exc:
            raise TutorialError("모델이 확인할 코드 행을 올바르게 제안하지 못했습니다.") from exc
        probes = validate_probe_plan(plan, source, language)
        observed = run_isolated_trace(language, source, probes, sandbox_settings)
        tutorial = build_verified_tutorial(language, source, probes, observed["observations"], observed["stdout"])
        return tutorial
    except TutorialError:
        raise
    except TraceError as exc:
        raise TutorialError(str(exc)) from None
    except Exception as exc:
        # 모델 파싱·격리 실행·문항 조립 오류는 API 장애로 잘못 안내하지 않는다.
        LOGGER.error("튜토리얼 조립 오류: %s", type(exc).__name__)
        raise TutorialError(
            "앱 내부 오류로 튜토리얼을 만들지 못했습니다. 개발자에게 문의해 주세요.",
            {"error_type": type(exc).__name__},
        ) from None


def run_generation_job(
    api_key: str, model: str, language: str, problem: str, source: str,
    sandbox_settings: dict[str, str] | None = None,
) -> dict[str, Any]:
    """스레드에서 오류를 자료로 바꿔 Streamlit 재실행 뒤에도 정확한 사유를 보존한다."""
    try:
        tutorial = generate_tutorial(api_key, model, language, problem, source, sandbox_settings)
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
        "pending_save_error": None,
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
        "pending_save_error": None,
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
        LOGGER.error("저장 기록 검증 실패 (%s): %s", type(exc).__name__, str(exc)[:300])
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
    if quiz.get("schema_version", 1) < VERIFIED_SCHEMA_VERSION:
        st.warning(
            "이 기록은 이전 검증 방식으로 만들었습니다. "
            "정답이 틀릴 수 있어 채점을 중단했습니다. 원본으로 검증된 새 문제를 생성해 주세요."
        )
        st.code(st.session_state["study_source"], language=LANGUAGES[language], line_numbers=True)
        if st.button("이 코드로 검증된 문제 생성 준비"):
            st.session_state["prefill_generation"] = {
                "language": language,
                "problem": st.session_state["study_problem"],
                "source": st.session_state["study_source"],
            }
            st.rerun()
        return
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
    if st.session_state.get("pending_save_error"):
        st.error(st.session_state["pending_save_error"])
    if st.button("생성된 문제 저장 다시 시도"):
        try:
            record = store.insert_tutorial(owner, pending)
            apply_record(st.session_state, record)
        except (StorageError, TutorialError) as exc:
            st.session_state["pending_save_error"] = str(exc)
            st.error(str(exc))
        else:
            st.session_state["pending_generation"] = None
            st.session_state["pending_save_error"] = None
            st.rerun()
    if st.button("저장하지 않고 버리기"):
        st.session_state["pending_generation"] = None
        st.session_state["pending_save_error"] = None
        st.rerun()


@st.fragment(run_every="1s")
def show_generation_status(store: ArchiveStore, owner: str) -> None:
    """백그라운드 API 응답을 확인하고 검증된 결과만 보관한다."""
    job = st.session_state.get("generation_job")
    if job:
        if job["owner"] != owner:
            return
        if not job["future"].done():
            st.info("핵심 행을 고른 뒤 격리 실행으로 정답을 검증하고 있습니다. 다른 화면을 사용해도 됩니다.")
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
                st.session_state["pending_save_error"] = str(exc)
            else:
                st.session_state["pending_generation"] = None
                st.session_state["pending_save_error"] = None
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
    sandbox_settings = {
        "MODAL_TOKEN_ID": setting("MODAL_TOKEN_ID"),
        "MODAL_TOKEN_SECRET": setting("MODAL_TOKEN_SECRET"),
    }
    sandbox_ready = all(sandbox_settings.values())
    if not sandbox_ready:
        st.warning("실행 검증용 샌드박스 설정이 아직 없습니다. 관리자 설정 후 새 문제를 생성할 수 있습니다.")
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
        st.caption(
            "입력한 코드는 OpenAI에 출제 지점 선별용으로 전송되고, Modal 격리 샌드박스에서 실행됩니다. "
            "정상 실행이 확인된 코드에서 3~7문항을 만듭니다."
        )
        requested = st.form_submit_button(
            "핵심 문제 생성", disabled=st.session_state["generation_job"] is not None or not sandbox_ready
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
                future = executor.submit(
                    run_generation_job, api_key, model.strip(), language, problem, source,
                    sandbox_settings,
                )
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
            st.caption(f"마지막 학습: {archive_time(summary.get('updated_at', ''))}")
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
                    if original["quiz"].get("schema_version", 1) < VERIFIED_SCHEMA_VERSION:
                        st.session_state["prefill_generation"] = {
                            "problem": original["problem"],
                            "language": original["language"],
                            "source": original["source"],
                        }
                        st.session_state["view"] = "학습"
                        st.rerun()
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
    st.caption("C · C++ · Java · Python | 앱 버전 4.8 · 실행 검증")

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
