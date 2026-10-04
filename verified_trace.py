"""격리 실행에서 관측한 값만 새 문제의 정답으로 사용한다."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from trace_worker import _assignment_line_is_safe, _python_line_is_safe


MAX_SOURCE_BYTES = 24_000
MAX_PROBES = 6
WORKER = Path(__file__).with_name("trace_worker.py")


class Probe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line_number: int
    target: str
    reason: str
    context_exprs: list[str] = Field(default_factory=list)


class ProbePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probes: list[Probe]


class TraceError(Exception):
    """사용자에게 표시해도 되는 실행 검증 실패."""


def plan_prompt(language: str, problem: str, source: str) -> str:
    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(source.split("\n"), 1))
    return f"""아래 {language} 코드를 공부할 핵심 지점을 최대 6개 고르세요.
정답 값이나 해설은 만들지 마세요. 실제 실행으로 확인할 읽기 전용 표현식만 제안하세요.
최종 출력, 포인터/참조/배열, 함수 부수 효과, 분기와 반복의 핵심 변화를 우선하세요.
단순 상수 초기화, 중복 질문, 단독 중괄호, 주석은 제외하세요.
같은 실행 행의 같은 값을 공백이나 표기만 달리해 두 번 제안하지 마세요.
각 probe의 line_number는 실행 직후 값을 볼 단일 행 대입문의 물리적 줄 번호입니다.
target은 그 행의 대입문 왼쪽 대상인 정수 변수/배열 원소/역참조 표현식입니다.
오른쪽 계산식 자체를 target으로 제안하지 마세요. 계산식은 context_exprs에 넣으세요.
실행 직전에도 이미 선언되고 초기화되어 있는 대상만 고르세요.
int x=3, int* p=arr, int** pp=&p 같은 선언문은 제외하세요. 포인터 주소 자체도 대상이 아닙니다.
포인터 문제라면 *(*arr+i)=... 같은 실제 원소 대입과 num=arr[2] 같은 결과 대입을 고르세요.
context_exprs에는 우변 계산을 설명하는 읽기 전용 정수 표현식 0~4개를 넣으세요.
예를 들어 (arr[i]+i)%size라면 arr[i], i, size, arr[i]+i를 제안할 수 있습니다.
함수 호출, ++/--, 대입, 문자열 접근, 부작용 있는 표현식은 target에 쓰지 마세요.
reason에는 출제 이유를 쉬운 한국어로 한 문장만 적고 정답은 적지 마세요.
실행 오류나 입력 미지정으로 값을 확정할 수 없으면 probes를 빈 배열로 두세요.

문제: {problem}
원본 소스:
{numbered}"""


def _valid_expression(expr: str, language: str) -> bool:
    if not expr or len(expr) > 90:
        return False
    # 표현식에 호출, 대입, 증감, 문자열, 임의 문장을 섞어 넣을 수 없다.
    if not re.fullmatch(r"[\w\s*+()\[\].>%-]+", expr, re.ASCII) or any(
        token in expr for token in ("++", "--", "->", "/*", "*/")
    ):
        return False
    if language == "Python":
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            return False
        allowed = (ast.Expression, ast.Name, ast.Load, ast.Subscript, ast.Constant,
                   ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Mod,
                   ast.UnaryOp, ast.USub)
        return all(isinstance(node, allowed) for node in ast.walk(tree))
    return not re.search(r"\b[A-Za-z_]\w*\s*\(", expr)


def _output_assignment(source: str, language: str) -> tuple[int, str, str] | None:
    """단순 출력 변수의 마지막 대입을 놓쳤다면 검증 후보로 보탠다."""
    output_pattern = {
        "C": r'\bprintf\s*\(\s*"[^"]*"\s*,\s*([A-Za-z_]\w*)\s*\)',
        "C++": r'\b(?:std::)?cout\s*<<\s*([A-Za-z_]\w*)\b',
        "Java": r'\bSystem\.out\.(?:print|println)\s*\(\s*([A-Za-z_]\w*)\s*\)',
        "Python": r'\bprint\s*\(\s*([A-Za-z_]\w*)\s*\)',
    }[language]
    lines = source.split("\n")
    for output_line in range(len(lines), 0, -1):
        match = re.search(output_pattern, lines[output_line - 1])
        if not match:
            continue
        target = match.group(1)
        suffix = "" if language == "Python" else ";"
        assignment = re.compile(rf"^\s*{re.escape(target)}\s*(?:[+*/%-]?=(?!=))\s*(.+?){suffix}\s*$")
        for line_number in range(output_line - 1, 0, -1):
            assigned = assignment.fullmatch(lines[line_number - 1])
            if assigned:
                return line_number, target, assigned.group(1).strip()
    return None


def _assigned_target(source: str, line_number: int, language: str) -> tuple[str, str] | None:
    """계측 가능한 대입문의 왼쪽 대상과 오른쪽 식을 원본에서 읽는다."""
    code = source.split("\n")[line_number - 1].strip()
    if language == "Python":
        if not _python_line_is_safe(source, line_number):
            return None
        node = ast.parse(code).body[0]
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1:
            return None
        left, right = ast.unparse(targets[0]), ast.unparse(node.value)
    else:
        if not _assignment_line_is_safe(code):
            return None
        operator = re.search(r"(?<![=!<>])(?:[+*/%&|^-]|<<|>>)?=(?!=)", code)
        if operator is None:
            return None
        left, right = code[:operator.start()].strip(), code[operator.end():].rstrip(";").strip()
    return (left, right) if _valid_expression(left, language) else None


def validate_probe_plan(plan: ProbePlan, source: str, language: str) -> list[dict[str, Any]]:
    """모델은 위치와 표현식만 제안하며 실행 가능 여부는 워커가 검사한다."""
    lines = source.split("\n")
    if len(plan.probes) > MAX_PROBES:
        raise TraceError("핵심 지점이 6개를 초과했습니다. 다시 생성해 주세요.")
    probes: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for item in plan.probes:
        line, expr = item.line_number, item.target.strip()
        if not 1 <= line <= len(lines) or not _valid_expression(expr, language):
            continue
        assigned = _assigned_target(source, line, language)
        model_target = expr
        if assigned:
            expr = assigned[0]
        identity = (line, re.sub(r"\s+", "", expr))
        if identity in seen:
            continue
        seen.add(identity)
        context_exprs = []
        for candidate in item.context_exprs[:4]:
            candidate = candidate.strip()
            if candidate != expr and candidate not in context_exprs and _valid_expression(candidate, language):
                context_exprs.append(candidate)
        if (assigned and re.sub(r"\s+", "", model_target) == re.sub(r"\s+", "", assigned[1])
                and not any(re.sub(r"\s+", "", value) == re.sub(r"\s+", "", model_target)
                            for value in context_exprs)):
            context_exprs = [model_target, *context_exprs][:4]
        probes.append({"id": len(probes), "line_number": line, "target": expr,
                       "context_exprs": context_exprs})
    output_assignment = _output_assignment(source, language)
    if output_assignment:
        line, target, right_side = output_assignment
        identity = (line, target)
        if identity not in seen:
            context = [right_side] if _valid_expression(right_side, language) else []
            probes.insert(0, {"line_number": line, "target": target, "context_exprs": context})
    probes = probes[:MAX_PROBES]
    for index, probe in enumerate(probes):
        probe["id"] = index
    return probes


def _modal_credentials(settings: dict[str, str]) -> tuple[str, str]:
    token_id = settings.get("MODAL_TOKEN_ID", "").strip()
    token_secret = settings.get("MODAL_TOKEN_SECRET", "").strip()
    if not token_id or not token_secret:
        raise TraceError("실행 검증용 샌드박스가 아직 설정되지 않았습니다. 관리자에게 문의해 주세요.")
    return token_id, token_secret


def run_isolated_trace(
    language: str, source: str, probes: list[dict[str, Any]], settings: dict[str, str]
) -> dict[str, Any]:
    """제출 코드를 앱 호스트가 아닌 요청별 Modal Sandbox에서 실행한다."""
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise TraceError("코드가 너무 깁니다. 24KB 이하로 줄여 주세요.")
    token_id, token_secret = _modal_credentials(settings)
    try:
        import modal
    except ImportError as exc:
        raise TraceError("실행 검증 의존성을 설치하지 못했습니다.") from exc

    sandbox = None
    client = None
    try:
        client = modal.Client.from_credentials(token_id, token_secret)
        app = modal.App.lookup("execution-trace-tutor", create_if_missing=True, client=client)
        image = modal.Image.debian_slim(python_version="3.11").apt_install(
            "gcc", "g++", "default-jdk-headless"
        )
        sandbox = modal.Sandbox.create(
            app=app, image=image, client=client, timeout=50, idle_timeout=30,
            cpu=1.0, memory=512, block_network=True,
        )
        sandbox.filesystem.write_text(WORKER.read_text(encoding="utf-8"), "/tmp/trace_worker.py")
        sandbox.filesystem.write_text(
            json.dumps({"language": language, "source": source, "probes": probes}),
            "/tmp/trace_request.json",
        )
        process = sandbox.exec(
            "python", "/tmp/trace_worker.py", "/tmp/trace_request.json", "/tmp/trace_result.json",
            timeout=25,
        )
        process.wait()
        if process.returncode != 0:
            raise TraceError("격리 실행이 중단됐습니다. 코드를 확인하거나 다시 시도해 주세요.")
        result = json.loads(sandbox.filesystem.read_text("/tmp/trace_result.json"))
        if not isinstance(result, dict) or not result.get("ok"):
            raise TraceError(str(result.get("error", "실행 결과를 확인할 수 없습니다."))[:300])
        return result
    except TraceError:
        raise
    except Exception as exc:
        raise TraceError("격리 실행 서비스에 연결하지 못했습니다. 설정과 상태를 확인해 주세요.") from exc
    finally:
        if sandbox is not None:
            try:
                sandbox.terminate()
            except Exception:
                pass
