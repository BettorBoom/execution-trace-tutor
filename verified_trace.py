"""격리 실행에서 관측한 값만 새 문제의 정답으로 사용한다."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
각 probe의 line_number는 실행 직후 값을 볼 단일 행 대입문의 물리적 줄 번호입니다.
target은 그 시점에 읽을 수 있는 정수 변수/배열 원소/역참조 표현식입니다.
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
        if (line, expr) in seen:
            continue
        seen.add((line, expr))
        context_exprs = []
        for candidate in item.context_exprs[:4]:
            candidate = candidate.strip()
            if candidate != expr and candidate not in context_exprs and _valid_expression(candidate, language):
                context_exprs.append(candidate)
        probes.append({"id": len(probes), "line_number": line, "target": expr,
                       "context_exprs": context_exprs})
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
