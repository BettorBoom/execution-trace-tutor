"""Modal의 일회용 격리 컨테이너 안에서만 실행되는 코드 검증 워커."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/tmp/trace_job")
OUTPUT_LIMIT = 64_000
EXECUTION_BUDGET = 22.0  # Modal exec의 25초 제한 안에서 검증과 보조 관측을 끝낸다.
TRACE_MARK = "__EXECUTION_TRACE_PROBE__"
SANITIZERS = ["-fsanitize=address,undefined"] if sys.platform.startswith("linux") else ["-fsanitize=undefined"]


def _run(command: list[str], *, timeout: int, deadline: float | None = None) -> subprocess.CompletedProcess[str]:
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("실행 검증 시간이 초과됐습니다. 코드를 줄여 다시 시도해 주세요.")
        timeout = min(timeout, remaining)
    try:
        result = subprocess.run(
            command, input="", capture_output=True, text=True, errors="replace",
            timeout=timeout, cwd=ROOT,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("실행 시간이 초과됐습니다. 반복이나 입력 대기를 확인해 주세요.") from exc
    if len(result.stdout) > OUTPUT_LIMIT or len(result.stderr) > OUTPUT_LIMIT:
        raise ValueError("프로그램 출력이 너무 깁니다. 출력량을 줄여 주세요.")
    return result


def _commands(language: str, source: str, stem: str) -> tuple[list[str] | None, list[str], Path]:
    # 원본에는 없던 실행 전 읽기가 미초기화 값을 관측하지 않도록 계측 빌드에서 검사한다.
    diagnostic_flags = [] if stem == "original" else ["-Werror=uninitialized"]
    if diagnostic_flags and sys.platform.startswith("linux"):
        diagnostic_flags.append("-Werror=maybe-uninitialized")
    optimization = "-O0" if stem == "original" else "-O1"
    if language == "Python":
        path = ROOT / f"{stem}.py"
        return None, [sys.executable, "-I", str(path)], path
    if language == "C":
        path = ROOT / f"{stem}.c"
        return ["gcc", "-std=c11", optimization, *diagnostic_flags, *SANITIZERS,
                "-fno-sanitize-recover=all", str(path), "-o", str(ROOT / stem)], [str(ROOT / stem)], path
    if language == "C++":
        path = ROOT / f"{stem}.cpp"
        return ["g++", "-std=c++17", optimization, *diagnostic_flags, *SANITIZERS,
                "-fno-sanitize-recover=all", str(path), "-o", str(ROOT / stem)], [str(ROOT / stem)], path
    if language == "Java":
        if re.search(r"^\s*package\s+", source, re.M):
            raise ValueError("현재 Java package 선언은 지원하지 않습니다.")
        match = re.search(r"\bpublic\s+class\s+([A-Za-z_$][\w$]*)", source)
        if not match:
            match = re.search(r"\bclass\s+([A-Za-z_$][\w$]*)", source)
        if not match:
            raise ValueError("Java 클래스 이름을 확인할 수 없습니다.")
        class_name = match.group(1)
        path = ROOT / f"{class_name}.java"
        return ["javac", "-encoding", "UTF-8", "-d", str(ROOT), str(path)], ["java", "-cp", str(ROOT), class_name], path
    raise ValueError("지원하지 않는 언어입니다.")


def _execute(language: str, source: str, stem: str, deadline: float | None = None) -> subprocess.CompletedProcess[str]:
    compile_command, command, path = _commands(language, source, stem)
    path.write_text(source, encoding="utf-8")
    if compile_command:
        compiled = _run(compile_command, timeout=8, deadline=deadline)
        if compiled.returncode:
            raise ValueError("코드 컴파일에 실패했습니다. 문법과 언어 선택을 확인해 주세요.")
    return _run(command, timeout=5, deadline=deadline)


def _python_line_is_safe(source: str, number: int) -> bool:
    try:
        tree = ast.parse(source.split("\n")[number - 1].strip())
    except SyntaxError:
        return False
    return (len(tree.body) == 1
            and isinstance(tree.body[0], (ast.Assign, ast.AugAssign, ast.AnnAssign))
            and not (isinstance(tree.body[0], ast.AnnAssign) and tree.body[0].value is None))


def _assignment_line_is_safe(code: str) -> bool:
    """선언·복합 문장 대신 독립된 단일 대입문만 블록으로 감싼다."""
    if not code.endswith(";") or code.count(";") != 1 or any(token in code for token in ("{", "}", "//", "/*", "*/")):
        return False
    assignment = re.search(r"(?<![=!<>])(?:[+*/%&|^-]|<<|>>)?=(?!=)", code)
    if assignment is None:
        return False
    lhs = code[:assignment.start()].strip()
    # int x / int *p 같은 선언과 if/for 등 제어문은 이 형태에 들어오지 않는다.
    return bool(re.fullmatch(
        r"(?:\*\s*)*(?:[A-Za-z_$][\w$]*|\([^;{}]+\))"
        r"(?:\s*(?:\[[^;{}]+\]|\.[A-Za-z_$][\w$]*|->[A-Za-z_$][\w$]*))*",
        lhs,
    ))


def _probe_statement(language: str, probe_id: int, phase: str, target: str) -> str:
    label = f"{TRACE_MARK}{probe_id}:{phase}:"
    if language == "Python":
        return (
            f"(lambda __trace_v: __import__('sys').stderr.write({label!r} + "
            f"type(__trace_v).__name__ + '|' + str(__trace_v) + '\\n'))({target})"
        )
    if language == "Java":
        return (
            f'System.err.println("{label}" + ((Object)({target})).getClass().getSimpleName() '
            f'+ "|" + String.valueOf({target}));'
        )
    if language == "C":
        signed = "signed char:1,short:1,int:1,long:1,long long:1,default:0"
        check = f'_Static_assert(_Generic(({target}), {signed}), "integer probe only"); '
    else:
        value_type = f'std::remove_cv_t<std::remove_reference_t<decltype(({target}))>>'
        check = f'static_assert(std::is_integral_v<{value_type}> && std::is_signed_v<{value_type}>); '
    return check + f'fprintf(stderr, "{label}int|%lld\\n", (long long)({target}));'


def _instrument(source: str, language: str, probes: list[dict]) -> str:
    lines = source.split("\n")
    grouped: dict[int, list[dict]] = {}
    for probe in probes:
        number = probe["line_number"]
        if not 1 <= number <= len(lines):
            continue
        code = lines[number - 1].strip()
        if language == "Python":
            eligible = _python_line_is_safe(source, number)
        else:
            eligible = _assignment_line_is_safe(code)
        if eligible:
            grouped.setdefault(number, []).append(probe)
    if not grouped:
        return source
    pieces = []
    for number, line in enumerate(lines, 1):
        selected = grouped.get(number, [])
        indent = re.match(r"\s*", line).group(0)
        if selected and language != "Python":
            pieces.append(indent + "{")
        for probe in selected:
            pieces.append(indent + _probe_statement(language, probe["id"], "before", probe["target"]))
            for context_index, expr in enumerate(probe.get("context_exprs", [])):
                pieces.append(indent + _probe_statement(language, probe["id"], f"context{context_index}", expr))
        pieces.append(line)
        for probe in selected:
            pieces.append(indent + _probe_statement(language, probe["id"], "after", probe["target"]))
        if selected and language != "Python":
            pieces.append(indent + "}")
    result = "\n".join(pieces)
    if language in ("C", "C++"):
        result = ("#include <type_traits>\n" if language == "C++" else "") + "#include <stdio.h>\n" + result
    return result


def _observations(stderr: str, probes: list[dict]) -> list[dict]:
    found: dict[tuple[int, int], dict] = {}
    counts: dict[int, int] = {}
    active: dict[int, list[dict]] = {}
    unbalanced: set[int] = set()
    event_index = 0
    pattern = re.compile(rf"^{TRACE_MARK}(\d+):(before|after|context\d+):(.*)$")
    for line in stderr.splitlines():
        match = pattern.fullmatch(line)
        if not match:
            continue
        probe_id, phase, encoded = int(match[1]), match[2], match[3].strip()
        value_type, separator, value = encoded.partition("|")
        valid = bool(separator and value_type in ("int", "Integer", "Long", "Short", "Byte")
                     and re.fullmatch(r"-?\d+", value))
        stack = active.setdefault(probe_id, [])
        if phase.startswith("context"):
            if stack and valid:
                stack[-1].setdefault("context_values", {})[phase] = value
            continue
        if phase == "before":
            counts[probe_id] = counts.get(probe_id, 0) + 1
            stack.append({"occurrence": counts[probe_id], "before": value if valid else None})
        elif stack:
            # 재귀 호출은 가장 최근에 시작한 같은 probe의 실행부터 완료된다.
            invocation = stack.pop()
            event_index += 1
            occurrence = invocation.pop("occurrence")
            if valid and invocation["before"] is not None and occurrence <= 100:
                found[(probe_id, occurrence)] = {**invocation, "after": value, "event_index": event_index}
        else:
            unbalanced.add(probe_id)
    observations = []
    for probe in probes:
        probe_id = probe["id"]
        # 例外 등으로 after가 빠진 재귀 프레임은 다른 호출과 연결될 수 있으므로 모두 제외한다.
        if active.get(probe_id) or probe_id in unbalanced:
            continue
        count = min(counts.get(probe_id, 0), 100)
        for occurrence in range(1, count + 1):
            values = found.get((probe_id, occurrence), {})
            if "before" in values and "after" in values:
                observations.append({
                    "id": probe_id,
                    "line_number": probe["line_number"],
                    "target": probe["target"],
                    "occurrence": occurrence,
                    "before": values["before"],
                    "after": values["after"],
                    "event_index": values["event_index"],
                    "context_values": {
                        expr: values.get("context_values", {}).get(f"context{index}")
                        for index, expr in enumerate(probe.get("context_exprs", []))
                        if values.get("context_values", {}).get(f"context{index}") is not None
                    },
                })
    return observations


def verify(request: dict) -> dict:
    deadline = time.monotonic() + EXECUTION_BUDGET
    language, source, probes = request["language"], request["source"], request["probes"]
    input_patterns = {
        "C": r"\b(scanf|fscanf|getchar|fgets|read)\s*\(",
        "C++": r"\b(scanf|fscanf|getchar|fgets|read)\s*\(|\b(?:std::)?cin\b",
        "Java": r"\bSystem\.in\b|\bScanner\s*\(",
        "Python": r"\binput\s*\(|\bsys\.stdin\b",
    }
    if re.search(input_patterns[language], source):
        raise ValueError("표준 입력이 필요한 코드는 입력값을 지정할 수 없어 출제하지 않습니다.")
    ROOT.mkdir(parents=True, exist_ok=True)
    original = _execute(language, source, "original", deadline)
    if original.returncode:
        raise ValueError("프로그램이 실행 중 오류로 종료됐습니다. 현재는 정상 종료하는 코드만 출제합니다.")
    if len(original.stdout) > 1000:
        raise ValueError("출력이 너무 깁니다. 학습용 출력은 1,000자 이하로 줄여 주세요.")
    repeat = _run(_commands(language, source, "original")[1], timeout=5, deadline=deadline)
    if repeat.returncode != original.returncode or repeat.stdout != original.stdout:
        raise ValueError("실행할 때마다 결과가 달라집니다. 이 코드는 확정 정답으로 출제하지 않습니다.")
    observations = []
    def stable_capture(instrumented_source: str, stem: str, active_probes: list[dict]) -> list[dict]:
        try:
            first = _execute(language, instrumented_source, stem, deadline)
            second = _run(_commands(language, instrumented_source, stem)[1], timeout=5, deadline=deadline)
        except ValueError:
            return []
        if (first.returncode or second.returncode or first.stdout != original.stdout
                or second.stdout != original.stdout):
            return []
        first_values = _observations(first.stderr, active_probes)
        return first_values if first_values == _observations(second.stderr, active_probes) else []

    if probes:
        instrumented = _instrument(source, language, probes)
        if instrumented != source:
            observations = stable_capture(instrumented, "instrumented", probes)
            if not observations:
                successful: list[dict] = []
                individual: list[list[dict]] = []
                # 하나의 잘못된 표현식이 모든 관측을 버리지 않도록 최대 6개를 개별 재시도한다.
                # ponytail: 전체 22초 예산 안에서만 재시도하며, 초과하면 검증된 출력만 남긴다.
                for probe in probes[:6]:
                    if time.monotonic() >= deadline:
                        break
                    candidates = [probe]
                    if probe.get("context_exprs"):
                        candidates.append({**probe, "context_exprs": []})
                    for candidate in candidates:
                        instrumented = _instrument(source, language, [candidate])
                        if instrumented == source:
                            break
                        captured = stable_capture(instrumented, "single_probe", [candidate])
                        if captured:
                            successful.append(candidate)
                            individual.append(captured)
                            break
                if individual:
                    # 별도 실행의 event_index는 서로 비교할 수 없으므로 함께 다시 관측한다.
                    # 예산이 부족하면 한 번의 실행에서 얻은 관측만 사용한다.
                    observations = individual[0]
                    if len(successful) > 1:
                        combined = stable_capture(
                            _instrument(source, language, successful), "recovered_probes", successful
                        )
                        if combined:
                            observations = combined
    return {"ok": True, "stdout": original.stdout, "observations": observations}


def main() -> None:
    request_path, result_path = map(Path, sys.argv[1:3])
    try:
        result = verify(json.loads(request_path.read_text(encoding="utf-8")))
    except (KeyError, TypeError, ValueError) as exc:
        result = {"ok": False, "error": str(exc)}
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
