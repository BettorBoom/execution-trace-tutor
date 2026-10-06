"""고정 예제로 계측의 제어 흐름·재귀·부분 실패 회복을 검증한다."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import trace_worker
from trace_worker import _instrument, _observations, verify


class TraceWorkerRegressions(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="trace-regressions-")
        self.addCleanup(self.directory.cleanup)
        self.root = patch.object(trace_worker, "ROOT", Path(self.directory.name))
        self.root.start()
        self.addCleanup(self.root.stop)

    def test_unbraced_control_flow_keeps_original_values(self):
        source = '''#include <stdio.h>
int main(void) {
 int x=0, y=0;
 if (0)
   x=1;
 else
   y=2;
 for (int i=0; i<2; i++)
   y=y+1;
 x=y+2;
 printf("done");
}'''
        probes = [
            {"id": 0, "line_number": 5, "target": "x"},
            {"id": 1, "line_number": 7, "target": "y"},
            {"id": 2, "line_number": 9, "target": "y"},
            {"id": 3, "line_number": 10, "target": "x"},
        ]
        java_source = source.replace("#include <stdio.h>", "public class Demo {").replace(
            "int main(void) {", "public static void main(String[] args) {"
        ).replace("if (0)", "if (false)").replace('printf("done");', 'System.out.print("done");') + "\n}"
        for language, compiler, code in (("C", "gcc", source), ("C++", "g++", source), ("Java", "javac", java_source)):
            if not shutil.which(compiler):
                continue
            with self.subTest(language=language):
                result = verify({"language": language, "source": code, "probes": probes})
                values = [(item["id"], item["occurrence"], item["after"]) for item in result["observations"]]
                self.assertEqual(values, [(1, 1, "2"), (2, 1, "3"), (2, 2, "4"), (3, 1, "6")])
                self.assertEqual(result["stdout"], "done")

    @unittest.skipUnless(shutil.which("gcc"), "C 컴파일러 필요")
    def test_inline_if_probe_only_runs_in_taken_branch_and_preserves_else(self):
        source = '''#include <stdio.h>
int main(void) {
 int x=0;
 if (0) x=1;
 else x=2;
 if (1) x=3;
 printf("%d", x);
}'''
        probes = [{"id": 0, "line_number": 4, "target": "x"},
                  {"id": 1, "line_number": 6, "target": "x"}]
        result = verify({"language": "C", "source": source, "probes": probes})
        self.assertEqual(result["stdout"], "3")
        self.assertEqual([(event["id"], event["before"], event["after"])
                          for event in result["observations"]], [(1, "2", "3")])

    def test_recursive_observations_pair_same_invocation(self):
        source = '''def f(n):
    if n == 0:
        return 0
    x = n * 10
    x = f(n-1) + 1
    return x
print(f(3))'''
        probes = [{"id": 0, "line_number": 5, "target": "x", "context_exprs": ["n"]}]
        result = verify({"language": "Python", "source": source, "probes": probes})
        self.assertEqual(result["stdout"], "3\n")
        self.assertEqual(
            [(item["occurrence"], item["before"], item["after"], item["context_values"]["n"], item["event_index"])
             for item in result["observations"]],
            [(1, "30", "3", "3", 3), (2, "20", "2", "2", 2), (3, "10", "1", "1", 1)],
        )

    def test_noninteger_recursive_frame_does_not_steal_parent_after(self):
        probes = [{"id": 0, "line_number": 1, "target": "x"}]
        mark = trace_worker.TRACE_MARK + "0:"
        stderr = "\n".join(mark + event for event in (
            "before:int|30", "before:str|text", "after:int|1", "after:int|3",
        ))
        self.assertEqual([(item["before"], item["after"]) for item in _observations(stderr, probes)], [("30", "3")])

    def test_recursive_frames_above_limit_do_not_shift_values(self):
        probes = [{"id": 0, "line_number": 1, "target": "x"}]
        mark = trace_worker.TRACE_MARK + "0:"
        events = [f"before:int|{number}" for number in range(1, 102)]
        events += [f"after:int|{number+10}" for number in range(101, 0, -1)]
        observations = _observations("\n".join(mark + event for event in events), probes)
        self.assertEqual(len(observations), 100)
        self.assertTrue(all(int(item["after"]) == int(item["before"]) + 10 for item in observations))

    def test_recursive_exception_does_not_pair_abandoned_frame(self):
        source = '''def f(n):
    x = n * 10
    if n == 0:
        raise ValueError()
    try:
        x = f(n-1) + 1
    except ValueError:
        return 0
    return x
print(f(2))'''
        probes = [{"id": 0, "line_number": 6, "target": "x", "context_exprs": ["n"]}]
        result = verify({"language": "Python", "source": source, "probes": probes})
        self.assertEqual(result["stdout"], "1\n")
        self.assertEqual(result["observations"], [])

    @unittest.skipUnless(shutil.which("gcc"), "C 컴파일러 필요")
    def test_uninitialized_before_value_is_not_observed(self):
        source = '#include <stdio.h>\nint main(void) {\n int x;\n x=4;\n printf("%d", x);\n}'
        probes = [{"id": 0, "line_number": 4, "target": "x"}]
        result = verify({"language": "C", "source": source, "probes": probes})
        self.assertEqual(result["stdout"], "4")
        self.assertEqual(result["observations"], [])

    @unittest.skipUnless(shutil.which("gcc"), "C 컴파일러 필요")
    def test_pointer_sample_survives_declaration_and_invalid_pointer_probe(self):
        from test_app import POINTER_SOURCE
        probes = [
            {"id": 0, "line_number": 11, "target": "num"},
            {"id": 1, "line_number": 13, "target": "p"},
            {"id": 2, "line_number": 4, "target": "*(*arr+i)", "context_exprs": ["i", "size"]},
            {"id": 3, "line_number": 13, "target": "num", "context_exprs": ["arr[2]"]},
        ]
        result = verify({"language": "C", "source": POINTER_SOURCE, "probes": probes})
        self.assertEqual(result["stdout"], "1")
        self.assertEqual({item["line_number"] for item in result["observations"]}, {4, 13})
        self.assertEqual({item["id"] for item in result["observations"]}, {2, 3})
        changed = [item for item in result["observations"] if item["before"] != item["after"]]
        self.assertTrue(any(item["line_number"] == 4 and item["occurrence"] == 3
                            and item["after"] == "1" and item["context_values"]["i"] == "2" for item in changed))
        self.assertTrue(any(item["line_number"] == 13 and item["after"] == "1" for item in changed))

    def test_invalid_probe_preserves_valid_probe(self):
        source = "x=1\ny=10\nx=x+2\ny=y*2\nprint(x+y)"
        probes = [
            {"id": 0, "line_number": 3, "target": "missing"},
            {"id": 1, "line_number": 4, "target": "y", "context_exprs": ["x"]},
        ]
        result = verify({"language": "Python", "source": source, "probes": probes})
        self.assertEqual([(item["id"], item["after"]) for item in result["observations"]], [(1, "20")])
        self.assertEqual(result["observations"][0]["context_values"], {"x": "3"})

    def test_invalid_context_is_dropped_without_losing_other_contexts_or_order(self):
        source = "x=1\ny=10\nx=x+2\ny=y*2\nprint(x+y)"
        probes = [
            {"id": 0, "line_number": 4, "target": "y", "context_exprs": ["x"]},
            {"id": 1, "line_number": 3, "target": "x", "context_exprs": ["missing"]},
        ]
        result = verify({"language": "Python", "source": source, "probes": probes})
        events = sorted(result["observations"], key=lambda item: item["event_index"])
        self.assertEqual([(item["id"], item["after"]) for item in events], [(1, "3"), (0, "20")])
        self.assertEqual(events[0]["context_values"], {})
        self.assertEqual(events[1]["context_values"], {"x": "3"})

    def test_declarations_and_multiple_statements_are_not_instrumented(self):
        for language, source in (
            ("C", "int x=1;"), ("C++", "int *p=nullptr;"),
            ("Java", "Integer x=1;"), ("C", "x=1; y=2;"),
            ("Python", "x=1; print(x)"), ("Python", "if True: x=1"),
            ("Python", "x: int"),
        ):
            with self.subTest(language=language, source=source):
                self.assertEqual(_instrument(source, language, [{"id": 0, "line_number": 1, "target": "x"}]), source)

    def test_run_respects_remaining_budget(self):
        with patch("trace_worker.time.monotonic", return_value=10), patch("trace_worker.subprocess.run") as run:
            run.return_value.stdout = ""
            run.return_value.stderr = ""
            trace_worker._run(["fixed-test"], timeout=5, deadline=10.25)
            self.assertEqual(run.call_args.kwargs["timeout"], 0.25)
            with self.assertRaisesRegex(ValueError, "시간이 초과"):
                trace_worker._run(["fixed-test"], timeout=5, deadline=9)
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
