"""외부 API 없이 실제 실행값과 새 채점 계약을 확인한다."""

from __future__ import annotations

import shutil
import unittest
import json
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

from app import TutorialError, apply_record, build_verified_tutorial, validate_tutorial
from test_app import POINTER_SOURCE, SOURCE, payload
from trace_worker import verify
from verified_trace import TraceError, run_isolated_trace
from storage import new_tutorial_record


class VerifiedTraceTests(unittest.TestCase):
    def test_three_questions_from_repeated_changes_in_four_languages(self):
        samples = [
            ("C", '#include <stdio.h>\nint main(){\n int x=0;\n for(int i=0;i<3;i++){\n  x=x+i+1;\n }\n printf("%d",x);\n}', 5, ("gcc",)),
            ("C++", '#include <iostream>\nint main(){\n int x=0;\n for(int i=0;i<3;i++){\n  x=x+i+1;\n }\n std::cout<<x;\n}', 5, ("g++",)),
            ("Java", 'class Demo {\n public static void main(String[] a){\n int x=0;\n for(int i=0;i<3;i++){\n  x=x+i+1;\n }\n System.out.print(x);\n }\n}', 5, ("javac", "java")),
            ("Python", 'x=0\nfor i in range(3):\n    x=x+i+1\nprint(x)', 3, ()),
        ]
        for language, source, line, commands in samples:
            if any(not shutil.which(command) for command in commands):
                continue
            with self.subTest(language=language):
                probes = [{"id": 0, "line_number": line, "target": "x", "context_exprs": ["i"]}]
                result = verify({"language": language, "source": source, "probes": probes})
                tutorial = build_verified_tutorial(
                    language, source, probes, result["observations"], result["stdout"]
                )
                self.assertEqual([step.answer for step in tutorial.steps], ["1", "6", "6"])
                self.assertEqual([step.verified_fact["occurrence"] for step in tutorial.steps[:2]], [1, 3])

    def test_four_languages_use_observed_value_and_output(self):
        samples = [
            ("C", "#include <stdio.h>\nint main(){\n int x=1;\n x=x+2;\n printf(\"%d\",x);\n}", 4, ("gcc",)),
            ("C++", "#include <iostream>\nint main(){\n int x=1;\n x=x+2;\n std::cout<<x;\n}", 4, ("g++",)),
            ("Java", "public class Demo {\n public static void main(String[] a) {\n int x=1;\n x=x+2;\n System.out.print(x);\n }\n}", 4, ("javac", "java")),
            ("Python", "x=1\nx=x+2\nprint(x)", 2, ()),
        ]
        for language, source, line, commands in samples:
            if any(not shutil.which(command) for command in commands):
                continue
            with self.subTest(language=language):
                probes = [{"id": 0, "line_number": line, "target": "x", "context_exprs": ["x+2"]}]
                result = verify({"language": language, "source": source, "probes": probes})
                tutorial = build_verified_tutorial(
                    language, source, probes, result["observations"], result["stdout"]
                )
                self.assertEqual([step.answer for step in tutorial.steps], ["3", "3"])
                self.assertEqual(tutorial.steps[0].changes[0].before, "1")
                self.assertEqual(tutorial.steps[0].verified_fact["after"], "3")
                self.assertEqual(tutorial.schema_version, 4)

    @unittest.skipUnless(shutil.which("gcc"), "C 컴파일러 필요")
    def test_pointer_loop_calculation_is_measured(self):
        probes = [
            {"id": 0, "line_number": 4, "target": "*(*arr+i)",
             "context_exprs": ["i", "size", "*(*arr+i)+i", "(*(*arr+i)+i)%size"]},
            {"id": 1, "line_number": 13, "target": "num", "context_exprs": ["arr[2]"]},
        ]
        result = verify({"language": "C", "source": POINTER_SOURCE, "probes": probes})
        tutorial = build_verified_tutorial("C", POINTER_SOURCE, probes, result["observations"], result["stdout"])
        self.assertEqual([step.answer for step in tutorial.steps], ["1", "1", "1"])
        self.assertEqual(tutorial.steps[0].verified_fact["occurrence"], 3)
        self.assertEqual(tutorial.steps[0].verified_fact["context_values"]["i"], "2")
        self.assertEqual(tutorial.steps[0].verified_fact["context_values"]["*(*arr+i)+i"], "6")
        self.assertIn("실행 직전 i=2", tutorial.steps[0].question)
        self.assertIn("`size` = 5", tutorial.steps[0].explanation)
        self.assertEqual(result["stdout"], "1")

        # Supabase JSONB가 context_values 키 순서를 바꿔도 저장본을 열 수 있어야 한다.
        stored = json.loads(json.dumps(tutorial.model_dump(), sort_keys=True))
        restored = validate_tutorial(stored, "C", POINTER_SOURCE)
        self.assertEqual([step.answer for step in restored.steps], ["1", "1", "1"])
        record = new_tutorial_record("출력값", "C", POINTER_SOURCE, "gpt-4.1-mini", stored)
        record["owner_id"] = "google:test"
        state = {"owner_id": "google:test", "generation_count": 0}
        apply_record(state, record)
        self.assertIn("4 → 1", state["quiz_data"]["annotated_code"])

        broken = tutorial.model_dump()
        broken["steps"][0]["answer"] = "0"
        with self.assertRaises(TutorialError):
            validate_tutorial(broken, "C", POINTER_SOURCE)

        broken = tutorial.model_dump()
        broken["steps"][0]["verified_fact"]["before"] = "99"
        with self.assertRaises(TutorialError):
            validate_tutorial(broken, "C", POINTER_SOURCE)

    @unittest.skipUnless(shutil.which("gcc"), "C 컴파일러 필요")
    def test_repeated_array_update_yields_three_distinct_questions(self):
        source = (
            "#include <stdio.h>\nint main(void){\n"
            " int w=3,h=2,x,y;\n int mines[2][3]={{0,1,0},{1,0,0}};\n"
            " for(y=0;y<h;y++){\n  for(x=0;x<w;x++)\n"
            "   mines[y][x]=mines[y][x]+x+y;\n }\n"
            " for(y=0;y<h;y++){\n  for(x=0;x<w;x++)\n"
            "   printf(\"%d\",mines[y][x]);\n  printf(\"\\n\");\n }\n}"
        )
        probes = [{"id": 0, "line_number": 7, "target": "mines[y][x]",
                   "context_exprs": ["x", "y", "mines[y][x]+x+y"]}]
        result = verify({"language": "C", "source": source, "probes": probes})
        tutorial = build_verified_tutorial("C", source, probes, result["observations"], result["stdout"])
        self.assertEqual(len(tutorial.steps), 3)
        self.assertEqual([step.verified_fact["occurrence"] for step in tutorial.steps[:2]], [2, 6])
        self.assertEqual([step.answer for step in tutorial.steps[:2]], ["2", "3"])
        self.assertEqual(tutorial.steps[-1].answer, '"022\\n223\\n"')
        self.assertEqual(set(tutorial.steps[-1].choices), {
            '"022\\n223\\n"', '"022\\n222\\n"', '"022\\n224\\n"',
        })

    def test_same_line_same_execution_is_asked_once(self):
        probes = [
            {"id": 0, "line_number": 4, "target": "*(*arr + i)"},
            {"id": 1, "line_number": 4, "target": "*(*arr+i)"},
        ]
        observations = [
            {"id": 0, "line_number": 4, "target": "*(*arr + i)", "occurrence": 3,
             "before": "4", "after": "1", "event_index": 3, "context_values": {"i": "2"}},
            {"id": 1, "line_number": 4, "target": "*(*arr+i)", "occurrence": 3,
             "before": "4", "after": "1", "event_index": 4, "context_values": {}},
        ]
        tutorial = build_verified_tutorial("C", POINTER_SOURCE, probes, observations, "1")
        self.assertEqual([(step.line_number, step.answer) for step in tutorial.steps], [(4, "1"), (14, "1")])
        self.assertIn("i=2", tutorial.steps[0].question)

    def test_repeated_line_prefers_distinct_value_change(self):
        source = "a=[0,0,1]\nfor i in range(3):\n    a[i]=a[i]+1\nprint(a[2])"
        probes = [{"id": 0, "line_number": 3, "target": "a[i]"}]
        observations = [
            {"id": 0, "line_number": 3, "target": "a[i]", "occurrence": n,
             "before": before, "after": after, "event_index": n,
             "context_values": {"i": str(n - 1)}}
            for n, before, after in ((1, "0", "1"), (2, "0", "1"), (3, "1", "2"))
        ]
        tutorial = build_verified_tutorial("Python", source, probes, observations, "2\n")
        self.assertEqual([step.verified_fact["occurrence"] for step in tutorial.steps[:2]], [1, 3])

    def test_old_model_answers_are_not_graded(self):
        page = AppTest.from_string(
            "from app import initialize_state, show_study_view\n"
            "initialize_state()\nshow_study_view()"
        ).run()
        page.session_state["quiz_data"] = payload()
        page.session_state["study_language"] = "C"
        page.session_state["study_source"] = SOURCE
        page.session_state["study_problem"] = "x의 값"
        page.run()
        self.assertTrue(any("채점을 중단" in item.value for item in page.warning))
        self.assertFalse(any((button.key or "").startswith("choice_") for button in page.button))

    def test_sandbox_configuration_is_required(self):
        with self.assertRaises(TraceError):
            run_isolated_trace("Python", "print(1)", [], {})

    def test_modal_sandbox_is_network_blocked_and_terminated(self):
        sandbox = MagicMock()
        sandbox.exec.return_value.returncode = 0
        sandbox.filesystem.read_text.return_value = json.dumps(
            {"ok": True, "stdout": "1\n", "observations": []}
        )
        with patch("modal.Client.from_credentials"), patch("modal.App.lookup"), \
             patch("modal.Sandbox.create", return_value=sandbox) as create:
            result = run_isolated_trace(
                "Python", "print(1)", [],
                {"MODAL_TOKEN_ID": "test-id", "MODAL_TOKEN_SECRET": "test-secret"},
            )
        self.assertEqual(result["stdout"], "1\n")
        self.assertTrue(create.call_args.kwargs["block_network"])
        self.assertNotIn("secrets", create.call_args.kwargs)
        self.assertNotIn("test-secret", " ".join(
            str(call.args[0]) for call in sandbox.filesystem.write_text.call_args_list
        ))
        sandbox.terminate.assert_called_once()

    def test_string_looking_like_number_is_not_a_numeric_probe(self):
        result = verify({"language": "Python", "source": 'x="0"\nx="1"\nprint(x)',
                         "probes": [{"id": 0, "line_number": 2, "target": "x"}]})
        self.assertEqual(result["observations"], [])
        self.assertEqual(result["stdout"], "1\n")

    def test_bad_context_falls_back_to_verified_target(self):
        source = "x=1\nx=x+2\nprint(x)"
        probes = [{"id": 0, "line_number": 2, "target": "x", "context_exprs": ["missing"]}]
        result = verify({"language": "Python", "source": source, "probes": probes})
        self.assertEqual(result["observations"][0]["after"], "3")
        self.assertEqual(result["observations"][0]["context_values"], {})

    def test_unspecified_input_is_not_given_a_guessed_answer(self):
        with self.assertRaisesRegex(ValueError, "표준 입력"):
            verify({"language": "Python", "source": "x=input()\nprint(x)", "probes": []})


if __name__ == "__main__":
    unittest.main()
