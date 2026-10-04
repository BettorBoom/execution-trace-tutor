"""API 키 없이 실행 가능한 핵심 동작 검증."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest
from openai import AuthenticationError, BadRequestError, InternalServerError, RateLimitError
import httpx2

from app import (
    GeneratedTutorial,
    GenerationValidationError,
    TutorialError,
    acknowledge_result,
    advance_step,
    api_error_diagnostics,
    describe_api_error,
    finalize_generated_tutorial,
    generate_tutorial,
    highlight_source,
    normalize_answer,
    output_lines,
    parse_json_object,
    run_generation_job,
    submit_answer,
    validate_tutorial,
    build_prompt,
    contact_html,
)


SOURCE = "\nint x = 1;\nint x = 1;\n"
POINTER_SOURCE = """#include <stdio.h>
void func(int** arr, int size){
    for(int i=0; i<size; i++){
        *(*arr + i) = (*(*arr+i) + i) % size;
    }
}
int main(){
    int arr[] = {3,1,4,1,5};
    int* p = arr;
    int** pp = &p;
    int num = 6;
    func(pp, 5);
    num = arr[2];
    printf("%d", num);
    return 0;
}"""
STEP = {
    "step_number": 1,
    "line_number": 2,
    "code_line": "int x = 1;",
    "question": "이 줄 실행 후 x의 값은? 숫자로 답하세요.",
    "choices": ["0", "1", "2"],
    "answer": "1",
    "hint": "대입 연산자를 살펴보세요.",
    "explanation": "x는 0에서 1로 바뀝니다.",
}


def payload(language="C", steps=None, source=SOURCE):
    return {
        "language": language,
        "steps": [STEP.copy()] if steps is None else steps,
        "annotated_code": source + "// 복습",
    }


def generated_payload(language="C", steps=None, source=SOURCE):
    value = payload(language, steps, source)
    generated_steps = []
    for execution_number, step in enumerate(value["steps"], start=1):
        generated_steps.append({
            "line_number": step["line_number"],
            "question_kind": "value_after",
            "target": "x",
            "context": f"main 실행 {execution_number}번째 시점",
            "answer": step["answer"],
            "distractors": [item for item in step["choices"] if item != step["answer"]][:2],
            "hint": step["hint"],
            "explanation": step["explanation"],
            "changes": [{"target": "x", "before": "0", "after": step["answer"], "calculations": []}],
        })
    return {"language": language, "steps": generated_steps, "line_notes": [
        {"line_number": min(2, len(source.split("\n"))), "note": "x의 실행 후 값은 1입니다."}
    ]}


def finish_generation(page):
    """백그라운드 작업의 결과를 앱 재실행으로 반영한다."""
    job = page.session_state["generation_job"]
    if job:
        try:
            job["future"].result(timeout=5)
        except TutorialError:
            pass
        page.run(timeout=15)


class TutorialTests(unittest.TestCase):
    @staticmethod
    def study_app():
        return AppTest.from_string(
            "from app import initialize_state, show_study_view\n"
            "initialize_state()\nshow_study_view()"
        ).run(timeout=15)

    @staticmethod
    def fake_store():
        store = MagicMock()
        store.has_key.return_value = True
        store.get_key.return_value = "test-key"
        store.insert_tutorial.side_effect = lambda owner, record: {**record, "owner_id": owner}
        return store

    def test_json_fallback_and_schema(self):
        value = payload()
        encoded = json.dumps(value, ensure_ascii=False)
        for text in (encoded, f"```json\n{encoded}\n```", f"결과:\n{encoded}\n완료"):
            self.assertEqual(validate_tutorial(parse_json_object(text), "C", SOURCE).language, "C")
        for text in ("", "```json\n{broken}\n```", "[]"):
            with self.assertRaises(TutorialError):
                parse_json_object(text)

    def test_openai_structured_response(self):
        response = SimpleNamespace(
            output_parsed=GeneratedTutorial.model_validate(
                generated_payload(steps=[STEP.copy(), {**STEP, "step_number": 8}])
            ),
            status="completed",
        )
        with patch("app.OpenAI") as client_class:
            client_class.return_value.responses.parse.return_value = response
            tutorial = generate_tutorial("test-key", "gpt-4.1-mini", "C", "x의 값", SOURCE)
            request = client_class.return_value.responses.parse.call_args.kwargs
        self.assertEqual(tutorial.steps[0].answer, "1")
        self.assertIn("1", tutorial.steps[0].choices)
        self.assertEqual(tutorial.schema_version, 2)
        self.assertEqual([step.step_number for step in tutorial.steps], [1, 2])
        self.assertEqual([step.code_line for step in tutorial.steps], ["int x = 1;", "int x = 1;"])
        self.assertEqual(request["model"], "gpt-4.1-mini")
        self.assertIs(request["text_format"], GeneratedTutorial)
        self.assertFalse(request["store"])
        self.assertEqual(client_class.call_args.kwargs["timeout"], 60.0)
        self.assertEqual(client_class.call_args.kwargs["max_retries"], 0)

    def test_openai_schema_avoids_unsupported_array_limits(self):
        schema = json.dumps(GeneratedTutorial.model_json_schema())
        self.assertNotIn('"minItems"', schema)
        self.assertNotIn('"maxItems"', schema)

    def test_malformed_json_retries_without_losing_valid_result(self):
        malformed = SimpleNamespace(output_parsed=None, status="completed", output_text="{broken")
        valid = SimpleNamespace(
            output_parsed=GeneratedTutorial.model_validate(generated_payload()), status="completed"
        )
        with patch("app.OpenAI") as client_class:
            client_class.return_value.responses.parse.side_effect = [malformed, valid]
            with self.assertLogs("execution_trace_tutor", level="WARNING"):
                tutorial = generate_tutorial("test-key", "gpt-4.1-mini", "C", "x의 값", SOURCE)
        self.assertEqual(tutorial.steps[0].answer, "1")
        self.assertEqual(client_class.return_value.responses.parse.call_count, 2)

    def test_background_job_returns_serializable_diagnostics(self):
        with patch("app.generate_tutorial", side_effect=GenerationValidationError(2, "값이 다름")):
            result = run_generation_job("test-key", "gpt-4.1-mini", "C", "출력값은?", SOURCE)
        self.assertIsNone(result["tutorial"])
        self.assertEqual(result["diagnostics"], {"step_number": 2, "reason": "값이 다름"})
        self.assertNotIn("test-key", json.dumps(result, ensure_ascii=False))

    def test_generated_line_number_still_validated(self):
        with self.assertRaises(TutorialError):
            finalize_generated_tutorial(
                generated_payload(steps=[{**STEP, "line_number": 99}]), "C", SOURCE
            )

    def test_seven_question_limit_and_selection_prompt(self):
        seven = generated_payload(steps=[STEP.copy() for _ in range(7)])
        self.assertEqual(len(finalize_generated_tutorial(seven, "C", SOURCE).steps), 7)
        with self.assertRaises(TutorialError):
            finalize_generated_tutorial(generated_payload(steps=[STEP.copy() for _ in range(8)]), "C", SOURCE)
        prompt = build_prompt("C", "출력은?", SOURCE)
        self.assertIn("1~7개", prompt)
        self.assertIn("단순 상수 초기화", prompt)
        self.assertIn("이번 출력", prompt)
        self.assertIn("distractors", prompt)

    def test_answer_and_choices_have_one_source(self):
        generated = generated_payload()
        generated["steps"][0]["distractors"] = ["0", "2"]
        tutorial = finalize_generated_tutorial(generated, "C", SOURCE)
        self.assertEqual(tutorial.steps[0].answer, "1")
        self.assertEqual(set(tutorial.steps[0].choices), {"0", "1", "2"})

    def test_invalid_choices_retry_only_once(self):
        duplicate = generated_payload()
        duplicate["steps"][0]["distractors"] = ["1", "2"]
        invalid_response = SimpleNamespace(
            output_parsed=GeneratedTutorial.model_validate(duplicate), status="completed"
        )
        valid_response = SimpleNamespace(
            output_parsed=GeneratedTutorial.model_validate(generated_payload()), status="completed"
        )
        retry_notice = MagicMock()
        with patch("app.OpenAI") as client_class:
            client_class.return_value.responses.parse.side_effect = [invalid_response, valid_response]
            tutorial = generate_tutorial("test-key", "gpt-4.1-mini", "C", "x의 값", SOURCE, retry_notice)
            self.assertEqual(client_class.return_value.responses.parse.call_count, 2)
        self.assertEqual(tutorial.steps[0].answer, "1")
        retry_notice.assert_called_once_with()

        with patch("app.OpenAI") as client_class:
            client_class.return_value.responses.parse.return_value = invalid_response
            with self.assertLogs("execution_trace_tutor", level="WARNING") as logs:
                with self.assertRaises(TutorialError) as raised:
                    generate_tutorial("test-key", "gpt-4.1-mini", "C", "x의 값", SOURCE)
            self.assertEqual(client_class.return_value.responses.parse.call_count, 2)
        self.assertEqual(len(logs.output), 2)
        self.assertIn('"attempt": 1', logs.output[0])
        self.assertIn('"attempt": 2', logs.output[1])
        self.assertNotIn("test-key", " ".join(logs.output))
        self.assertEqual(raised.exception.diagnostics["step_number"], 1)
        self.assertEqual(raised.exception.diagnostics["reason"], "중복 선택지 있음")
        self.assertEqual(raised.exception.diagnostics["retry_count"], 1)

    def test_empty_or_missing_choice_is_retryable(self):
        for choices, reason in (
            (["0", " "], "빈 선택지 있음"),
            (["0"], "오답 후보 개수가 2개가 아님"),
        ):
            with self.subTest(choices=choices):
                generated = generated_payload()
                generated["steps"][0]["distractors"] = choices
                with self.assertRaises(TutorialError) as raised:
                    finalize_generated_tutorial(generated, "C", SOURCE)
                self.assertEqual(raised.exception.diagnostics["reason"], reason)

    def test_answer_must_match_value_and_arithmetic(self):
        generated = generated_payload()
        generated["steps"][0]["answer"] = "2"
        with self.assertRaises(TutorialError) as raised:
            finalize_generated_tutorial(generated, "C", SOURCE)
        self.assertEqual(raised.exception.diagnostics["reason"], "정답과 모든 실행 후 값이 다름")

    def test_value_question_uses_matching_state_target(self):
        generated = generated_payload()
        generated["steps"][0]["target"] = "x의 값"
        result = finalize_generated_tutorial(generated, "C", SOURCE)
        self.assertEqual(result.steps[0].target, "x")
        self.assertIn("`x`의 값", result.steps[0].question)

    def test_numeric_aliases_and_repeated_questions_are_rejected(self):
        generated = generated_payload()
        generated["steps"][0]["distractors"] = ["01", "2"]
        with self.assertRaises(TutorialError) as raised:
            finalize_generated_tutorial(generated, "C", SOURCE)
        self.assertEqual(raised.exception.diagnostics["reason"], "중복 선택지 있음")

        generated["steps"][0]["distractors"] = ["잘 모르겠음", "2"]
        with self.assertRaises(TutorialError) as raised:
            finalize_generated_tutorial(generated, "C", SOURCE)
        self.assertEqual(raised.exception.diagnostics["reason"], "숫자 질문에 숫자가 아닌 선택지 있음")

        generated = generated_payload(steps=[STEP.copy(), STEP.copy()])
        generated["steps"][1]["context"] = generated["steps"][0]["context"]
        with self.assertRaises(TutorialError) as raised:
            finalize_generated_tutorial(generated, "C", SOURCE)
        self.assertEqual(raised.exception.diagnostics["reason"], "같은 시점의 질문이 중복됨")

    def test_other_languages_keep_original_lines_and_value_notes(self):
        samples = [
            ("C++", "int main(){\n int x=3;\n int* p=&x;\n *p=5;\n}", 4, "x", "3", "5"),
            ("Java", "class A {\n public static void main(String[] a){\n int[] xs={1,2};\n xs[1]=4;\n }\n}", 4, "xs[1]", "2", "4"),
            ("Python", "xs = [1, 2, 3]\nxs[1] = 4", 2, "xs[1]", "2", "4"),
        ]
        for language, source, line_number, target, before, after in samples:
            with self.subTest(language=language):
                generated = {
                    "language": language,
                    "steps": [{
                        "line_number": line_number, "question_kind": "value_after",
                        "target": target, "context": "대입 직후", "answer": after,
                        "distractors": [before, "9"], "hint": "대입식을 확인하세요.",
                        "explanation": f"{target}는 {before}에서 {after}로 바뀝니다.",
                        "changes": [{"target": target, "before": before, "after": after, "calculations": []}],
                    }],
                    "line_notes": [{"line_number": line_number, "note": f"{target}가 {before}에서 {after}로 변경됩니다."}],
                }
                tutorial = finalize_generated_tutorial(generated, language, source)
                self.assertEqual(tutorial.steps[0].code_line, source.split("\n")[line_number - 1])
                self.assertIn(f"{before} → {after}", tutorial.annotated_code)
                self.assertEqual(validate_tutorial(tutorial.model_dump(), language, source), tutorial)

    def test_pointer_example_rejects_missing_correct_choice(self):
        generated = {
            "language": "C",
            "steps": [{
                "line_number": 13, "question_kind": "value_after", "target": "num",
                "context": "func(pp, 5) 실행을 마친 뒤", "answer": "1",
                "distractors": ["0", "4"], "hint": "arr[2]의 새 값을 확인하세요.",
                "explanation": "i=2일 때 arr[2]=(4+2)%5=1이고 num은 6에서 1이 됩니다.",
                "changes": [{
                    "target": "num", "before": "6", "after": "1",
                    "calculations": [
                        {"left": 4, "operator": "+", "right": 2, "result": 6},
                        {"left": 6, "operator": "%", "right": 5, "result": 1},
                    ],
                }],
            }],
            "line_notes": [{
                "line_number": 4,
                "note": "i=0..4: arr 값이 [3,1,4,1,5]에서 [3,2,1,4,4]로 바뀝니다.",
            }],
        }
        result = finalize_generated_tutorial(generated, "C", POINTER_SOURCE)
        self.assertEqual(result.steps[0].answer, "1")
        self.assertEqual(set(result.steps[0].choices), {"0", "1", "4"})
        self.assertIn("num 6 → 1", result.annotated_code)
        self.assertIn("[3,2,1,4,4]", result.annotated_code)
        self.assertIn("num", result.steps[0].question)

        wrong = json.loads(json.dumps(generated))
        wrong["steps"][0]["answer"] = "2"
        with self.assertRaises(TutorialError):
            finalize_generated_tutorial(wrong, "C", POINTER_SOURCE)
        wrong["steps"][0]["answer"] = "1"
        wrong["steps"][0]["changes"][0]["calculations"][1]["result"] = 2
        with self.assertRaises(TutorialError):
            finalize_generated_tutorial(wrong, "C", POINTER_SOURCE)

        app = self.study_app()
        app.session_state["quiz_data"] = result.model_dump()
        app.session_state["study_language"] = "C"
        app.session_state["study_source"] = POINTER_SOURCE
        app.session_state["study_problem"] = "출력값은?"
        app.session_state["current_step_idx"] = 1
        app.session_state["awaiting_next"] = True
        app.session_state["outcomes"] = [{"status": "correct", "wrong_count": 0}]
        app.run(timeout=15)
        self.assertTrue(any("맞았습니다" in item.value for item in app.success))
        self.assertTrue(any("6 → 1" in item.value for item in app.markdown))
        self.assertTrue(any(button.label == "학습 마치기" for button in app.button))

    def test_output_problem_requires_output_question(self):
        self.assertEqual(output_lines(POINTER_SOURCE, "C"), {14})
        self.assertEqual(output_lines("#print(0)\nprint(1)", "Python"), {2})
        self.assertEqual(output_lines("std::cout << x;", "C++"), {1})
        self.assertEqual(output_lines("System.out.println(x);", "Java"), {1})

        generated = {
            "language": "C", "steps": [{
                "line_number": 13, "question_kind": "value_after", "target": "num",
                "context": "함수 호출 후", "answer": "1", "distractors": ["0", "4"],
                "hint": "배열의 세 번째 값을 살펴보세요.",
                "explanation": "num은 6에서 1로 바뀝니다.",
                "changes": [{"target": "num", "before": "6", "after": "1", "calculations": []}],
            }],
            "line_notes": [{"line_number": 4, "note": "i=2에서 arr[2]는 4에서 1이 됩니다."}],
        }
        with self.assertRaises(GenerationValidationError) as raised:
            finalize_generated_tutorial(generated, "C", POINTER_SOURCE, "출력값은?")
        self.assertIn("출력", raised.exception.diagnostics["reason"])

        generated["steps"].append({
            "line_number": 14, "question_kind": "output_this_step", "target": "printf(\"%d\", num)",
            "context": "num에 arr[2]를 넣은 뒤", "answer": "1", "distractors": ["0", "4"],
            "hint": "printf가 받는 num의 값을 확인하세요.",
            "explanation": "arr[2]가 1이므로 printf는 1을 출력합니다.", "changes": [],
        })
        result = finalize_generated_tutorial(generated, "C", POINTER_SOURCE, "출력값은?")
        self.assertEqual(result.steps[-1].line_number, 14)
        self.assertEqual(result.steps[-1].answer, "1")

        generated["steps"][0]["answer"] = "2"
        with self.assertLogs("execution_trace_tutor", level="WARNING"):
            filtered = finalize_generated_tutorial(generated, "C", POINTER_SOURCE, "출력값은?")
        self.assertEqual(len(filtered.steps), 1)
        self.assertEqual(filtered.steps[0].step_number, 1)
        self.assertEqual(filtered.steps[0].question_kind, "output_this_step")
        self.assertEqual(filtered.excluded_steps, 1)

    def test_raw_json_fallback_builds_step_numbers(self):
        response = SimpleNamespace(
            output_parsed=None,
            status="completed",
            output_text=f"```json\n{json.dumps(generated_payload(), ensure_ascii=False)}\n```",
        )
        with patch("app.OpenAI") as client_class:
            client_class.return_value.responses.parse.return_value = response
            tutorial = generate_tutorial("test-key", "gpt-4.1-mini", "C", "x의 값", SOURCE)
        self.assertEqual(tutorial.steps[0].step_number, 1)
        self.assertEqual(tutorial.steps[0].code_line, "int x = 1;")

    def test_api_error_diagnostics_hide_key(self):
        response = httpx2.Response(
            429,
            request=httpx2.Request("POST", "https://example.test/responses"),
            headers={"x-request-id": "request-123"},
        )
        exc = RateLimitError(
            "Rate limit",
            response=response,
            body={"error": {"code": "rate_limit_exceeded", "message": "Too many; key=secret-key"}},
        )
        details = api_error_diagnostics(exc, "secret-key", "gpt-4.1-mini")
        self.assertEqual(details["http_status"], 429)
        self.assertEqual(details["api_status"], "rate_limit_exceeded")
        self.assertEqual(details["request_id"], "request-123")
        self.assertNotIn("secret-key", json.dumps(details))
        self.assertIn("한도", describe_api_error(exc))

    def test_server_and_auth_errors(self):
        for code, cls in ((500, InternalServerError), (401, AuthenticationError)):
            with self.subTest(code=code):
                response = httpx2.Response(
                    code, request=httpx2.Request("POST", "https://example.test/responses")
                )
                exc = cls("Failure", response=response, body={"error": {"message": "Failure"}})
                self.assertEqual(api_error_diagnostics(exc, "test-key", "gpt-4.1-mini")["http_status"], code)
                self.assertIn("OpenAI", describe_api_error(exc))

    def test_schema_400_identifies_app_error(self):
        response = httpx2.Response(
            400, request=httpx2.Request("POST", "https://example.test/responses")
        )
        exc = BadRequestError(
            "Bad request", response=response,
            body={"error": {"message": "Invalid schema for response_format", "param": "text.format.schema"}},
        )
        self.assertIn("앱의 OpenAI 응답 형식", describe_api_error(exc))

    def test_invalid_trace_is_rejected(self):
        cases = [
            (payload(language="Java"), "C", SOURCE),
            (payload(steps=[]), "C", SOURCE),
            (payload(steps=[{**STEP, "line_number": 99}]), "C", SOURCE),
            (payload(steps=[{**STEP, "code_line": "wrong"}]), "C", SOURCE),
            (payload(steps=[{**STEP, "line_number": 1, "code_line": ""}]), "C", SOURCE),
            (payload(steps=[{**STEP, "question": " "}]), "C", SOURCE),
            (payload(steps=[{**STEP, "step_number": 2}]), "C", SOURCE),
            (payload(steps=[{**STEP, "choices": ["1", "2"]}]), "C", SOURCE),
            (payload(steps=[{**STEP, "choices": ["1", "1", "2"]}]), "C", SOURCE),
            (payload(steps=[{**STEP, "choices": ["0", "2", "3"]}]), "C", SOURCE),
        ]
        for value, language, source in cases:
            with self.subTest(value=value), self.assertRaises(TutorialError):
                validate_tutorial(value, language, source)

    def test_state_transitions(self):
        state = {
            "quiz_data": {"steps": [STEP.copy(), {**STEP, "step_number": 2, "line_number": 3}]},
            "current_step_idx": 0,
            "hint_opened": False,
            "feedback": None,
            "last_result": None,
        }
        self.assertFalse(advance_step(state, passed=True))
        self.assertFalse(submit_answer(state, "2"))
        self.assertEqual(state["current_step_idx"], 0)
        state["hint_opened"] = True
        self.assertFalse(submit_answer(state, "2"))
        self.assertTrue(state["hint_opened"])
        self.assertTrue(submit_answer(state, " 1 \n"))
        self.assertEqual(state["current_step_idx"], 1)
        self.assertTrue(state["awaiting_next"])
        self.assertFalse(submit_answer(state, "1"))
        self.assertFalse(state["hint_opened"])
        self.assertTrue(state["last_result"]["explanation"])
        self.assertEqual(state["last_result"]["step_number"], 1)
        self.assertEqual(state["last_result"]["line_number"], 2)
        self.assertTrue(acknowledge_result(state))
        state["hint_opened"] = True
        self.assertTrue(advance_step(state, passed=True))
        self.assertEqual(state["current_step_idx"], 2)
        self.assertTrue(state["awaiting_next"])
        self.assertFalse(advance_step(state, passed=True))
        self.assertTrue(acknowledge_result(state))
        self.assertFalse(acknowledge_result(state))

    def test_answer_format_and_highlight(self):
        self.assertEqual(normalize_answer(" a\r\nb \n"), "a\nb")
        self.assertNotEqual(normalize_answer("A"), normalize_answer("a"))
        self.assertNotEqual(normalize_answer("a b"), normalize_answer("ab"))
        code = "\n<script>\n</pre>\n"
        for language in ("C", "C++", "Java", "Python"):
            with self.subTest(language=language):
                rendered = highlight_source(code, language, 2)
                self.assertIn("&lt;", rendered)
                self.assertIn("&gt;", rendered)
                self.assertNotIn("<script>", rendered)
                self.assertIn('<span class="hll"><span class="linenos">2</span>', rendered)

    def test_streamlit_hint_and_completion(self):
        app = self.study_app()
        app.session_state["quiz_data"] = {
            "language": "C",
            "steps": [STEP.copy()],
            "annotated_code": SOURCE + "// 복습",
        }
        app.session_state["study_language"] = "C"
        app.session_state["study_source"] = SOURCE
        app.session_state["study_problem"] = "x의 값"
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len([button for button in app.button if (button.key or "").startswith("choice_")]), 3)
        self.assertEqual(len([button for button in app.button if "Pass" in button.label]), 0)
        app.button(key="choice_0_0_1").click().run()
        self.assertEqual(app.session_state["current_step_idx"], 0)
        self.assertTrue(any("정답" in item.value for item in app.warning))
        app.button(key="hint_0_0").click().run()
        self.assertEqual(len([button for button in app.button if "Pass" in button.label]), 1)
        app.button(key="pass_0_0").click().run()
        self.assertEqual(app.session_state["current_step_idx"], 1)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("패스했습니다" in item.value for item in app.success))
        app.button(key="next_0_0").click().run()
        self.assertTrue(any("모든 단계를 완료" in item.value for item in app.success))

    def test_streamlit_correct_choice_advances(self):
        app = self.study_app()
        app.session_state["quiz_data"] = {
            "language": "C",
            "steps": [STEP.copy()],
            "annotated_code": SOURCE + "// 복습",
        }
        app.session_state["study_language"] = "C"
        app.session_state["study_source"] = SOURCE
        app.session_state["study_problem"] = "x의 값"
        app.run()
        app.button(key="choice_0_0_2").click().run()
        self.assertEqual(app.session_state["current_step_idx"], 1)
        self.assertTrue(app.session_state["awaiting_next"])
        self.assertTrue(any("맞았습니다" in item.value for item in app.success))
        self.assertTrue(any("현재 2행" in item.value for item in app.markdown))
        self.assertTrue(any(button.label == "학습 마치기" for button in app.button))
        self.assertEqual(len(app.exception), 0)

    def test_repeated_line_does_not_claim_total_executions(self):
        app = self.study_app()
        app.session_state["quiz_data"] = {
            "language": "C",
            "steps": [STEP.copy(), {**STEP, "step_number": 2, "answer": "2"}],
            "annotated_code": SOURCE + "// 복습",
        }
        app.session_state["study_language"] = "C"
        app.session_state["study_source"] = SOURCE
        app.session_state["study_problem"] = "반복 실행"
        app.session_state["current_step_idx"] = 1
        app.session_state["last_result"] = {
            "step_number": 1, "line_number": 2, "answer": "1", "explanation": "첫 실행"
        }
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertFalse(any("총 2회 중 2번째 실행" in item.value for item in app.markdown))
        self.assertTrue(any("현재 2행" in item.value for item in app.markdown))
        self.assertTrue(any("지난 단계 다시 보기" in item.label for item in app.expander))
        self.assertEqual(list(app.dataframe[0].value["정답"]), ["1"])

    def test_completed_questions_remain_available_for_review(self):
        app = self.study_app()
        app.session_state["quiz_data"] = {
            "language": "C",
            "steps": [STEP.copy(), {**STEP, "step_number": 2, "answer": "2"}],
            "annotated_code": SOURCE + "// 복습",
        }
        app.session_state["study_language"] = "C"
        app.session_state["study_source"] = SOURCE
        app.session_state["study_problem"] = "반복 실행"
        app.session_state["current_step_idx"] = 2
        app.session_state["completed_celebrated"] = True
        app.run(timeout=15)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(list(app.dataframe[0].value["정답"]), ["1", "2"])
        self.assertTrue(any("지난 단계 다시 보기 (2개 완료)" in item.label for item in app.expander))

    def test_generation_failure_preserves_study(self):
        store = self.fake_store()
        with patch("app.google_owner", return_value="google:test"), patch("app.make_store", return_value=store):
            app = AppTest.from_string("import app\napp.main()").run(timeout=15)
            app.text_area[0].set_value("x의 값은?")
            app.text_area[1].set_value(SOURCE)
            with patch("app.OpenAI") as client_class:
                client_class.return_value.responses.parse.return_value = SimpleNamespace(
                    output_parsed=GeneratedTutorial.model_validate(generated_payload()), status="completed"
                )
                app.button(key="FormSubmitter:generate_form-핵심 문제 생성").click().run(timeout=15)
                finish_generation(app)
            self.assertEqual(app.session_state["current_step_idx"], 0)
            self.assertEqual(app.session_state["generation_count"], 2)
            app.button(key="hint_2_0").click().run(timeout=15)
            with patch("app.OpenAI") as client_class:
                client_class.return_value.responses.parse.return_value = SimpleNamespace(
                    output_parsed=None, status="completed", output_text="bad"
                )
                app.button(key="FormSubmitter:generate_form-핵심 문제 생성").click().run(timeout=15)
                finish_generation(app)
            self.assertEqual(app.session_state["generation_count"], 2)
            self.assertTrue(app.session_state["hint_opened"])
            self.assertEqual(app.session_state["quiz_data"]["steps"][0]["answer"], "1")
            self.assertEqual(len(app.exception), 0)

    def test_streamlit_api_error_details_hide_key(self):
        store = self.fake_store()
        store.get_key.return_value = "secret-key"
        with patch("app.google_owner", return_value="google:test"), patch("app.make_store", return_value=store):
            app = AppTest.from_string("import app\napp.main()").run(timeout=15)
            app.text_area[0].set_value("x의 값은?")
            app.text_area[1].set_value(SOURCE)
            response = httpx2.Response(
                429, request=httpx2.Request("POST", "https://example.test/responses")
            )
            with patch("app.OpenAI") as client_class:
                client_class.return_value.responses.parse.side_effect = RateLimitError(
                    "Too many", response=response,
                    body={"error": {"code": "rate_limit_exceeded", "message": "Busy; key=secret-key"}},
                )
                app.button(key="FormSubmitter:generate_form-핵심 문제 생성").click().run(timeout=15)
                finish_generation(app)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("한도" in item.value for item in app.error))
        self.assertTrue(any("생성 오류 상세" in item.label for item in app.expander))
        self.assertEqual(len(app.json), 1)
        self.assertEqual(json.loads(app.json[0].value)["api_status"], "rate_limit_exceeded")
        self.assertNotIn("secret-key", app.json[0].value)

    def test_streamlit_choice_error_shows_reason(self):
        store = self.fake_store()
        generated = generated_payload()
        generated["steps"][0]["distractors"] = ["1", "2"]
        invalid = SimpleNamespace(
            output_parsed=GeneratedTutorial.model_validate(generated), status="completed"
        )
        with patch("app.google_owner", return_value="google:test"), patch("app.make_store", return_value=store):
            app = AppTest.from_string("import app\napp.main()").run(timeout=15)
            app.text_area[0].set_value("x의 값은?")
            app.text_area[1].set_value(SOURCE)
            with patch("app.OpenAI") as client_class:
                client_class.return_value.responses.parse.return_value = invalid
                app.button(key="FormSubmitter:generate_form-핵심 문제 생성").click().run(timeout=15)
                finish_generation(app)
                self.assertEqual(client_class.return_value.responses.parse.call_count, 2)
        self.assertTrue(any("1번 문항" in item.value for item in app.error))
        self.assertTrue(any("생성 오류 상세" in item.label for item in app.expander))
        self.assertEqual(json.loads(app.json[0].value)["reason"], "중복 선택지 있음")
        self.assertIsNone(app.session_state["quiz_data"])

    def test_contact_links_are_encoded(self):
        html = contact_html("be0128st@gmail.com")
        self.assertIn("to=be0128st%40gmail.com", html)
        self.assertIn("subject=%5B%EC%8B%A4%ED%96%89", html)
        self.assertIn("개발자에게 문의하기", html)
        self.assertIn("mailto:be0128st@gmail.com", html)


if __name__ == "__main__":
    unittest.main()
