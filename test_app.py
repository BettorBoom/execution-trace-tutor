"""API 키 없이 실행 가능한 핵심 동작 검증."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest
from openai import AuthenticationError, InternalServerError, RateLimitError
import httpx2

from app import (
    GeneratedTutorial,
    TutorialError,
    advance_step,
    api_error_diagnostics,
    describe_api_error,
    finalize_generated_tutorial,
    generate_tutorial,
    highlight_source,
    normalize_answer,
    parse_json_object,
    submit_answer,
    validate_tutorial,
    build_prompt,
    contact_html,
)


SOURCE = "\nint x = 1;\nint x = 1;\n"
STEP = {
    "step_number": 1,
    "line_number": 2,
    "code_line": "int x = 1;",
    "question": "이 줄 실행 후 x의 값은? 숫자로 답하세요.",
    "choices": ["0", "1", "2"],
    "answer": "1",
    "hint": "대입 연산자를 살펴보세요.",
    "explanation": "x에 1을 대입합니다.",
}


def payload(language="C", steps=None, source=SOURCE):
    return {
        "language": language,
        "steps": [STEP.copy()] if steps is None else steps,
        "annotated_code": source + "// 복습",
    }


def generated_payload(language="C", steps=None, source=SOURCE):
    value = payload(language, steps, source)
    value["steps"] = [
        {key: item for key, item in step.items() if key not in ("step_number", "code_line")}
        for step in value["steps"]
    ]
    return value


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
        self.assertEqual([step.step_number for step in tutorial.steps], [1, 2])
        self.assertEqual([step.code_line for step in tutorial.steps], ["int x = 1;", "int x = 1;"])
        self.assertEqual(request["model"], "gpt-4.1-mini")
        self.assertIs(request["text_format"], GeneratedTutorial)
        self.assertFalse(request["store"])
        self.assertEqual(client_class.call_args.kwargs["timeout"], 60.0)
        self.assertEqual(client_class.call_args.kwargs["max_retries"], 0)

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
        self.assertIn("이번 실행에서 출력되는 값", prompt)

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
        self.assertFalse(state["hint_opened"])
        self.assertTrue(state["last_result"]["explanation"])
        self.assertEqual(state["last_result"]["step_number"], 1)
        self.assertEqual(state["last_result"]["line_number"], 2)
        state["hint_opened"] = True
        self.assertTrue(advance_step(state, passed=True))
        self.assertEqual(state["current_step_idx"], 2)
        self.assertFalse(advance_step(state, passed=True))

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
        self.assertEqual(len([button for button in app.button if button.key.startswith("choice_")]), 3)
        self.assertEqual(len([button for button in app.button if "Pass" in button.label]), 0)
        app.button(key="choice_0_0_1").click().run()
        self.assertEqual(app.session_state["current_step_idx"], 0)
        self.assertTrue(any("정답" in item.value for item in app.warning))
        app.button(key="hint_0_0").click().run()
        self.assertEqual(len([button for button in app.button if "Pass" in button.label]), 1)
        app.button(key="pass_0_0").click().run()
        self.assertEqual(app.session_state["current_step_idx"], 1)
        self.assertEqual(len(app.exception), 0)
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
        self.assertTrue(any("직전 1단계 (2행)의 정답" in item.value for item in app.success))
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
            self.assertEqual(app.session_state["current_step_idx"], 0)
            self.assertEqual(app.session_state["generation_count"], 2)
            app.button(key="hint_2_0").click().run(timeout=15)
            with patch("app.OpenAI") as client_class:
                client_class.return_value.responses.parse.return_value = SimpleNamespace(
                    output_parsed=None, status="completed", output_text="bad"
                )
                app.button(key="FormSubmitter:generate_form-핵심 문제 생성").click().run(timeout=15)
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
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("한도" in item.value for item in app.error))
        self.assertTrue(any("API 오류 상세" in item.label for item in app.expander))
        self.assertEqual(len(app.json), 1)
        self.assertEqual(json.loads(app.json[0].value)["api_status"], "rate_limit_exceeded")
        self.assertNotIn("secret-key", app.json[0].value)

    def test_contact_links_are_encoded(self):
        html = contact_html("be0128st@gmail.com")
        self.assertIn("to=be0128st%40gmail.com", html)
        self.assertIn("subject=%5B%EC%8B%A4%ED%96%89", html)
        self.assertIn("개발자에게 문의하기", html)
        self.assertIn("mailto:be0128st@gmail.com", html)


if __name__ == "__main__":
    unittest.main()
