"""외부 계정 없이 키 보호와 사용자별 저장을 검증한다."""

import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from cryptography.fernet import Fernet
from streamlit.testing.v1 import AppTest
from types import SimpleNamespace as Namespace

from app import GeneratedTutorial, apply_record, open_hint, submit_answer
from storage import (
    ArchiveStore,
    KeyUnavailable,
    StorageConflict,
    StorageError,
    StorageNotFound,
    check_progress,
    fresh_progress,
    google_owner,
    new_tutorial_record,
)
from test_app import SOURCE, generated_payload, payload


class FakeQuery:
    def __init__(self, client, table):
        self.client = client
        self.table_name = table
        self.filters = []
        self.operation = "select"
        self.value = None
        self.maximum = None
        self.descending = False
        self.order_field = None

    def select(self, *_):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def limit(self, count):
        self.maximum = count
        return self

    def order(self, field, desc=False):
        self.order_field = field
        self.descending = desc
        return self

    def insert(self, value):
        self.operation, self.value = "insert", value
        return self

    def upsert(self, value, on_conflict=None):
        self.operation, self.value = "upsert", value
        return self

    def update(self, value):
        self.operation, self.value = "update", value
        return self

    def delete(self):
        self.operation = "delete"
        return self

    def execute(self):
        rows = self.client.tables[self.table_name]
        if self.client.fail_next or self.client.fail_operation == self.operation:
            self.client.fail_next = False
            self.client.fail_operation = None
            raise RuntimeError("DB unavailable")
        selected = [
            row for row in rows
            if all(row.get(key) == value for key, value in self.filters)
        ]
        if self.operation == "insert":
            if any(row.get("id") == self.value.get("id") for row in rows):
                raise RuntimeError("duplicate id")
            row = deepcopy(self.value)
            rows.append(row)
            selected = [row]
        elif self.operation == "upsert":
            selected = [row for row in rows if row.get("owner_id") == self.value["owner_id"]]
            if selected:
                selected[0].update(deepcopy(self.value))
            else:
                row = deepcopy(self.value)
                rows.append(row)
                selected = [row]
        elif self.operation == "update":
            for row in selected:
                row.update(deepcopy(self.value))
        elif self.operation == "delete":
            for row in selected:
                rows.remove(row)
        if self.order_field:
            selected.sort(key=lambda row: row.get(self.order_field, ""), reverse=self.descending)
        if self.maximum is not None:
            selected = selected[:self.maximum]
        return SimpleNamespace(data=deepcopy(selected))


class FakeClient:
    def __init__(self):
        self.tables = {"user_settings": [], "tutorials": []}
        self.options = SimpleNamespace(headers={"Authorization": "Bearer sb_secret_fake", "apiKey": "sb_secret_fake"})
        self.fail_next = False
        self.fail_operation = None

    def table(self, name):
        return FakeQuery(self, name)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.master_key = Fernet.generate_key().decode()
        with patch("storage.create_client", return_value=self.client):
            self.store = ArchiveStore("https://example.supabase.co", "sb_secret_fake", self.master_key)

    def test_google_owner_must_be_verified_google_subject(self):
        valid = {"iss": "https://accounts.google.com", "sub": "123"}
        self.assertEqual(google_owner(SimpleNamespace(is_logged_in=True, get=valid.get)), "google:123")
        for value in (
            SimpleNamespace(is_logged_in=False, get=valid.get),
            SimpleNamespace(is_logged_in=True, get={"iss": "https://other", "sub": "123"}.get),
            SimpleNamespace(is_logged_in=True, get={"iss": "https://accounts.google.com"}.get),
        ):
            self.assertIsNone(google_owner(value))

    def test_key_is_encrypted_scoped_and_deletable(self):
        self.store.save_key("google:a", "sk-test-secret")
        row = self.client.tables["user_settings"][0]
        self.assertNotIn("sk-test-secret", json.dumps(row))
        self.assertTrue(self.store.has_key("google:a"))
        self.assertFalse(self.store.has_key("google:b"))
        self.assertEqual(self.store.get_key("google:a"), "sk-test-secret")
        with self.assertRaises(KeyUnavailable):
            self.store.get_key("google:b")
        self.client.tables["user_settings"].append({"owner_id": "google:b", "key_ciphertext": row["key_ciphertext"]})
        with self.assertRaises(KeyUnavailable):
            self.store.get_key("google:b")
        self.store.delete_key("google:a")
        with self.assertRaises(KeyUnavailable):
            self.store.get_key("google:a")

    def test_bad_master_key_does_not_fall_back(self):
        self.store.save_key("google:a", "sk-test-secret")
        with patch("storage.create_client", return_value=self.client):
            other = ArchiveStore("https://example.supabase.co", "sb_secret_fake", Fernet.generate_key().decode())
        with self.assertRaises(KeyUnavailable):
            other.get_key("google:a")
        self.client.tables["user_settings"][0]["key_ciphertext"] = "tampered"
        with self.assertRaises(KeyUnavailable):
            self.store.get_key("google:a")

    def test_key_update_failure_keeps_old_key(self):
        self.store.save_key("google:a", "sk-original")
        self.client.fail_next = True
        with self.assertRaises(StorageError):
            self.store.save_key("google:a", "sk-replacement")
        self.assertEqual(self.store.get_key("google:a"), "sk-original")

    def test_owner_filter_idempotent_insert_and_version_conflict(self):
        self.store.save_key("google:a", "sk-a")
        self.store.save_key("google:b", "sk-b")
        record = new_tutorial_record("x의 값", "C", SOURCE, "gpt-4.1-mini", payload())
        saved = self.store.insert_tutorial("google:a", record)
        self.assertEqual(saved["owner_id"], "google:a")
        self.assertEqual(self.store.insert_tutorial("google:a", record)["id"], record["id"])
        self.assertEqual(len(self.client.tables["tutorials"]), 1)
        with self.assertRaises(StorageNotFound):
            self.store.get_tutorial("google:b", record["id"])
        with self.assertRaises(StorageNotFound):
            self.store.delete_tutorial("google:b", record["id"])
        first = fresh_progress(1)
        first["hint_opened"] = True
        self.assertEqual(self.store.update_progress("google:a", record["id"], 0, first), 1)
        with self.assertRaises(StorageConflict):
            self.store.update_progress("google:a", record["id"], 0, first)
        self.assertEqual(self.store.get_tutorial("google:a", record["id"])["progress"], first)
        self.assertEqual(len(self.store.list_tutorials("google:a")), 1)
        self.assertEqual(len(self.store.list_tutorials("google:b")), 0)

    def test_failed_progress_save_leaves_ui_at_same_step(self):
        self.store.save_key("google:a", "sk-a")
        record = new_tutorial_record("x의 값", "C", SOURCE, "gpt-4.1-mini", payload())
        saved = self.store.insert_tutorial("google:a", record)
        state = {"owner_id": "google:a", "generation_count": 0, "needs_reload": False}
        apply_record(state, saved)
        with patch("app.make_store", return_value=self.store):
            self.client.fail_next = True
            with self.assertRaises(StorageError):
                open_hint(state)
            self.assertFalse(state["hint_opened"])
            open_hint(state)
            self.assertTrue(state["hint_opened"])
            self.assertFalse(submit_answer(state, "0"))
            self.assertEqual(state["outcomes"][0]["wrong_count"], 1)
            self.assertTrue(submit_answer(state, "1"))
        self.assertEqual(state["current_step_idx"], 1)
        self.assertEqual(state["outcomes"][0]["status"], "correct")
        self.assertEqual(state["active_version"], 3)

    def test_progress_validation_rejects_corruption(self):
        valid = fresh_progress(2)
        self.assertEqual(check_progress(valid, 2), valid)
        for broken in (
            {**valid, "current_step_idx": 3},
            {**valid, "outcomes": []},
            {**valid, "hint_opened": "yes"},
        ):
            with self.assertRaises(StorageError):
                check_progress(broken, 2)

    def test_archive_reopens_on_another_session_without_openai(self):
        self.store.save_key("google:a", "sk-a")
        record = new_tutorial_record("x의 값", "C", SOURCE, "gpt-4.1-mini", payload())
        self.store.insert_tutorial("google:a", record)
        with patch("app.google_owner", return_value="google:a"), patch("app.make_store", return_value=self.store), patch("app.generate_tutorial") as generate:
            first = AppTest.from_string("import app\napp.main()").run(timeout=15)
            next(button for button in first.button if button.label == "내 문제 보관함").click().run(timeout=15)
            first.button(key=f"open_{record['id']}").click().run(timeout=15)
            self.assertEqual(first.session_state["current_step_idx"], 0)
            choice_key = f"choice_{first.session_state['generation_count']}_0_2"
            first.button(key=choice_key).click().run(timeout=15)
            self.assertEqual(first.session_state["current_step_idx"], 1)

            second = AppTest.from_string("import app\napp.main()").run(timeout=15)
            next(button for button in second.button if button.label == "내 문제 보관함").click().run(timeout=15)
            second.button(key=f"open_{record['id']}").click().run(timeout=15)
            self.assertEqual(second.session_state["current_step_idx"], 1)
            self.assertTrue(any("모든 단계를 완료" in item.value for item in second.success))
            generate.assert_not_called()

    def test_generated_result_retries_database_without_openai(self):
        self.store.save_key("google:a", "sk-a")
        parsed = GeneratedTutorial.model_validate(generated_payload())
        with patch("app.google_owner", return_value="google:a"), patch("app.make_store", return_value=self.store), patch("app.OpenAI") as client_class:
            client_class.return_value.responses.parse.return_value = Namespace(output_parsed=parsed, status="completed")
            page = AppTest.from_string("import app\napp.main()").run(timeout=15)
            page.text_area[0].set_value("x의 값")
            page.text_area[1].set_value(SOURCE)
            self.client.fail_operation = "insert"
            page.button(key="FormSubmitter:generate_form-핵심 문제 생성").click().run(timeout=15)
            self.assertIsNone(page.session_state["quiz_data"])
            self.assertIsNotNone(page.session_state["pending_generation"])
            self.assertEqual(client_class.return_value.responses.parse.call_count, 1)
            next(button for button in page.button if button.label == "생성된 문제 저장 다시 시도").click().run(timeout=15)
            self.assertIsNotNone(page.session_state["quiz_data"])
            self.assertIsNone(page.session_state["pending_generation"])
            self.assertEqual(client_class.return_value.responses.parse.call_count, 1)


if __name__ == "__main__":
    unittest.main()
