"""로그인 사용자별 API 키와 학습 기록을 Supabase에 보관한다."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from cryptography.fernet import Fernet, InvalidToken
from supabase import create_client


class StorageError(Exception):
    """화면에 표시해도 안전한 저장소 오류."""


class StorageConflict(StorageError):
    """다른 기기에서 같은 학습 기록을 먼저 변경했다."""


class StorageNotFound(StorageError):
    """로그인 사용자가 이 기록에 접근할 수 없다."""


class KeyUnavailable(StorageError):
    """암호화된 개인 API 키를 사용할 수 없다."""


def google_owner(user: Any) -> str | None:
    """Google OIDC가 확인한 고정 식별자만 소유자 키로 사용한다."""
    if not getattr(user, "is_logged_in", False):
        return None
    issuer = user.get("iss")
    subject = user.get("sub")
    if issuer not in ("https://accounts.google.com", "accounts.google.com"):
        return None
    if not isinstance(subject, str) or not subject.strip():
        return None
    return f"google:{subject}"


def fresh_progress(steps: int) -> dict[str, Any]:
    """풀이 상태와 문항별 결과를 처음 상태로 만든다."""
    return {
        "current_step_idx": 0,
        "hint_opened": False,
        "outcomes": [{"status": "pending", "wrong_count": 0} for _ in range(steps)],
    }


def check_progress(progress: Any, steps: int) -> dict[str, Any]:
    """저장된 진행 상태를 UI에 적용하기 전에 검사한다."""
    if not isinstance(progress, dict):
        raise StorageError("저장된 학습 상태를 읽을 수 없습니다.")
    index = progress.get("current_step_idx")
    outcomes = progress.get("outcomes")
    hint = progress.get("hint_opened")
    if (
        type(index) is not int
        or not 0 <= index <= steps
        or type(hint) is not bool
        or not isinstance(outcomes, list)
        or len(outcomes) != steps
    ):
        raise StorageError("저장된 학습 상태를 읽을 수 없습니다.")
    for number, item in enumerate(outcomes):
        if (
            not isinstance(item, dict)
            or item.get("status") not in ("pending", "correct", "passed")
            or type(item.get("wrong_count")) is not int
            or item["wrong_count"] < 0
            or (number < index and item["status"] == "pending")
            or (number >= index and item["status"] != "pending")
        ):
            raise StorageError("저장된 학습 상태를 읽을 수 없습니다.")
    if index == steps and hint:
        raise StorageError("저장된 학습 상태를 읽을 수 없습니다.")
    return progress


class ArchiveStore:
    """모든 작업에 확인된 사용자 소유자를 함께 전달한다."""

    def __init__(self, url: str, secret_key: str, encryption_key: str):
        if not url or not secret_key or not encryption_key:
            raise StorageError("저장소 설정이 빠졌습니다. 배포 Secrets를 확인해 주세요.")
        try:
            self.cipher = Fernet(encryption_key.encode())
            self.client = create_client(url, secret_key)
            # 새 sb_secret 키는 JWT가 아니므로 Data API에는 apikey 헤더만 보낸다.
            if secret_key.startswith("sb_secret_"):
                self.client.options.headers.pop("Authorization", None)
        except Exception as exc:
            raise StorageError("저장소 설정이 올바르지 않습니다.") from None

    def has_key(self, owner: str) -> bool:
        try:
            result = (
                self.client.table("user_settings")
                .select("key_ciphertext")
                .eq("owner_id", owner)
                .limit(1)
                .execute()
            )
        except Exception:
            raise StorageError("API 키 등록 상태를 불러오지 못했습니다.") from None
        return bool(result.data and result.data[0].get("key_ciphertext"))

    def save_key(self, owner: str, key: str) -> None:
        key = key.strip()
        if not key:
            raise StorageError("OpenAI API 키를 입력해 주세요.")
        encrypted = self.cipher.encrypt(
            json.dumps({"owner": owner, "key": key}).encode("utf-8")
        ).decode("ascii")
        try:
            self.client.table("user_settings").upsert(
                {"owner_id": owner, "key_ciphertext": encrypted, "updated_at": _utc_now()},
                on_conflict="owner_id",
            ).execute()
        except Exception:
            raise StorageError("API 키를 저장하지 못했습니다. 다시 시도해 주세요.") from None

    def get_key(self, owner: str) -> str:
        try:
            result = (
                self.client.table("user_settings")
                .select("key_ciphertext")
                .eq("owner_id", owner)
                .limit(1)
                .execute()
            )
        except Exception:
            raise StorageError("API 키를 불러오지 못했습니다.") from None
        if not result.data or not result.data[0].get("key_ciphertext"):
            raise KeyUnavailable("API 키를 먼저 등록해 주세요.")
        try:
            decoded = self.cipher.decrypt(result.data[0]["key_ciphertext"].encode())
            value = json.loads(decoded)
            if value.get("owner") == owner and isinstance(value.get("key"), str) and value["key"]:
                return value["key"]
        except (InvalidToken, ValueError, TypeError, UnicodeError, AttributeError):
            pass
        raise KeyUnavailable("저장된 API 키를 읽을 수 없습니다. 키를 다시 등록해 주세요.")

    def delete_key(self, owner: str) -> None:
        try:
            self.client.table("user_settings").update(
                {"key_ciphertext": None, "updated_at": _utc_now()}
            ).eq("owner_id", owner).execute()
        except Exception:
            raise StorageError("API 키를 삭제하지 못했습니다.") from None

    def insert_tutorial(self, owner: str, record: dict[str, Any]) -> dict[str, Any]:
        """같은 ID 재시도는 기존 결과를 확인하며 타인 행을 덮어쓰지 않는다."""
        row = {**record, "owner_id": owner}
        try:
            result = self.client.table("tutorials").insert(row).execute()
            if result.data:
                return result.data[0]
        except Exception:
            # 응답이 끊겼지만 첫 INSERT가 성공했을 수 있다.
            existing = self.get_tutorial(owner, record["id"], missing_ok=True)
            if existing and all(existing.get(k) == record.get(k) for k in ("problem", "source", "quiz")):
                return existing
        raise StorageError("문제 기록을 저장하지 못했습니다. 다시 시도해 주세요.")

    def get_tutorial(self, owner: str, record_id: str, *, missing_ok: bool = False) -> dict[str, Any] | None:
        try:
            result = (
                self.client.table("tutorials")
                .select("*")
                .eq("id", record_id)
                .eq("owner_id", owner)
                .limit(1)
                .execute()
            )
        except Exception:
            raise StorageError("문제 기록을 불러오지 못했습니다.") from None
        if result.data:
            return result.data[0]
        if missing_ok:
            return None
        raise StorageNotFound("이 문제 기록을 찾을 수 없습니다.")

    def list_tutorials(self, owner: str, limit: int = 100) -> list[dict[str, Any]]:
        try:
            result = (
                self.client.table("tutorials")
                .select("id,language,problem,progress,version,created_at,updated_at,completed_at")
                .eq("owner_id", owner)
                .order("updated_at", desc=True)
                .limit(limit)
                .execute()
            )
        except Exception:
            raise StorageError("문제 보관함을 불러오지 못했습니다.") from None
        return result.data or []

    def update_progress(self, owner: str, record_id: str, version: int, progress: dict[str, Any]) -> int:
        """버전이 같을 때만 갱신해 다른 기기의 진행 상황을 지킨다."""
        try:
            result = (
                self.client.table("tutorials")
                .update(
                    {
                        "progress": progress,
                        "version": version + 1,
                        "updated_at": _utc_now(),
                        "completed_at": _utc_now() if progress["current_step_idx"] == len(progress["outcomes"]) else None,
                    }
                )
                .eq("id", record_id)
                .eq("owner_id", owner)
                .eq("version", version)
                .execute()
            )
        except Exception:
            raise StorageError("풀이 상태를 저장하지 못했습니다. 다시 시도해 주세요.") from None
        if len(result.data or []) != 1:
            raise StorageConflict("다른 기기에서 이 문제를 먼저 변경했습니다. 최신 상태를 불러와 주세요.")
        return version + 1

    def delete_tutorial(self, owner: str, record_id: str) -> None:
        try:
            result = (
                self.client.table("tutorials")
                .delete()
                .eq("id", record_id)
                .eq("owner_id", owner)
                .execute()
            )
        except Exception:
            raise StorageError("문제 기록을 삭제하지 못했습니다.") from None
        if not result.data:
            raise StorageNotFound("이 문제 기록을 찾을 수 없습니다.")


def new_tutorial_record(problem: str, language: str, source: str, model: str, quiz: dict[str, Any]) -> dict[str, Any]:
    """한 번 생성한 내용을 재시도할 때도 같은 ID로 저장한다."""
    return {
        "id": str(uuid4()),
        "problem": problem,
        "language": language,
        "source": source,
        "model": model,
        "quiz": quiz,
        "progress": fresh_progress(len(quiz["steps"])),
        "version": 0,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
