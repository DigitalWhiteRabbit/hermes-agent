from __future__ import annotations

import inspect
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

import gateway.profile_switching.store as store_module
from gateway.profile_switching.migrations import LATEST_SCHEMA_VERSION
from gateway.profile_switching.models import ReasonCode, ScopeKey, ScopeKind
from gateway.profile_switching.store import (
    ProfileRoutingStore,
    ProfileRoutingStoreUnavailable,
)


def _scope(thread_id: str | None = None) -> ScopeKey:
    return ScopeKey("telegram", "telegram:primary", "chat-1", thread_id)


def _reject_audit_inserts(store: ProfileRoutingStore) -> None:
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            """
            CREATE TRIGGER reject_profile_switch_audit
            BEFORE INSERT ON profile_switch_audit
            BEGIN
                SELECT RAISE(ABORT, 'injected audit failure');
            END
            """
        )


def test_migration_creates_schema_and_enables_foreign_keys(tmp_path):
    path = tmp_path / "profile-routing.db"
    store = ProfileRoutingStore(path)
    ProfileRoutingStore(path)

    with sqlite3.connect(path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        version = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()[0]

    assert {
        "schema_meta",
        "profile_bindings",
        "profile_once",
        "profile_session_bindings",
        "profile_switch_audit",
        "profile_picker_nonces",
    } <= tables
    assert version == str(LATEST_SCHEMA_VERSION)
    with closing(store._connect()) as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 250


def test_corrupt_database_and_sidecars_are_quarantined_before_recovery(tmp_path):
    path = tmp_path / "profile-routing.db"
    original = b"not a sqlite database"
    path.write_bytes(original)
    Path(f"{path}-wal").write_bytes(b"wal")
    Path(f"{path}-shm").write_bytes(b"shm")

    store = ProfileRoutingStore(path)

    assert store.quarantined_paths
    assert {item.read_bytes() for item in store.quarantined_paths} == {
        original,
        b"wal",
        b"shm",
    }
    assert path.exists()
    with sqlite3.connect(path) as conn:
        assert (
            conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            is not None
        )


def test_malformed_schema_version_is_quarantined_and_rebuilt(tmp_path):
    path = tmp_path / "profile-routing.db"
    ProfileRoutingStore(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE schema_meta SET value = 'not-an-integer' "
            "WHERE key = 'schema_version'"
        )

    recovered = ProfileRoutingStore(path)

    assert recovered.quarantined_paths
    assert any(
        item.name.startswith("profile-routing.db.corrupt-")
        for item in recovered.quarantined_paths
    )


def test_locked_database_is_unavailable_without_quarantine(tmp_path):
    path = tmp_path / "profile-routing.db"
    ProfileRoutingStore(path)
    blocker = sqlite3.connect(path, isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(ProfileRoutingStoreUnavailable):
            ProfileRoutingStore(path)
    finally:
        blocker.rollback()
        blocker.close()

    assert path.exists()
    assert not list(tmp_path.glob("*.corrupt-*"))


def test_uncreatable_database_path_is_unavailable_and_preserved(tmp_path):
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("preserve me", encoding="utf-8")

    with pytest.raises(ProfileRoutingStoreUnavailable):
        ProfileRoutingStore(parent_file / "profile-routing.db")

    assert parent_file.read_text(encoding="utf-8") == "preserve me"


def test_malformed_binding_row_is_wrapped_as_store_unavailability(
    tmp_path, monkeypatch
):
    store = ProfileRoutingStore(tmp_path / "profile-routing.db")

    class MalformedRow(dict):
        pass

    row = MalformedRow(
        scope_kind="not-a-scope",
        platform="telegram",
        account_id="telegram:primary",
        chat_id="chat-1",
        thread_id="",
        profile_name="coder",
        created_by_user_id="user-1",
        created_at=1.0,
        updated_at=1.0,
        expires_at=None,
        version=1,
    )
    monkeypatch.setattr(store, "_select_binding", lambda *_args: row)

    with pytest.raises(ProfileRoutingStoreUnavailable):
        store.get_binding(_scope(), ScopeKind.CHAT)


@pytest.mark.requires_wal
def test_connections_use_wal_when_runtime_supports_it(tmp_path):
    store = ProfileRoutingStore(tmp_path / "profile-routing.db")

    with closing(store._connect()) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

    assert mode.lower() == "wal"


def test_set_binding_upserts_and_increments_version(tmp_path):
    store = ProfileRoutingStore(tmp_path / "profile-routing.db", clock=lambda: 10.0)
    first = store.set_binding(
        _scope(),
        ScopeKind.CHAT,
        profile_name="coder",
        created_by_user_id="user-1",
    )

    store._clock = lambda: 20.0
    second = store.set_binding(
        _scope(),
        ScopeKind.CHAT,
        profile_name="research",
        created_by_user_id="user-2",
    )

    assert first.version == 1
    assert second.profile_name == "research"
    assert second.created_by_user_id == "user-2"
    assert second.created_at == 10.0
    assert second.updated_at == 20.0
    assert second.version == 2
    assert second.scope.thread_id is None

    with sqlite3.connect(store.path) as conn:
        assert (
            conn.execute("SELECT thread_id FROM profile_bindings").fetchone()[0] == ""
        )


def test_atomic_set_binding_rolls_back_when_audit_insert_fails(tmp_path):
    store = ProfileRoutingStore(tmp_path / "profile-routing.db")
    store.set_binding(
        _scope(),
        ScopeKind.CHAT,
        profile_name="research",
        created_by_user_id="user-1",
    )
    _reject_audit_inserts(store)

    with pytest.raises(ProfileRoutingStoreUnavailable):
        store.set_binding_with_audit(
            _scope(),
            ScopeKind.CHAT,
            profile_name="coder",
            created_by_user_id="user-1",
            source="command",
        )

    binding = store.get_binding(_scope(), ScopeKind.CHAT)
    assert binding is not None and binding.profile_name == "research"
    assert binding.version == 1


def test_thread_and_chat_bindings_are_distinct(tmp_path):
    store = ProfileRoutingStore(tmp_path / "profile-routing.db")
    threaded = _scope("thread-7")
    store.set_binding(
        threaded,
        ScopeKind.CHAT,
        profile_name="coder",
        created_by_user_id="user-1",
    )
    store.set_binding(
        threaded,
        ScopeKind.THREAD,
        profile_name="research",
        created_by_user_id="user-1",
    )

    chat = store.get_binding(threaded, ScopeKind.CHAT)
    thread = store.get_binding(threaded, ScopeKind.THREAD)

    assert chat is not None and chat.profile_name == "coder"
    assert chat.scope.thread_id is None
    assert thread is not None and thread.profile_name == "research"
    assert thread.scope.thread_id == "thread-7"


def test_expired_binding_is_not_returned(tmp_path):
    store = ProfileRoutingStore(tmp_path / "profile-routing.db", clock=lambda: 20.0)
    store.set_binding(
        _scope(),
        ScopeKind.CHAT,
        profile_name="coder",
        created_by_user_id="user-1",
        expires_at=20.0,
    )

    assert store.get_binding(_scope(), ScopeKind.CHAT) is None


def test_clear_binding_only_removes_requested_scope(tmp_path):
    store = ProfileRoutingStore(tmp_path / "profile-routing.db")
    threaded = _scope("thread-7")
    store.set_binding(
        threaded,
        ScopeKind.CHAT,
        profile_name="coder",
        created_by_user_id="user-1",
    )
    store.set_binding(
        threaded,
        ScopeKind.THREAD,
        profile_name="research",
        created_by_user_id="user-1",
    )

    assert store.clear_binding(threaded, ScopeKind.THREAD) is True
    assert store.clear_binding(threaded, ScopeKind.THREAD) is False
    assert store.get_binding(threaded, ScopeKind.CHAT) is not None


def test_atomic_clear_rolls_back_when_audit_insert_fails(tmp_path):
    store = ProfileRoutingStore(tmp_path / "profile-routing.db")
    binding = store.set_binding(
        _scope(),
        ScopeKind.CHAT,
        profile_name="coder",
        created_by_user_id="user-1",
    )
    _reject_audit_inserts(store)

    with pytest.raises(ProfileRoutingStoreUnavailable):
        store.clear_binding_with_audit(
            _scope(),
            ScopeKind.CHAT,
            expected_version=binding.version,
            expected_profile=binding.profile_name,
            actor_user_id="user-1",
            source="clear",
        )

    remaining = store.get_binding(_scope(), ScopeKind.CHAT)
    assert remaining is not None and remaining.profile_name == "coder"


def test_atomic_clear_rejects_stale_version_without_deleting_replacement(tmp_path):
    store = ProfileRoutingStore(tmp_path / "profile-routing.db")
    stale = store.set_binding(
        _scope(),
        ScopeKind.CHAT,
        profile_name="coder",
        created_by_user_id="user-1",
    )
    replacement = store.set_binding(
        _scope(),
        ScopeKind.CHAT,
        profile_name="research",
        created_by_user_id="user-2",
    )

    with pytest.raises(store_module.ProfileBindingChanged):
        store.clear_binding_with_audit(
            _scope(),
            ScopeKind.CHAT,
            expected_version=stale.version,
            expected_profile=stale.profile_name,
            actor_user_id="user-1",
            source="clear",
        )

    remaining = store.get_binding(_scope(), ScopeKind.CHAT)
    assert remaining == replacement
    with sqlite3.connect(store.path) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM profile_switch_audit").fetchone()[0] == 0
        )


def test_atomic_clear_rejects_delete_and_recreate_with_reset_version(tmp_path):
    store = ProfileRoutingStore(tmp_path / "profile-routing.db")
    stale = store.set_binding(
        _scope(),
        ScopeKind.CHAT,
        profile_name="coder",
        created_by_user_id="user-1",
    )
    assert store.clear_binding(_scope(), ScopeKind.CHAT)
    replacement = store.set_binding(
        _scope(),
        ScopeKind.CHAT,
        profile_name="research",
        created_by_user_id="user-2",
    )
    assert replacement.version == stale.version == 1

    with pytest.raises(store_module.ProfileBindingChanged):
        store.clear_binding_with_audit(
            _scope(),
            ScopeKind.CHAT,
            expected_version=stale.version,
            expected_profile=stale.profile_name,
            actor_user_id="user-1",
            source="clear",
        )

    assert store.get_binding(_scope(), ScopeKind.CHAT) == replacement


def test_atomic_clear_treats_expired_binding_as_logically_absent(tmp_path):
    store = ProfileRoutingStore(tmp_path / "profile-routing.db", clock=lambda: 20.0)
    store.set_binding(
        _scope(),
        ScopeKind.CHAT,
        profile_name="coder",
        created_by_user_id="user-1",
        expires_at=20.0,
    )

    assert store.clear_binding_with_audit(
        _scope(),
        ScopeKind.CHAT,
        expected_version=None,
        expected_profile=None,
        actor_user_id="user-1",
        source="clear",
    )

    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM profile_bindings").fetchone()[0] == 0


def test_claim_once_returns_value_exactly_once(tmp_path):
    store = ProfileRoutingStore(tmp_path / "profile-routing.db")
    store.set_once(
        _scope(),
        profile_name="research",
        created_by_user_id="user-1",
    )

    claimed = store.claim_once(_scope())

    assert claimed is not None
    assert claimed.profile_name == "research"
    assert claimed.scope.thread_id is None
    assert store.claim_once(_scope()) is None


def test_atomic_set_once_rolls_back_when_audit_insert_fails(tmp_path):
    store = ProfileRoutingStore(tmp_path / "profile-routing.db")
    store.set_once(
        _scope(),
        profile_name="research",
        created_by_user_id="user-1",
    )
    _reject_audit_inserts(store)

    with pytest.raises(ProfileRoutingStoreUnavailable):
        store.set_once_with_audit(
            _scope(),
            profile_name="coder",
            created_by_user_id="user-1",
            source="once",
        )

    once = store.claim_once(_scope())
    assert once is not None and once.profile_name == "research"


def test_empty_thread_id_is_returned_as_absent_from_once_binding(tmp_path):
    store = ProfileRoutingStore(tmp_path / "profile-routing.db")
    store.set_once(
        _scope(""),
        profile_name="research",
        created_by_user_id="user-1",
    )

    claimed = store.claim_once(_scope(""))

    assert claimed is not None
    assert claimed.scope.thread_id is None
    assert claimed.scope_kind is ScopeKind.CHAT


def test_expired_once_is_deleted_when_claimed(tmp_path):
    path = tmp_path / "profile-routing.db"
    store = ProfileRoutingStore(path, clock=lambda: 20.0)
    store.set_once(
        _scope(),
        profile_name="research",
        created_by_user_id="user-1",
        expires_at=20.0,
    )

    assert store.claim_once(_scope()) is None
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM profile_once").fetchone()[0] == 0


def test_two_connections_cannot_claim_same_once(tmp_path):
    path = tmp_path / "profile-routing.db"
    first = ProfileRoutingStore(path)
    second = ProfileRoutingStore(path)
    first.set_once(
        _scope("thread-7"),
        profile_name="research",
        created_by_user_id="user-1",
    )
    barrier = threading.Barrier(2)

    def claim(store: ProfileRoutingStore):
        barrier.wait()
        return store.claim_once(_scope("thread-7"))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(claim, (first, second)))

    assert sum(result is not None for result in results) == 1
    assert [result.profile_name for result in results if result is not None] == [
        "research"
    ]


def test_record_session_is_scoped_by_profile(tmp_path):
    store = ProfileRoutingStore(tmp_path / "profile-routing.db", clock=lambda: 10.0)
    scope = _scope("thread-7")
    store.record_session(scope, "coder", "coder-session")
    store.record_session(scope, "research", "research-session")

    assert store.get_session(scope, "coder") == "coder-session"
    assert store.get_session(scope, "research") == "research-session"
    assert store.get_session(_scope(), "coder") is None


def test_audit_never_accepts_message_text_field(tmp_path):
    store = ProfileRoutingStore(tmp_path / "profile-routing.db")
    parameters = inspect.signature(store.append_audit).parameters

    assert "message_text" not in parameters
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in parameters.items()
        if name != "self"
    )
    with pytest.raises(TypeError, match="message_text"):
        store.append_audit(
            actor_user_id="user-1",
            scope=_scope(),
            old_profile=None,
            new_profile="coder",
            scope_kind=ScopeKind.CHAT,
            source="command",
            result="allowed",
            reason_code=ReasonCode.ALLOWED,
            message_text="do not persist me",
        )


def test_append_audit_persists_only_the_scalar_contract(tmp_path):
    path = tmp_path / "profile-routing.db"
    store = ProfileRoutingStore(path, clock=lambda: 42.0)
    store.append_audit(
        actor_user_id="user-1",
        scope=_scope(),
        old_profile=None,
        new_profile="coder",
        scope_kind=ScopeKind.CHAT,
        source="command",
        result="allowed",
        reason_code=ReasonCode.ALLOWED,
    )

    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM profile_switch_audit").fetchone()

    assert dict(row) == {
        "id": 1,
        "created_at": 42.0,
        "actor_user_id": "user-1",
        "platform": "telegram",
        "account_id": "telegram:primary",
        "chat_id": "chat-1",
        "thread_id": None,
        "old_profile": None,
        "new_profile": "coder",
        "scope_kind": "chat",
        "source": "command",
        "result": "allowed",
        "reason_code": "allowed",
    }


def test_prune_removes_old_audit_and_expired_nonces(tmp_path):
    path = tmp_path / "profile-routing.db"
    store = ProfileRoutingStore(path, clock=lambda: 100.0)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO profile_switch_audit (
                created_at, actor_user_id, platform, account_id, chat_id,
                source, result, reason_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                10.0,
                "user-1",
                "telegram",
                "telegram:primary",
                "chat-1",
                "command",
                "allowed",
                "allowed",
            ),
        )
        conn.execute(
            """
            INSERT INTO profile_switch_audit (
                created_at, actor_user_id, platform, account_id, chat_id,
                source, result, reason_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                90.0,
                "user-1",
                "telegram",
                "telegram:primary",
                "chat-1",
                "command",
                "allowed",
                "allowed",
            ),
        )
        conn.execute(
            """
            INSERT INTO profile_picker_nonces (
                nonce_hash, platform, account_id, actor_user_id, chat_id,
                message_id, action, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "expired",
                "telegram",
                "telegram:primary",
                "user-1",
                "chat-1",
                "message-1",
                "set",
                99.0,
            ),
        )
        conn.execute(
            """
            INSERT INTO profile_picker_nonces (
                nonce_hash, platform, account_id, actor_user_id, chat_id,
                message_id, action, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "live",
                "telegram",
                "telegram:primary",
                "user-1",
                "chat-1",
                "message-2",
                "set",
                101.0,
            ),
        )

    audit_removed, nonces_removed = store.prune(audit_before=50.0)

    assert (audit_removed, nonces_removed) == (1, 1)
    with sqlite3.connect(path) as conn:
        assert conn.execute(
            "SELECT created_at FROM profile_switch_audit"
        ).fetchall() == [(90.0,)]
        assert conn.execute(
            "SELECT nonce_hash FROM profile_picker_nonces"
        ).fetchall() == [("live",)]
