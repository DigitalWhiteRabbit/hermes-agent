from __future__ import annotations

import inspect
import json
import sqlite3

import pytest

from gateway.config import ProfileSwitchRule, ProfileSwitchingConfig
from gateway.profile_switching.models import ReasonCode, ScopeKey, ScopeKind
from gateway.profile_switching.policy import ProfilePolicy
from gateway.profile_switching.service import (
    InvalidProfileScope,
    ProfileSwitchBusy,
    ProfileSwitchDenied,
    ProfileSwitchingService,
)
from gateway.profile_switching.store import (
    ProfileRoutingStore,
    ProfileRoutingStoreUnavailable,
)


def _scope(thread_id: str | None = "thread-7") -> ScopeKey:
    return ScopeKey("telegram", "telegram:primary", "chat-1", thread_id)


def _policy(*profiles: str) -> ProfilePolicy:
    return ProfilePolicy(
        ProfileSwitchingConfig(
            rules=tuple(
                ProfileSwitchRule(
                    profile,
                    users=("user-1",),
                    chats=("*",),
                )
                for profile in profiles
            )
        ),
        served_profiles=profiles,
        existing_profiles=profiles,
    )


def _service(tmp_path, *profiles: str):
    store = ProfileRoutingStore(tmp_path / "routing.db")
    return ProfileSwitchingService(store, _policy(*profiles)), store


def _audit_rows(store: ProfileRoutingStore) -> list[sqlite3.Row]:
    with sqlite3.connect(store.path) as conn:
        conn.row_factory = sqlite3.Row
        return list(conn.execute("SELECT * FROM profile_switch_audit ORDER BY id"))


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


def test_set_profile_validates_before_write(tmp_path):
    service, store = _service(tmp_path, "coder")

    with pytest.raises(ValueError, match="Invalid profile name"):
        service.set_profile(
            scope=_scope(),
            scope_kind=ScopeKind.THREAD,
            actor_user_id="user-1",
            profile_name="../coder",
            active_turn=False,
        )

    assert store.get_binding(_scope(), ScopeKind.THREAD) is None


def test_set_profile_normalizes_before_authorization_and_write(tmp_path):
    service, store = _service(tmp_path, "coder")

    binding = service.set_profile(
        scope=_scope(),
        scope_kind=ScopeKind.THREAD,
        actor_user_id="user-1",
        profile_name=" Coder ",
        active_turn=False,
    )

    assert binding.profile_name == "coder"
    stored = store.get_binding(_scope(), ScopeKind.THREAD)
    assert stored is not None and stored.profile_name == "coder"


def test_set_profile_rejects_policy_denial_before_write(tmp_path):
    service, store = _service(tmp_path, "coder")

    with pytest.raises(ProfileSwitchDenied) as exc_info:
        service.set_profile(
            scope=_scope(),
            scope_kind=ScopeKind.THREAD,
            actor_user_id="user-1",
            profile_name="private",
            active_turn=False,
        )

    assert exc_info.value.reason is ReasonCode.PROFILE_UNKNOWN
    assert store.get_binding(_scope(), ScopeKind.THREAD) is None
    audit = _audit_rows(store)
    assert len(audit) == 1
    assert audit[0]["result"] == "denied"
    assert audit[0]["reason_code"] == "profile_unknown"


def test_set_profile_rejects_active_turn(tmp_path):
    service, store = _service(tmp_path, "coder")

    with pytest.raises(ProfileSwitchBusy):
        service.set_profile(
            scope=_scope(),
            scope_kind=ScopeKind.THREAD,
            actor_user_id="user-1",
            profile_name="coder",
            active_turn=True,
        )

    assert store.get_binding(_scope(), ScopeKind.THREAD) is None
    audit = _audit_rows(store)
    assert len(audit) == 1
    assert audit[0]["result"] == "denied"
    assert audit[0]["reason_code"] == "active_turn"


def test_clear_rejects_active_turn(tmp_path):
    service, store = _service(tmp_path, "coder")
    store.set_binding(
        _scope(),
        ScopeKind.THREAD,
        profile_name="coder",
        created_by_user_id="user-1",
    )

    with pytest.raises(ProfileSwitchBusy):
        service.clear(
            scope=_scope(),
            scope_kind=ScopeKind.THREAD,
            actor_user_id="user-1",
            active_turn=True,
        )

    binding = store.get_binding(_scope(), ScopeKind.THREAD)
    assert binding is not None and binding.profile_name == "coder"
    audit = _audit_rows(store)
    assert len(audit) == 1
    assert audit[0]["result"] == "denied"
    assert audit[0]["reason_code"] == "active_turn"


def test_set_thread_requires_thread_id(tmp_path):
    service, store = _service(tmp_path, "coder")

    with pytest.raises(InvalidProfileScope):
        service.set_profile(
            scope=_scope(None),
            scope_kind=ScopeKind.THREAD,
            actor_user_id="user-1",
            profile_name="coder",
            active_turn=False,
        )

    assert store.get_binding(_scope(None), ScopeKind.THREAD) is None
    audit = _audit_rows(store)
    assert len(audit) == 1
    assert audit[0]["result"] == "denied"
    assert audit[0]["reason_code"] == "thread_denied"


def test_clear_only_removes_requested_dynamic_scope(tmp_path):
    service, store = _service(tmp_path, "coder", "writer")
    store.set_binding(
        _scope(),
        ScopeKind.CHAT,
        profile_name="writer",
        created_by_user_id="user-1",
    )
    store.set_binding(
        _scope(),
        ScopeKind.THREAD,
        profile_name="coder",
        created_by_user_id="user-1",
    )

    assert service.clear(
        scope=_scope(),
        scope_kind=ScopeKind.THREAD,
        actor_user_id="user-1",
        active_turn=False,
    )

    assert store.get_binding(_scope(), ScopeKind.THREAD) is None
    chat = store.get_binding(_scope(), ScopeKind.CHAT)
    assert chat is not None and chat.profile_name == "writer"


def test_set_once_does_not_replace_permanent_binding(tmp_path):
    service, store = _service(tmp_path, "coder", "research")
    store.set_binding(
        _scope(),
        ScopeKind.THREAD,
        profile_name="coder",
        created_by_user_id="user-1",
    )

    once = service.set_once(
        scope=_scope(),
        actor_user_id="user-1",
        profile_name=" Research ",
        active_turn=False,
    )

    permanent = store.get_binding(_scope(), ScopeKind.THREAD)
    assert once.profile_name == "research"
    assert permanent is not None and permanent.profile_name == "coder"
    claimed = store.claim_once(_scope())
    assert claimed is not None and claimed.profile_name == "research"


def test_set_once_audit_captures_replaced_one_shot(tmp_path):
    service, store = _service(tmp_path, "coder", "research")
    service.set_once(
        scope=_scope(),
        actor_user_id="user-1",
        profile_name="research",
        active_turn=False,
    )

    service.set_once(
        scope=_scope(),
        actor_user_id="user-1",
        profile_name="coder",
        active_turn=False,
    )

    audit = _audit_rows(store)
    assert audit[1]["old_profile"] == "research"
    assert audit[1]["new_profile"] == "coder"


def test_set_profile_rolls_back_when_allowed_audit_insert_fails(tmp_path):
    service, store = _service(tmp_path, "coder", "research")
    store.set_binding(
        _scope(),
        ScopeKind.THREAD,
        profile_name="research",
        created_by_user_id="user-1",
    )
    _reject_audit_inserts(store)

    with pytest.raises(ProfileRoutingStoreUnavailable):
        service.set_profile(
            scope=_scope(),
            scope_kind=ScopeKind.THREAD,
            actor_user_id="user-1",
            profile_name="coder",
            active_turn=False,
        )

    binding = store.get_binding(_scope(), ScopeKind.THREAD)
    assert binding is not None and binding.profile_name == "research"
    assert binding.version == 1


def test_set_once_rolls_back_when_allowed_audit_insert_fails(tmp_path):
    service, store = _service(tmp_path, "coder", "research")
    store.set_once(
        _scope(),
        profile_name="research",
        created_by_user_id="user-1",
    )
    _reject_audit_inserts(store)

    with pytest.raises(ProfileRoutingStoreUnavailable):
        service.set_once(
            scope=_scope(),
            actor_user_id="user-1",
            profile_name="coder",
            active_turn=False,
        )

    once = store.claim_once(_scope())
    assert once is not None and once.profile_name == "research"


def test_clear_rolls_back_when_allowed_audit_insert_fails(tmp_path):
    service, store = _service(tmp_path, "coder")
    store.set_binding(
        _scope(),
        ScopeKind.THREAD,
        profile_name="coder",
        created_by_user_id="user-1",
    )
    _reject_audit_inserts(store)

    with pytest.raises(ProfileRoutingStoreUnavailable):
        service.clear(
            scope=_scope(),
            scope_kind=ScopeKind.THREAD,
            actor_user_id="user-1",
            active_turn=False,
        )

    binding = store.get_binding(_scope(), ScopeKind.THREAD)
    assert binding is not None and binding.profile_name == "coder"


def test_clear_reauthorizes_binding_replaced_after_initial_read(tmp_path):
    path = tmp_path / "routing.db"
    replacement_store = ProfileRoutingStore(path)

    class RacingStore(ProfileRoutingStore):
        raced = False

        def clear_binding_with_audit(self, *args, **kwargs):
            if not self.raced:
                self.raced = True
                replacement_store.set_binding(
                    _scope(),
                    ScopeKind.THREAD,
                    profile_name="research",
                    created_by_user_id="user-2",
                )
            return super().clear_binding_with_audit(*args, **kwargs)

    class RecordingPolicy:
        def __init__(self):
            self.profiles = []
            self.delegate = _policy("coder", "research")

        def evaluate(self, profile_name, scope, actor_user_id):
            self.profiles.append(profile_name)
            return self.delegate.evaluate(profile_name, scope, actor_user_id)

    store = RacingStore(path)
    store.set_binding(
        _scope(),
        ScopeKind.THREAD,
        profile_name="coder",
        created_by_user_id="user-1",
    )
    policy = RecordingPolicy()
    service = ProfileSwitchingService(store, policy)

    assert service.clear(
        scope=_scope(),
        scope_kind=ScopeKind.THREAD,
        actor_user_id="user-1",
        active_turn=False,
    )

    assert policy.profiles == ["coder", "research"]
    assert store.get_binding(_scope(), ScopeKind.THREAD) is None
    audit = _audit_rows(store)
    assert len(audit) == 1
    assert audit[0]["old_profile"] == "research"


def test_each_mutation_appends_redacted_audit(tmp_path):
    service, store = _service(tmp_path, "coder", "research")

    service.set_profile(
        scope=_scope(),
        scope_kind=ScopeKind.THREAD,
        actor_user_id="user-1",
        profile_name="coder",
        active_turn=False,
    )
    service.set_once(
        scope=_scope(),
        actor_user_id="user-1",
        profile_name="research",
        active_turn=False,
    )
    service.clear(
        scope=_scope(),
        scope_kind=ScopeKind.THREAD,
        actor_user_id="user-1",
        active_turn=False,
    )

    audit = _audit_rows(store)
    assert [row["source"] for row in audit] == ["command", "once", "clear"]
    assert [row["result"] for row in audit] == ["allowed", "allowed", "allowed"]
    assert all(row["reason_code"] == "allowed" for row in audit)
    forbidden_fields = {
        "message",
        "message_text",
        "prompt",
        "tool_data",
        "tool_output",
    }
    for method_name in ("set_profile", "set_once", "clear"):
        parameters = inspect.signature(getattr(service, method_name)).parameters
        assert forbidden_fields.isdisjoint(parameters)

    with pytest.raises(TypeError, match="message_text"):
        service.set_once(
            scope=_scope(),
            actor_user_id="user-1",
            profile_name="research",
            active_turn=False,
            message_text="do not accept this",
        )


def test_resolution_audit_is_bounded_and_redacts_identifiers(tmp_path):
    store = ProfileRoutingStore(tmp_path / "routing.db")
    service = ProfileSwitchingService(
        store,
        _policy("static"),
        resolution_audit_limit=2,
    )
    for profile in ("secret-one", "secret-two", "secret-three"):
        service.resolve(
            scope=_scope(),
            actor_user_id="sensitive-user-id",
            consume_once=False,
            static_profile=profile,
        )

    audit = service.resolution_audit
    rendered = json.dumps(audit)
    assert len(audit) == 2
    assert service.resolution_metrics[ReasonCode.PROFILE_UNKNOWN.value] == 3
    assert "sensitive-user-id" not in rendered
    assert "telegram:primary" not in rendered
    assert "chat-1" not in rendered
    assert "secret-" not in rendered


def test_note_session_keeps_one_session_per_profile_and_scope(tmp_path):
    service, store = _service(tmp_path, "coder", "research")
    other_scope = ScopeKey("telegram", "telegram:primary", "chat-2", None)

    service.note_session(scope=_scope(), profile_name="Coder", session_id="coder-old")
    service.note_session(
        scope=_scope(), profile_name="coder", session_id="coder-current"
    )
    service.note_session(
        scope=_scope(), profile_name="research", session_id="research-current"
    )
    service.note_session(
        scope=other_scope, profile_name="coder", session_id="other-current"
    )

    assert store.get_session(_scope(), "coder") == "coder-current"
    assert store.get_session(_scope(), "research") == "research-current"
    assert store.get_session(other_scope, "coder") == "other-current"
