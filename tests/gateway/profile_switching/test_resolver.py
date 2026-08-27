from __future__ import annotations

import pytest

from gateway.config import ProfileSwitchRule, ProfileSwitchingConfig
from gateway.profile_switching.models import (
    ReasonCode,
    ResolutionSource,
    ScopeKey,
    ScopeKind,
)
from gateway.profile_switching.policy import ProfilePolicy
from gateway.profile_switching.resolver import ProfileResolver
from gateway.profile_switching.store import (
    ProfileRoutingStore,
    ProfileRoutingStoreUnavailable,
)


def _scope() -> ScopeKey:
    return ScopeKey("telegram", "telegram:primary", "chat-1", "thread-7")


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


def _set_binding(
    store: ProfileRoutingStore,
    scope_kind: ScopeKind,
    profile_name: str,
) -> None:
    store.set_binding(
        _scope(),
        scope_kind,
        profile_name=profile_name,
        created_by_user_id="user-1",
    )


def test_once_wins_over_thread_chat_and_static(tmp_path):
    store = ProfileRoutingStore(tmp_path / "routing.db")
    store.set_once(_scope(), profile_name="research", created_by_user_id="user-1")
    _set_binding(store, ScopeKind.THREAD, "coder")
    _set_binding(store, ScopeKind.CHAT, "writer")
    resolver = ProfileResolver(store, _policy("research", "coder", "writer", "static"))

    resolution = resolver.resolve(
        _scope(),
        actor_user_id="user-1",
        consume_once=True,
        static_profile="static",
    )

    assert resolution.profile_name == "research"
    assert resolution.source is ResolutionSource.ONCE
    assert resolution.binding is not None


def test_thread_wins_over_chat_and_static(tmp_path):
    store = ProfileRoutingStore(tmp_path / "routing.db")
    _set_binding(store, ScopeKind.THREAD, "coder")
    _set_binding(store, ScopeKind.CHAT, "writer")
    resolver = ProfileResolver(store, _policy("coder", "writer", "static"))

    resolution = resolver.resolve(
        _scope(),
        actor_user_id="user-1",
        consume_once=True,
        static_profile="static",
    )

    assert resolution.profile_name == "coder"
    assert resolution.source is ResolutionSource.THREAD
    assert resolution.binding is not None


def test_chat_wins_over_static(tmp_path):
    store = ProfileRoutingStore(tmp_path / "routing.db")
    _set_binding(store, ScopeKind.CHAT, "writer")
    resolver = ProfileResolver(store, _policy("writer", "static"))

    resolution = resolver.resolve(
        _scope(),
        actor_user_id="user-1",
        consume_once=True,
        static_profile="static",
    )

    assert resolution.profile_name == "writer"
    assert resolution.source is ResolutionSource.CHAT
    assert resolution.binding is not None


def test_static_used_when_no_dynamic_binding(tmp_path):
    resolver = ProfileResolver(
        ProfileRoutingStore(tmp_path / "routing.db"), _policy("static")
    )

    resolution = resolver.resolve(
        _scope(),
        actor_user_id="user-1",
        consume_once=True,
        static_profile="static",
    )

    assert resolution.profile_name == "static"
    assert resolution.source is ResolutionSource.STATIC
    assert resolution.binding is None


def test_default_returned_when_nothing_matches(tmp_path):
    resolver = ProfileResolver(
        ProfileRoutingStore(tmp_path / "routing.db"), _policy("static")
    )

    resolution = resolver.resolve(
        _scope(),
        actor_user_id="user-1",
        consume_once=True,
        static_profile=None,
    )

    assert resolution.profile_name is None
    assert resolution.source is ResolutionSource.DEFAULT
    assert resolution.reason is ReasonCode.NO_MATCH
    assert resolution.binding is None


def test_command_resolution_does_not_consume_once(tmp_path):
    store = ProfileRoutingStore(tmp_path / "routing.db")
    store.set_once(_scope(), profile_name="research", created_by_user_id="user-1")
    _set_binding(store, ScopeKind.THREAD, "coder")
    resolver = ProfileResolver(store, _policy("research", "coder"))

    resolution = resolver.resolve(
        _scope(),
        actor_user_id="user-1",
        consume_once=False,
        static_profile=None,
    )

    assert resolution.profile_name == "coder"
    assert resolution.source is ResolutionSource.THREAD
    claimed = store.claim_once(_scope())
    assert claimed is not None and claimed.profile_name == "research"


def test_denied_once_is_consumed_then_thread_is_considered(tmp_path):
    store = ProfileRoutingStore(tmp_path / "routing.db")
    store.set_once(_scope(), profile_name="private", created_by_user_id="user-1")
    _set_binding(store, ScopeKind.THREAD, "coder")
    resolver = ProfileResolver(store, _policy("coder"))

    resolution = resolver.resolve(
        _scope(),
        actor_user_id="user-1",
        consume_once=True,
        static_profile=None,
    )

    assert resolution.profile_name == "coder"
    assert resolution.source is ResolutionSource.THREAD
    assert store.claim_once(_scope()) is None


def test_denied_thread_falls_back_to_valid_chat(tmp_path):
    store = ProfileRoutingStore(tmp_path / "routing.db")
    _set_binding(store, ScopeKind.THREAD, "private")
    _set_binding(store, ScopeKind.CHAT, "writer")
    resolver = ProfileResolver(store, _policy("writer"))

    resolution = resolver.resolve(
        _scope(),
        actor_user_id="user-1",
        consume_once=True,
        static_profile=None,
    )

    assert resolution.profile_name == "writer"
    assert resolution.source is ResolutionSource.CHAT


def test_db_failure_falls_back_to_static_not_sensitive_candidate():
    class UnavailableStore:
        def claim_once(self, scope):
            raise ProfileRoutingStoreUnavailable("database unavailable")

        def get_binding(self, scope, scope_kind):
            raise AssertionError("dynamic reads must stop after database failure")

    resolver = ProfileResolver(UnavailableStore(), _policy("static"))

    resolution = resolver.resolve(
        _scope(),
        actor_user_id="user-1",
        consume_once=True,
        static_profile="static",
    )

    assert resolution.profile_name == "static"
    assert resolution.source is ResolutionSource.STATIC


def test_db_failure_still_evaluates_static_fail_closed():
    class UnavailableStore:
        def get_binding(self, scope, scope_kind):
            raise ProfileRoutingStoreUnavailable("database unavailable")

    resolver = ProfileResolver(UnavailableStore(), _policy("coder"))

    resolution = resolver.resolve(
        _scope(),
        actor_user_id="user-1",
        consume_once=False,
        static_profile="private",
    )

    assert resolution.profile_name is None
    assert resolution.source is ResolutionSource.DEFAULT
    assert resolution.reason is ReasonCode.NO_MATCH


def test_policy_errors_are_not_treated_as_database_fallback(tmp_path):
    class BrokenPolicy:
        def evaluate(self, profile_name, scope, actor_user_id):
            raise RuntimeError("policy bug")

    resolver = ProfileResolver(
        ProfileRoutingStore(tmp_path / "routing.db"), BrokenPolicy()
    )

    with pytest.raises(RuntimeError, match="policy bug"):
        resolver.resolve(
            _scope(),
            actor_user_id="user-1",
            consume_once=False,
            static_profile="static",
        )


def test_non_store_unavailable_read_errors_propagate():
    class BrokenStore:
        def get_binding(self, scope, scope_kind):
            raise RuntimeError("store programming bug")

    resolver = ProfileResolver(BrokenStore(), _policy("static"))

    with pytest.raises(RuntimeError, match="store programming bug"):
        resolver.resolve(
            _scope(),
            actor_user_id="user-1",
            consume_once=False,
            static_profile="static",
        )
