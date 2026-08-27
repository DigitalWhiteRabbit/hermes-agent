from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import (
    GatewayConfig,
    Platform,
    PlatformConfig,
    ProfileSwitchRule,
    ProfileSwitchingConfig,
)
from gateway.platforms.base import BasePlatformAdapter, MessageEvent
from gateway.profile_routing import ProfileRoute
from gateway.profile_switching.models import ScopeKey, ScopeKind
from gateway.profile_switching.policy import ProfilePolicy
from gateway.profile_switching.service import ProfileSwitchingService
from gateway.profile_switching.store import (
    ProfileRoutingStore,
    ProfileRoutingStoreUnavailable,
)
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource
from hermes_constants import get_hermes_home


_PROFILES = ("static", "coder", "research", "writer", "origin")


class _StubAdapter(BasePlatformAdapter):
    pass


_StubAdapter.__abstractmethods__ = frozenset()  # type: ignore[attr-defined]


def _profile_switching_config(*, enabled: bool = True) -> ProfileSwitchingConfig:
    return ProfileSwitchingConfig(
        enabled=enabled,
        rules=tuple(
            ProfileSwitchRule(
                profile,
                users=("user-1",),
                chats=("*",),
            )
            for profile in ("default", *_PROFILES)
        ),
    )


def _gateway_config(*, enabled: bool = True) -> GatewayConfig:
    return GatewayConfig(
        multiplex_profiles=True,
        multiplex_profile_allowlist=list(_PROFILES),
        profile_routes=[
            ProfileRoute(
                name="static-chat-route",
                platform="telegram",
                profile="static",
                chat_id="chat-1",
            )
        ],
        profile_switching=_profile_switching_config(enabled=enabled),
    )


def _create_profile_homes() -> dict[str, Path]:
    root = get_hermes_home()
    homes = {"default": root}
    for profile in _PROFILES:
        home = root / "profiles" / profile
        home.mkdir(parents=True, exist_ok=True)
        homes[profile] = home
    return homes


def _runner_with_store(tmp_path) -> tuple[GatewayRunner, ProfileRoutingStore]:
    _create_profile_homes()
    config = _gateway_config()
    store = ProfileRoutingStore(tmp_path / "profile-routing.db")
    policy = ProfilePolicy(
        config.profile_switching,
        served_profiles={"default", *_PROFILES},
        existing_profiles={"default", *_PROFILES},
    )
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner._profile_switching_service = ProfileSwitchingService(store, policy)
    runner.adapters = {}
    runner._profile_adapters = {}
    return runner, store


def _adapter(runner: GatewayRunner) -> _StubAdapter:
    adapter = _StubAdapter.__new__(_StubAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.gateway_runner = runner
    return adapter


def _event(
    runner: GatewayRunner,
    text: str = "hello",
    *,
    thread_id: str | None = None,
    internal: bool = False,
) -> MessageEvent:
    source = _adapter(runner).build_source(
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
        thread_id=thread_id,
    )
    assert source.profile == "static"
    return MessageEvent(text=text, source=source, internal=internal)


def _scope(event: MessageEvent) -> ScopeKey:
    return ScopeKey.from_source(event.source, account_id="telegram:primary")


async def _dispatch_through_primary_wrapper(
    runner: GatewayRunner,
    event: MessageEvent,
) -> dict[str, object]:
    captured: dict[str, object] = {}

    async def capture(resolved_event: MessageEvent):
        captured["profile"] = resolved_event.source.profile
        captured["session_key"] = runner._session_key_for_source(resolved_event.source)
        captured["runtime_home"] = get_hermes_home()
        captured["authorization_home"] = getattr(
            resolved_event.source,
            "_authorization_profile_home",
            None,
        )
        return captured["session_key"]

    runner._handle_message = capture
    result = await runner._make_default_profile_message_handler()(event)
    captured["result"] = result
    return captured


def _set_binding(
    store: ProfileRoutingStore,
    scope: ScopeKey,
    scope_kind: ScopeKind,
    profile: str,
) -> None:
    store.set_binding(
        scope,
        scope_kind,
        profile_name=profile,
        created_by_user_id="user-1",
    )


def test_enabled_runner_initializes_real_service_and_default_policy():
    homes = _create_profile_homes()
    config = _gateway_config()

    runner = GatewayRunner(config)

    assert runner._profile_switching_service is not None
    assert (homes["default"] / "state" / "profile-routing.db").exists()
    resolution = runner._profile_switching_service.resolve(
        scope=ScopeKey("telegram", "telegram:primary", "chat-1", None),
        actor_user_id="user-1",
        consume_once=True,
        static_profile="default",
    )
    assert resolution.profile_name == "default"


@pytest.mark.asyncio
async def test_chat_binding_overrides_static_route_before_session_key(tmp_path):
    runner, store = _runner_with_store(tmp_path)
    event = _event(runner)
    _set_binding(store, _scope(event), ScopeKind.CHAT, "coder")

    captured = await _dispatch_through_primary_wrapper(runner, event)

    assert captured["profile"] == "coder"
    assert captured["session_key"] == "agent:coder:telegram:dm:chat-1"


@pytest.mark.asyncio
async def test_thread_binding_overrides_chat_binding(tmp_path):
    runner, store = _runner_with_store(tmp_path)
    event = _event(runner, thread_id="thread-7")
    _set_binding(store, _scope(event), ScopeKind.CHAT, "writer")
    _set_binding(store, _scope(event), ScopeKind.THREAD, "coder")

    captured = await _dispatch_through_primary_wrapper(runner, event)

    assert captured["profile"] == "coder"
    assert captured["session_key"] == ("agent:coder:telegram:dm:chat-1:thread-7")


@pytest.mark.asyncio
async def test_slash_command_uses_permanent_binding_but_does_not_consume_once(
    tmp_path,
):
    runner, store = _runner_with_store(tmp_path)
    command = _event(runner, "/help")
    _set_binding(store, _scope(command), ScopeKind.CHAT, "coder")
    store.set_once(
        _scope(command),
        profile_name="research",
        created_by_user_id="user-1",
    )

    command_result = await _dispatch_through_primary_wrapper(runner, command)
    plain_result = await _dispatch_through_primary_wrapper(runner, _event(runner))

    assert command_result["profile"] == "coder"
    assert command_result["session_key"] == "agent:coder:telegram:dm:chat-1"
    assert plain_result["profile"] == "research"


@pytest.mark.asyncio
async def test_plain_message_consumes_once_before_session_key(tmp_path):
    runner, store = _runner_with_store(tmp_path)
    first = _event(runner)
    _set_binding(store, _scope(first), ScopeKind.CHAT, "coder")
    store.set_once(
        _scope(first),
        profile_name="research",
        created_by_user_id="user-1",
    )

    first_result = await _dispatch_through_primary_wrapper(runner, first)
    second_result = await _dispatch_through_primary_wrapper(runner, _event(runner))

    assert first_result["session_key"] == "agent:research:telegram:dm:chat-1"
    assert second_result["session_key"] == "agent:coder:telegram:dm:chat-1"


@pytest.mark.asyncio
async def test_primary_adapter_resolves_before_busy_session_key(tmp_path):
    runner, store = _runner_with_store(tmp_path)
    event = _event(runner)
    store.set_once(
        _scope(event),
        profile_name="research",
        created_by_user_id="user-1",
    )
    adapter = _StubAdapter(
        PlatformConfig(enabled=True),
        Platform.TELEGRAM,
    )
    adapter.gateway_runner = runner
    adapter.set_message_handler(runner._make_default_profile_message_handler())
    captured_keys: list[str] = []
    adapter._start_session_processing = lambda _event, session_key: (
        captured_keys.append(session_key)
    )

    await adapter.handle_message(event)

    assert captured_keys == ["agent:research:telegram:dm:chat-1"]


@pytest.mark.asyncio
async def test_secondary_adapter_skips_primary_routing_and_preserves_once(
    tmp_path,
):
    runner, store = _runner_with_store(tmp_path)
    homes = _create_profile_homes()
    adapter = _StubAdapter(
        PlatformConfig(enabled=True),
        Platform.TELEGRAM,
    )
    adapter.gateway_runner = runner
    adapter.set_owner_profile("origin")
    event = MessageEvent(
        text="hello",
        source=adapter.build_source(
            chat_id="chat-1",
            chat_type="dm",
            user_id="user-1",
        ),
    )
    store.set_once(
        _scope(event),
        profile_name="research",
        created_by_user_id="user-1",
    )
    handler = runner._make_profile_message_handler("origin")
    adapter.set_message_handler(handler)
    captured_keys: list[str] = []
    adapter._start_session_processing = lambda _event, session_key: (
        captured_keys.append(session_key)
    )
    captured: dict[str, object] = {}

    async def capture(resolved_event: MessageEvent):
        captured["runtime_home"] = get_hermes_home()
        captured["authorization_home"] = getattr(
            resolved_event.source,
            "_authorization_profile_home",
            None,
        )

    runner._handle_message = capture

    await adapter.handle_message(event)
    await handler(event)

    assert captured_keys == ["agent:origin:telegram:dm:chat-1"]
    assert event.source.profile == "origin"
    assert event.source.transport_owner_profile == "origin"
    assert not hasattr(event.source, "_dynamic_profile_resolved")
    assert captured["runtime_home"] == homes["origin"]
    assert captured["authorization_home"] == homes["origin"]

    primary = _event(runner)
    await runner._resolve_dynamic_profile_for_event(primary)
    assert primary.source.profile == "research"


@pytest.mark.asyncio
async def test_primary_topic_recovery_preserves_live_transport_owner(tmp_path):
    runner, store = _runner_with_store(tmp_path)
    adapter = _StubAdapter(
        PlatformConfig(enabled=True),
        Platform.TELEGRAM,
    )
    adapter.gateway_runner = runner
    adapter.set_topic_recovery_fn(lambda _source: "topic-7")
    adapter.set_message_handler(AsyncMock())
    runner.adapters = {Platform.TELEGRAM: adapter}
    event = MessageEvent(
        text="hello",
        source=adapter.build_source(
            chat_id="chat-1",
            chat_type="dm",
            user_id="user-1",
        ),
    )
    _set_binding(store, _scope(event), ScopeKind.CHAT, "coder")
    captured_keys: list[str] = []
    adapter._start_session_processing = lambda _event, session_key: (
        captured_keys.append(session_key)
    )

    await adapter.handle_message(event)

    assert captured_keys == ["agent:coder:telegram:dm:chat-1:topic-7"]
    assert event.source.profile == "coder"
    assert event.source.transport_owner_profile == "default"
    assert event.source._transport_adapter_ref() is adapter
    assert runner._adapter_for_source(event.source) is adapter


@pytest.mark.asyncio
async def test_disabled_switching_preserves_secondary_runtime_scope(tmp_path):
    runner, _store = _runner_with_store(tmp_path)
    homes = _create_profile_homes()
    runner._profile_switching_service = None
    event = _event(runner)
    captured: dict[str, object] = {}

    async def capture(_event: MessageEvent):
        captured["runtime_home"] = get_hermes_home()

    runner._handle_message = capture

    await runner._make_profile_message_handler("origin")(event)

    assert captured["runtime_home"] == homes["origin"]
    assert event.source.profile == "static"


@pytest.mark.asyncio
async def test_dynamic_resolution_keeps_authorization_home_on_primary_profile(
    tmp_path,
):
    runner, store = _runner_with_store(tmp_path)
    homes = _create_profile_homes()
    event = _event(runner)
    _set_binding(store, _scope(event), ScopeKind.CHAT, "coder")

    captured = await _dispatch_through_primary_wrapper(runner, event)

    assert captured["runtime_home"] == homes["coder"]
    assert captured["authorization_home"] == homes["default"]


@pytest.mark.asyncio
async def test_direct_internal_source_preserves_originating_profile(tmp_path):
    runner, store = _runner_with_store(tmp_path)
    internal_event = _event(runner, internal=True)
    internal_event.source.profile = "origin"
    _set_binding(store, _scope(internal_event), ScopeKind.CHAT, "coder")
    store.set_once(
        _scope(internal_event),
        profile_name="research",
        created_by_user_id="user-1",
    )

    await runner._resolve_dynamic_profile_for_event(internal_event)

    assert internal_event.source.profile == "origin"
    assert internal_event.source._profile_transport_account_id == "telegram:primary"
    assert internal_event.source._dynamic_profile_resolved is True

    plain = _event(runner)
    await runner._resolve_dynamic_profile_for_event(plain)
    assert plain.source.profile == "research"


@pytest.mark.asyncio
async def test_direct_handle_message_uses_dynamic_resolution_fallback(tmp_path):
    runner, store = _runner_with_store(tmp_path)
    event = _event(runner)
    event.source.profile_route_rejected = True
    store.set_once(
        _scope(event),
        profile_name="research",
        created_by_user_id="user-1",
    )

    result = await GatewayRunner._handle_message(runner, event)

    assert result is None
    assert event.source.profile == "research"


@pytest.mark.asyncio
async def test_direct_unstamped_source_evaluates_allowed_static_candidate(tmp_path):
    runner, _store = _runner_with_store(tmp_path)
    event = MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            chat_type="dm",
            user_id="user-1",
        ),
    )
    runner._startup_restore_in_progress = True
    runner._queue_startup_restore_event = lambda _event: None

    assert await GatewayRunner._handle_message(runner, event) is None
    assert event.source.profile == "static"
    assert event.source._dynamic_profile_resolution_source == "static"


def _install_denied_static_service(
    runner: GatewayRunner,
    tmp_path: Path,
    *,
    denial: str,
    db_unavailable: bool,
) -> None:
    static_users = ("other-user",) if denial == "unauthorized" else ("user-1",)
    config = ProfileSwitchingConfig(
        enabled=True,
        hidden=("static",) if denial == "hidden" else (),
        rules=(
            ProfileSwitchRule("default", users=("user-1",), chats=("*",)),
            ProfileSwitchRule("static", users=static_users, chats=("*",)),
        ),
    )

    if db_unavailable:

        class _UnavailableStore:
            def claim_once(self, _scope):
                raise ProfileRoutingStoreUnavailable("database unavailable")

            def get_binding(self, _scope, _scope_kind):
                raise ProfileRoutingStoreUnavailable("database unavailable")

        routing_store = _UnavailableStore()
    else:
        routing_store = ProfileRoutingStore(tmp_path / f"{denial}.db")

    policy = ProfilePolicy(
        config,
        served_profiles=(
            {"default"} if denial == "unserved" else {"default", "static"}
        ),
        existing_profiles={"default", "static"},
    )
    runner._profile_switching_service = ProfileSwitchingService(
        routing_store,
        policy,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("denial", ["unserved", "hidden", "unauthorized"])
@pytest.mark.parametrize("db_unavailable", [False, True])
async def test_denied_static_is_not_reapplied_after_dynamic_resolution(
    tmp_path,
    denial,
    db_unavailable,
):
    runner, _store = _runner_with_store(tmp_path)
    _install_denied_static_service(
        runner,
        tmp_path,
        denial=denial,
        db_unavailable=db_unavailable,
    )
    adapter = _StubAdapter(
        PlatformConfig(enabled=True),
        Platform.TELEGRAM,
    )
    adapter.gateway_runner = runner
    adapter.set_session_store(
        SimpleNamespace(
            _resolve_profile_for_key=lambda source: source.profile or "default"
        )
    )
    event = MessageEvent(
        text="hello",
        source=adapter.build_source(
            chat_id="chat-1",
            chat_type="dm",
            user_id="user-1",
        ),
    )
    handler = runner._make_default_profile_message_handler()
    adapter.set_message_handler(handler)
    captured_keys: list[str] = []
    adapter._start_session_processing = lambda _event, session_key: (
        captured_keys.append(session_key)
    )

    await adapter.handle_message(event)
    assert captured_keys == ["agent:main:telegram:dm:chat-1"]
    assert event.source.profile is None

    captured = await _dispatch_through_primary_wrapper(runner, event)
    assert captured["profile"] is None
    assert captured["session_key"] == "agent:main:telegram:dm:chat-1"

    direct = _event(runner)
    runner._startup_restore_in_progress = True
    runner._queue_startup_restore_event = lambda _event: None
    assert await GatewayRunner._handle_message(runner, direct) is None
    assert direct.source.profile is None


def test_transport_account_identity_is_stable_across_runtime_profiles(tmp_path):
    runner, _store = _runner_with_store(tmp_path)
    first = _event(runner).source
    second = _event(runner).source
    first.profile = "coder"
    second.profile = "research"
    first.user_name = "different-user-name"

    assert runner._profile_account_id_for_source(first) == "telegram:primary"
    assert runner._profile_account_id_for_source(second) == "telegram:primary"


@pytest.mark.asyncio
async def test_session_observation_failure_is_nonfatal(tmp_path, caplog):
    runner, _store = _runner_with_store(tmp_path)

    class _BrokenObservationService:
        def note_session(self, **_kwargs):
            raise RuntimeError("observation unavailable")

    runner._profile_switching_service = _BrokenObservationService()
    source = _event(runner).source
    source._profile_transport_account_id = "telegram:primary"
    now = datetime.now()
    entry = SessionEntry(
        session_key="agent:static:telegram:dm:chat-1",
        session_id="session-1",
        created_at=now,
        updated_at=now,
    )
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(return_value=entry),
    )
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._resolve_async_delegation_session = AsyncMock(return_value=None)
    event = MessageEvent(
        text="hello",
        source=source,
        metadata={"gateway_session_id": "stop-after-observation"},
    )

    result = await runner._handle_message_with_agent(
        event,
        source,
        entry.session_key,
        1,
    )

    assert result is None
    assert "Failed to record dynamic profile session observation" in caplog.text


@pytest.mark.asyncio
async def test_secondary_session_skips_primary_account_observation(tmp_path):
    runner, _store = _runner_with_store(tmp_path)
    note_session = MagicMock()
    runner._profile_switching_service = SimpleNamespace(note_session=note_session)
    source = _event(runner).source
    source.profile = "origin"
    source.transport_owner_profile = "origin"
    now = datetime.now()
    entry = SessionEntry(
        session_key="agent:origin:telegram:dm:chat-1",
        session_id="session-1",
        created_at=now,
        updated_at=now,
    )
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(return_value=entry),
    )
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._resolve_async_delegation_session = AsyncMock(return_value=None)
    event = MessageEvent(
        text="hello",
        source=source,
        metadata={"gateway_session_id": "stop-after-observation"},
    )

    result = await runner._handle_message_with_agent(
        event,
        source,
        entry.session_key,
        1,
    )

    assert result is None
    note_session.assert_not_called()


@pytest.mark.asyncio
async def test_topic_recovery_keeps_transport_account_for_session_observation(
    tmp_path,
):
    runner, store = _runner_with_store(tmp_path)
    event = _event(runner)
    await runner._resolve_dynamic_profile_for_event(event)
    source = event.source
    now = datetime.now()
    entry = SessionEntry(
        session_key="agent:static:telegram:dm:chat-1:thread-7",
        session_id="session-1",
        created_at=now,
        updated_at=now,
    )
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(return_value=entry),
    )
    runner._recover_telegram_topic_thread_id = lambda _source: "thread-7"
    runner._resolve_async_delegation_session = AsyncMock(return_value=None)
    event.metadata["gateway_session_id"] = "stop-after-observation"

    result = await runner._handle_message_with_agent(
        event,
        source,
        entry.session_key,
        1,
    )

    observed_scope = ScopeKey(
        "telegram",
        "telegram:primary",
        "chat-1",
        "thread-7",
    )
    assert result is None
    assert store.get_session(observed_scope, "static") == "session-1"
