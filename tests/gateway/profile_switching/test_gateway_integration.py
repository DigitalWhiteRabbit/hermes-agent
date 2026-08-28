from __future__ import annotations

import asyncio
import os
import shutil
import threading
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
from gateway.profile_switching.models import ReasonCode, ScopeKey, ScopeKind
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


class _PairingStoreProbe:
    def __init__(
        self,
        *approvals: bool,
        profile: str | None = None,
        code: str | None = None,
    ) -> None:
        self.profile = profile
        self._approvals = list(approvals)
        self._code = code
        self.approval_calls: list[tuple[str, str, Path]] = []
        self.rate_limit_calls: list[tuple[str, str, Path]] = []
        self.record_rate_limit_calls: list[tuple[str, str, Path]] = []
        self.code_calls: list[tuple[str, str, str, Path]] = []

    def is_approved(self, platform: str, user_id: str) -> bool:
        self.approval_calls.append((platform, user_id, get_hermes_home()))
        return self._approvals.pop(0) if self._approvals else False

    def _is_rate_limited(self, platform: str, user_id: str) -> bool:
        self.rate_limit_calls.append((platform, user_id, get_hermes_home()))
        return False

    def _record_rate_limit(self, platform: str, user_id: str) -> None:
        self.record_rate_limit_calls.append(
            (platform, user_id, get_hermes_home())
        )

    def generate_code(
        self,
        platform: str,
        user_id: str,
        user_name: str,
    ) -> str | None:
        self.code_calls.append(
            (platform, user_id, user_name, get_hermes_home())
        )
        return self._code


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
    runner._is_user_authorized_for_source = lambda _source: True
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


@pytest.mark.asyncio
async def test_enabled_runner_lazily_initializes_real_service_and_default_policy():
    homes = _create_profile_homes()
    config = _gateway_config()
    path = homes["default"] / "state" / "profile-routing.db"

    runner = GatewayRunner(config)

    assert runner._profile_switching_service is None
    assert not path.exists()
    service = await runner._ensure_profile_switching_service()
    assert service is runner._profile_switching_service
    assert path.exists()
    assert runner._profile_switching_health["status"] == "ready"
    resolution = service.resolve(
        scope=ScopeKey("telegram", "telegram:primary", "chat-1", None),
        actor_user_id="user-1",
        consume_once=True,
        static_profile="default",
    )
    assert resolution.profile_name == "default"


@pytest.mark.asyncio
async def test_feature_off_constructor_and_first_use_do_no_routing_db_work(
    monkeypatch,
):
    homes = _create_profile_homes()
    path = homes["default"] / "state" / "profile-routing.db"
    from gateway.profile_switching.service import ProfileSwitchingService

    invoked = False

    def forbidden(*_args, **_kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("feature-off must not initialize routing SQLite")

    monkeypatch.setattr(ProfileSwitchingService, "from_gateway_config", forbidden)
    runner = GatewayRunner(_gateway_config(enabled=False))

    assert await runner._ensure_profile_switching_service() is None
    assert invoked is False
    assert not path.exists()
    assert runner._profile_switching_health["status"] == "disabled"


@pytest.mark.asyncio
async def test_lazy_routing_store_initialization_never_blocks_event_loop(
    monkeypatch,
):
    _create_profile_homes()
    from gateway.profile_switching.service import ProfileSwitchingService

    started = threading.Event()
    release = threading.Event()
    fake_service = object()

    def blocking_init(*_args, **_kwargs):
        started.set()
        assert release.wait(timeout=2)
        return fake_service

    monkeypatch.setattr(ProfileSwitchingService, "from_gateway_config", blocking_init)
    runner = GatewayRunner(_gateway_config())
    task = asyncio.create_task(runner._ensure_profile_switching_service())
    assert await asyncio.to_thread(started.wait, 1)

    loop_progressed = False

    async def tick():
        nonlocal loop_progressed
        await asyncio.sleep(0)
        loop_progressed = True

    await tick()
    assert loop_progressed is True
    release.set()
    assert await task is fake_service


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["database is locked", "permission denied"])
async def test_failed_lazy_initialization_enters_diagnostic_degraded_state(
    monkeypatch, failure
):
    _create_profile_homes()
    from gateway.profile_switching.service import ProfileSwitchingService

    def fail(*_args, **_kwargs):
        raise ProfileRoutingStoreUnavailable(failure)

    monkeypatch.setattr(ProfileSwitchingService, "from_gateway_config", fail)
    runner = GatewayRunner(_gateway_config())
    runner._is_user_authorized_for_source = lambda _source: True
    service = await runner._ensure_profile_switching_service()

    assert service is not None
    assert runner._profile_switching_service is None
    assert runner._profile_switching_health["status"] == "degraded"
    assert runner._profile_switching_health["reason"] == "store_unavailable"

    event = _event(runner)
    event.source.profile = "default"
    decision = service._policy.evaluate("default", _scope(event), "user-1")
    assert decision.allowed, decision
    await runner._resolve_dynamic_profile_for_event(event)
    assert event.source.profile == "default"
    assert event.source._dynamic_profile_resolution_source == "static"
    assert service.resolution_metrics[ReasonCode.DB_UNAVAILABLE.value] == 1


@pytest.mark.asyncio
async def test_degraded_lazy_initialization_recovers_on_a_later_turn(monkeypatch):
    _create_profile_homes()
    from gateway.profile_switching.service import ProfileSwitchingService

    real_factory = ProfileSwitchingService.from_gateway_config
    attempts = 0

    def flaky(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ProfileRoutingStoreUnavailable("database is locked")
        return real_factory(*args, **kwargs)

    monkeypatch.setattr(ProfileSwitchingService, "from_gateway_config", flaky)
    runner = GatewayRunner(_gateway_config())
    degraded = await runner._ensure_profile_switching_service()
    assert degraded is not None
    runner._profile_switching_retry_after = 0.0

    recovered = await runner._ensure_profile_switching_service()

    assert recovered is runner._profile_switching_service
    assert attempts == 2
    assert runner._profile_switching_health["status"] == "ready"


@pytest.mark.asyncio
async def test_failed_degraded_policy_build_respects_initialization_retry_cooldown(
    monkeypatch,
):
    _create_profile_homes()
    from gateway.profile_switching.service import ProfileSwitchingService

    attempts = 0
    degraded_attempts = 0

    def fail_store(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise ProfileRoutingStoreUnavailable("database is locked")

    def fail_policy(*_args, **_kwargs):
        nonlocal degraded_attempts
        degraded_attempts += 1
        raise RuntimeError("profile policy unavailable")

    monkeypatch.setattr(ProfileSwitchingService, "from_gateway_config", fail_store)
    monkeypatch.setattr(
        ProfileSwitchingService,
        "degraded_from_gateway_config",
        fail_policy,
    )
    runner = GatewayRunner(_gateway_config())

    assert await runner._ensure_profile_switching_service() is None
    assert await runner._ensure_profile_switching_service() is None
    assert attempts == 1
    assert degraded_attempts == 1
    assert runner._profile_switching_health["status"] == "degraded"


@pytest.mark.asyncio
async def test_lazy_initialization_reports_recoverable_corruption_quarantine():
    homes = _create_profile_homes()
    path = homes["default"] / "state" / "profile-routing.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"corrupt routing state")
    runner = GatewayRunner(_gateway_config())

    service = await runner._ensure_profile_switching_service()

    assert service is runner._profile_switching_service
    assert runner._profile_switching_health["status"] == "ready"
    assert runner._profile_switching_health["reason"] == "corrupt_store_quarantined"
    quarantined = runner._profile_switching_health["quarantined"]
    assert len(quarantined) == 1
    assert Path(quarantined[0]).read_bytes() == b"corrupt routing state"


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
async def test_primary_pairing_identity_is_stable_after_dynamic_resolution(
    tmp_path,
    monkeypatch,
):
    runner, store = _runner_with_store(tmp_path)
    event = _event(runner)
    store.set_once(
        _scope(event),
        profile_name="research",
        created_by_user_id="user-1",
    )
    primary_store = _PairingStoreProbe(True, True, True, profile="default")
    research_store = _PairingStoreProbe(False, profile="research")
    runner.pairing_store = primary_store
    runner.pairing_stores = {"research": research_store}
    runner._is_user_authorized_for_source = (
        GatewayRunner._is_user_authorized_for_source.__get__(runner)
    )
    monkeypatch.setattr("gateway.authz_mixin._auth_env", lambda *_args: "")
    monkeypatch.setattr(
        "gateway.authz_mixin._platform_gate_env",
        lambda _name, default="": default,
    )

    assert runner._primary_transport_allows_dynamic_routing(event) is True
    await runner._resolve_dynamic_profile_for_event(event)

    assert event.source.profile == "research"
    assert runner._session_key_for_source(event.source) == (
        "agent:research:telegram:dm:chat-1"
    )
    assert runner._is_user_authorized_for_source(event.source) is True
    assert store.claim_once(_scope(event)) is None
    assert len(primary_store.approval_calls) == 3
    assert research_store.approval_calls == []


@pytest.mark.asyncio
async def test_primary_denial_cannot_consume_route_approved_by_runtime_store(
    tmp_path,
    monkeypatch,
):
    runner, store = _runner_with_store(tmp_path)
    event = _event(runner)
    store.set_once(
        _scope(event),
        profile_name="research",
        created_by_user_id="user-1",
    )
    primary_store = _PairingStoreProbe(False, profile="default")
    research_store = _PairingStoreProbe(True, profile="research")
    runner.pairing_store = primary_store
    runner.pairing_stores = {"research": research_store}
    runner._is_user_authorized_for_source = (
        GatewayRunner._is_user_authorized_for_source.__get__(runner)
    )
    monkeypatch.setattr("gateway.authz_mixin._auth_env", lambda *_args: "")
    monkeypatch.setattr(
        "gateway.authz_mixin._platform_gate_env",
        lambda _name, default="": default,
    )

    await runner._resolve_dynamic_profile_for_event(event)

    assert event.source.profile is None
    assert event.source._dynamic_profile_resolution_source == "default"
    assert store.claim_once(_scope(event)).profile_name == "research"
    assert len(primary_store.approval_calls) == 1
    assert research_store.approval_calls == []


def test_pairing_store_transport_owner_isolated_from_runtime_profile(tmp_path):
    runner, _store = _runner_with_store(tmp_path)
    primary_store = _PairingStoreProbe(profile="default")
    research_store = _PairingStoreProbe(profile="research")
    runner.pairing_store = primary_store
    runner.pairing_stores = {"research": research_store}

    primary_routed = _event(runner).source
    primary_routed.profile = "research"
    primary_routed.transport_owner_profile = "default"
    secondary_owned = _event(runner).source
    secondary_owned.profile = "research"
    secondary_owned.transport_owner_profile = "research"

    assert runner._pairing_store_for(primary_routed) is primary_store
    assert runner._pairing_store_for(secondary_owned) is research_store

    runner.pairing_stores = {}
    assert runner._pairing_store_for(secondary_owned) is None


def test_pairing_store_transport_owner_feature_off_uses_runtime_profile(tmp_path):
    runner, _store = _runner_with_store(tmp_path)
    runner.config = _gateway_config(enabled=False)
    primary_store = _PairingStoreProbe(profile="default")
    research_store = _PairingStoreProbe(profile="research")
    runner.pairing_store = primary_store
    runner.pairing_stores = {"research": research_store}
    source = _event(runner).source
    source.profile = "research"
    source.transport_owner_profile = "default"

    assert runner._pairing_store_for(source) is research_store

    runner.config.profile_switching = SimpleNamespace(enabled=1)
    assert runner._pairing_store_for(source) is research_store


def test_secondary_transport_uses_only_its_pairing_approval_and_policy(
    tmp_path,
    monkeypatch,
):
    runner, _store = _runner_with_store(tmp_path)
    primary_adapter = _StubAdapter(
        PlatformConfig(enabled=True),
        Platform.TELEGRAM,
    )
    primary_adapter._dm_policy = "pairing"
    research_adapter = _StubAdapter(
        PlatformConfig(enabled=True),
        Platform.TELEGRAM,
    )
    research_adapter.gateway_runner = runner
    research_adapter.set_owner_profile("research")
    research_adapter._dm_policy = "disabled"
    runner.adapters = {Platform.TELEGRAM: primary_adapter}
    runner._profile_adapters = {
        "research": {Platform.TELEGRAM: research_adapter},
    }
    source = research_adapter.build_source(
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )
    source.profile = "research"
    primary_store = _PairingStoreProbe(True, profile="default")
    research_store = _PairingStoreProbe(False, True, profile="research")
    runner.pairing_store = primary_store
    runner.pairing_stores = {"research": research_store}
    runner._is_user_authorized_for_source = (
        GatewayRunner._is_user_authorized_for_source.__get__(runner)
    )
    monkeypatch.setattr("gateway.authz_mixin._auth_env", lambda *_args: "")
    monkeypatch.setattr(
        "gateway.authz_mixin._platform_gate_env",
        lambda _name, default="": default,
    )

    assert runner._is_user_authorized_for_source(source) is False
    assert runner._is_user_authorized_for_source(source) is True
    assert primary_store.approval_calls == []
    assert len(research_store.approval_calls) == 2
    assert runner._get_unauthorized_dm_behavior(
        source.platform,
        profile=runner._authorization_profile_for_source(source),
    ) == "ignore"


@pytest.mark.asyncio
async def test_secondary_explicit_ignore_sends_no_primary_pairing_response(
    tmp_path,
    monkeypatch,
):
    runner, _store = _runner_with_store(tmp_path)
    primary_adapter = _StubAdapter(
        PlatformConfig(
            enabled=True,
            extra={"unauthorized_dm_behavior": "pair"},
        ),
        Platform.TELEGRAM,
    )
    primary_adapter.gateway_runner = runner
    research_adapter = _StubAdapter(
        PlatformConfig(
            enabled=True,
            extra={"unauthorized_dm_behavior": "ignore"},
        ),
        Platform.TELEGRAM,
    )
    research_adapter.gateway_runner = runner
    research_adapter.set_owner_profile("research")
    research_adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: primary_adapter}
    runner._profile_adapters = {
        "research": {Platform.TELEGRAM: research_adapter},
    }
    event = MessageEvent(
        text="hello",
        source=research_adapter.build_source(
            chat_id="chat-1",
            chat_type="dm",
            user_id="user-1",
        ),
    )
    pairing_store = _PairingStoreProbe(
        False,
        profile="research",
        code="RESEARCH-CODE",
    )
    runner.pairing_stores = {"research": pairing_store}
    runner._is_user_authorized_for_source = lambda _source: False
    runner._scale_to_zero_note_real_inbound = lambda: None
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda *_args, **_kwargs: [],
    )

    result = await runner._handle_message(event)

    assert result is None
    research_adapter.send.assert_not_awaited()
    assert pairing_store.rate_limit_calls == []
    assert pairing_store.code_calls == []


@pytest.mark.parametrize("enabled", [False, 1])
def test_secondary_explicit_policy_preserves_primary_precedence_feature_off(
    tmp_path,
    enabled,
):
    runner, _store = _runner_with_store(tmp_path)
    runner.config = _gateway_config(enabled=False)
    runner.config.profile_switching = SimpleNamespace(enabled=enabled)
    runner.config.platforms[Platform.TELEGRAM] = PlatformConfig(
        enabled=True,
        extra={"unauthorized_dm_behavior": "pair"},
    )
    primary_adapter = _StubAdapter(
        runner.config.platforms[Platform.TELEGRAM],
        Platform.TELEGRAM,
    )
    research_adapter = _StubAdapter(
        PlatformConfig(
            enabled=True,
            extra={"unauthorized_dm_behavior": "ignore"},
        ),
        Platform.TELEGRAM,
    )
    research_adapter.set_owner_profile("research")
    runner.adapters = {Platform.TELEGRAM: primary_adapter}
    runner._profile_adapters = {
        "research": {Platform.TELEGRAM: research_adapter},
    }

    assert runner._get_unauthorized_dm_behavior(
        Platform.TELEGRAM,
        profile="research",
    ) == "pair"


@pytest.mark.asyncio
async def test_primary_canonical_pairing_uses_transport_policy_store_and_home(
    tmp_path,
    monkeypatch,
):
    runner, store = _runner_with_store(tmp_path)
    homes = _create_profile_homes()
    primary_adapter = _StubAdapter(
        PlatformConfig(enabled=True, extra={"dm_policy": "pairing"}),
        Platform.TELEGRAM,
    )
    primary_adapter.gateway_runner = runner
    primary_adapter._dm_policy = "pairing"
    primary_adapter.send = AsyncMock()
    research_adapter = _StubAdapter(
        PlatformConfig(enabled=True, extra={"dm_policy": "disabled"}),
        Platform.TELEGRAM,
    )
    research_adapter._dm_policy = "disabled"
    runner.adapters = {Platform.TELEGRAM: primary_adapter}
    runner._profile_adapters = {
        "research": {Platform.TELEGRAM: research_adapter},
    }
    event = MessageEvent(
        text="hello",
        source=primary_adapter.build_source(
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
    primary_store = _PairingStoreProbe(
        True,
        False,
        profile="default",
        code="PRIMARY-CODE",
    )
    research_store = _PairingStoreProbe(False, profile="research")
    runner.pairing_store = primary_store
    runner.pairing_stores = {"research": research_store}
    runner._is_user_authorized_for_source = (
        GatewayRunner._is_user_authorized_for_source.__get__(runner)
    )
    runner._scale_to_zero_note_real_inbound = lambda: None
    monkeypatch.setattr("gateway.authz_mixin._auth_env", lambda *_args: "")
    monkeypatch.setattr(
        "gateway.authz_mixin._platform_gate_env",
        lambda _name, default="": default,
    )
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda *_args, **_kwargs: [],
    )

    result = await runner._make_default_profile_message_handler()(event)

    assert result is None
    assert event.source.profile == "research"
    assert len(primary_store.approval_calls) == 2
    assert research_store.approval_calls == []
    assert primary_store.rate_limit_calls == [
        ("telegram", "user-1", homes["default"]),
    ]
    assert primary_store.code_calls == [
        ("telegram", "user-1", "", homes["default"]),
    ]
    primary_adapter.send.assert_awaited_once()
    assert "PRIMARY-CODE" in primary_adapter.send.await_args.args[1]


@pytest.mark.asyncio
async def test_unauthorized_primary_input_cannot_claim_or_enter_routed_busy_state(
    tmp_path,
):
    runner, store = _runner_with_store(tmp_path)
    homes = _create_profile_homes()
    event = _event(runner)
    event.source.profile_route_rejected = True
    store.set_once(
        _scope(event),
        profile_name="research",
        created_by_user_id="user-1",
    )
    runner._is_user_authorized_for_source = lambda _source: False
    adapter = _StubAdapter(
        PlatformConfig(enabled=True),
        Platform.TELEGRAM,
    )
    adapter.gateway_runner = runner
    handler = runner._make_default_profile_message_handler()
    adapter.set_message_handler(handler)
    captured_keys: list[str] = []
    adapter._start_session_processing = lambda _event, session_key: (
        captured_keys.append(session_key)
    )
    captured_runtime: list[tuple[Path, str | None]] = []

    async def capture(rejected_event: MessageEvent):
        captured_runtime.append((get_hermes_home(), rejected_event.source.profile))

    runner._handle_message = capture

    await adapter.handle_message(event)
    await handler(event)

    assert captured_keys == ["agent:main:telegram:dm:chat-1"]
    assert captured_runtime == [(homes["default"], None)]
    assert event.source.profile_route_rejected is False
    assert store.claim_once(_scope(event)).profile_name == "research"


@pytest.mark.asyncio
async def test_pairing_relevant_input_runs_canonical_pairing_only_in_default_scope(
    tmp_path, monkeypatch
):
    runner, store = _runner_with_store(tmp_path)
    homes = _create_profile_homes()
    event = _event(runner)
    store.set_once(
        _scope(event),
        profile_name="research",
        created_by_user_id="user-1",
    )
    runner._is_user_authorized_for_source = lambda _source: False
    runner._scale_to_zero_note_real_inbound = lambda: None
    hook_homes: list[Path] = []

    import hermes_cli.lifecycle as lifecycle

    monkeypatch.setattr(
        lifecycle,
        "invoke_hook",
        lambda *_args, **_kwargs: hook_homes.append(get_hermes_home()) or [],
    )

    pairing_store = SimpleNamespace(
        profile="default",
        _is_rate_limited=lambda *_args: False,
        generate_code=lambda *_args: "PAIR-CODE",
    )
    runner._pairing_store_for = lambda _source: pairing_store
    runner._get_unauthorized_dm_behavior = lambda *_args, **_kwargs: "pair"
    send = AsyncMock()
    runner._adapter_for_source = lambda _source: SimpleNamespace(send=send)

    result = await runner._make_default_profile_message_handler()(event)

    assert result is None
    assert event.source.profile is None
    assert hook_homes == [homes["default"]]
    send.assert_awaited_once()
    assert "PAIR-CODE" in send.await_args.args[1]
    assert store.claim_once(_scope(event)).profile_name == "research"


@pytest.mark.asyncio
async def test_ignored_slack_input_is_dropped_before_claim_or_busy_state(tmp_path):
    runner, store = _runner_with_store(tmp_path)
    runner.config.platforms[Platform.SLACK] = PlatformConfig(
        enabled=True,
        extra={"ignored_channels": ["C-ignored"]},
    )
    adapter = _StubAdapter(
        runner.config.platforms[Platform.SLACK],
        Platform.SLACK,
    )
    adapter.gateway_runner = runner
    event = MessageEvent(
        text="hello",
        source=adapter.build_source(
            chat_id="C-ignored",
            chat_type="channel",
            user_id="user-1",
        ),
    )
    scope = ScopeKey.from_source(event.source, account_id="slack:primary")
    store.set_once(
        scope,
        profile_name="research",
        created_by_user_id="user-1",
    )
    handler = AsyncMock()
    adapter.set_message_handler(handler)
    captured_keys: list[str] = []
    adapter._start_session_processing = lambda _event, session_key: (
        captured_keys.append(session_key)
    )

    await adapter.handle_message(event)

    assert captured_keys == []
    handler.assert_not_awaited()
    assert store.claim_once(scope).profile_name == "research"


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
    runner.config = _gateway_config(enabled=False)
    runner._profile_switching_service = None
    event = _event(runner)
    captured: dict[str, object] = {}

    async def capture(_event: MessageEvent):
        captured["runtime_home"] = get_hermes_home()

    runner._handle_message = capture

    await runner._make_profile_message_handler("origin")(event)

    assert captured["runtime_home"] == homes["origin"]
    assert event.source.profile == "static"
    assert event.source.transport_owner_profile is None
    assert event.source.transport_platform is None


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


def test_service_revalidates_deleted_profile_after_startup(tmp_path):
    homes = _create_profile_homes()
    config = _gateway_config()
    service = ProfileSwitchingService.from_gateway_config(
        config,
        db_path=tmp_path / "profile-routing.db",
    )
    scope = ScopeKey("telegram", "telegram:primary", "chat-2", None)
    service._store.set_binding(
        scope,
        ScopeKind.CHAT,
        profile_name="coder",
        created_by_user_id="user-1",
    )
    shutil.rmtree(homes["coder"])

    resolution = service.resolve(
        scope=scope,
        actor_user_id="user-1",
        consume_once=True,
        static_profile=None,
    )

    assert resolution.profile_name is None
    assert resolution.source.value == "default"


def test_explicit_allowlist_excludes_installed_production_from_served_topology(
    tmp_path,
):
    from gateway.run import _multiplex_profile_homes

    homes = _create_profile_homes()
    production_home = get_hermes_home() / "profiles" / "production"
    production_home.mkdir(parents=True)
    config = GatewayConfig(
        multiplex_profiles=True,
        multiplex_profile_allowlist=["coder"],
        profile_switching=ProfileSwitchingConfig(
            enabled=True,
            rules=(
                ProfileSwitchRule(
                    "coder",
                    users=("user-1",),
                    chats=("*",),
                ),
            ),
        ),
    )

    multiplex_homes = dict(_multiplex_profile_homes(config))
    service = ProfileSwitchingService.from_gateway_config(
        config,
        db_path=tmp_path / "profile-routing.db",
    )

    assert multiplex_homes == {
        "default": homes["default"],
        "coder": homes["coder"],
    }
    assert "production" not in service._policy._served_profiles


def test_service_revalidates_unreadable_profile_after_startup(tmp_path):
    homes = _create_profile_homes()
    config = _gateway_config()
    service = ProfileSwitchingService.from_gateway_config(
        config,
        db_path=tmp_path / "profile-routing.db",
    )
    scope = ScopeKey("telegram", "telegram:primary", "chat-2", None)
    service._store.set_binding(
        scope,
        ScopeKind.CHAT,
        profile_name="coder",
        created_by_user_id="user-1",
    )
    homes["coder"].chmod(0)
    try:
        assert not os.access(homes["coder"], os.R_OK | os.X_OK)
        resolution = service.resolve(
            scope=scope,
            actor_user_id="user-1",
            consume_once=True,
            static_profile=None,
        )
    finally:
        homes["coder"].chmod(0o700)

    assert resolution.profile_name is None
    assert resolution.source.value == "default"


@pytest.mark.asyncio
async def test_deleted_profile_between_resolution_and_runtime_fails_the_turn(
    tmp_path,
):
    runner, store = _runner_with_store(tmp_path)
    homes = _create_profile_homes()
    event = _event(runner)
    _set_binding(store, _scope(event), ScopeKind.CHAT, "coder")
    resolve = runner._resolve_dynamic_profile_for_event

    async def resolve_then_delete(resolved_event: MessageEvent):
        await resolve(resolved_event)
        shutil.rmtree(homes["coder"], ignore_errors=True)

    runner._resolve_dynamic_profile_for_event = resolve_then_delete
    runner._handle_message = AsyncMock()

    result = await runner._make_default_profile_message_handler()(event)

    assert "coder" in result
    assert "unavailable" in result.lower()
    runner._handle_message.assert_not_awaited()
    assert get_hermes_home() == homes["default"]


@pytest.mark.asyncio
async def test_profile_home_toctou_at_scope_entry_never_enters_default_runtime(
    tmp_path,
):
    runner, _store = _runner_with_store(tmp_path)
    homes = _create_profile_homes()
    event = _event(runner)
    event.source.profile = "coder"
    event.source._dynamic_profile_resolved = True
    runner._handle_message = AsyncMock()

    def resolve_then_delete(_source: SessionSource) -> Path:
        shutil.rmtree(homes["coder"], ignore_errors=True)
        return homes["coder"]

    runner._resolve_profile_home_for_source = resolve_then_delete

    result = await runner._make_default_profile_message_handler()(event)

    assert "coder" in result
    assert "unavailable" in result.lower()
    runner._handle_message.assert_not_awaited()
    assert get_hermes_home() == homes["default"]


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
    queued: list[MessageEvent] = []
    runner._startup_restore_in_progress = True
    runner._queue_startup_restore_event = queued.append

    result = await GatewayRunner._handle_message(runner, event)

    assert result is None
    assert event.source.profile == "research"
    assert event.source.profile_route_rejected is False
    assert queued == [event]


@pytest.mark.asyncio
async def test_feature_off_keeps_real_static_route_rejection(tmp_path):
    runner, _store = _runner_with_store(tmp_path)
    runner.config = _gateway_config(enabled=False)
    runner._profile_switching_service = None
    event = _event(runner)
    event.source.profile = None
    event.source.profile_route_rejected = True
    queued: list[MessageEvent] = []
    runner._startup_restore_in_progress = True
    runner._queue_startup_restore_event = queued.append

    assert await GatewayRunner._handle_message(runner, event) is None
    assert event.source.profile_route_rejected is True
    assert queued == []


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
    monkeypatch,
    denial,
    db_unavailable,
):
    runner, _store = _runner_with_store(tmp_path)
    homes = _create_profile_homes()
    for profile in ("default", "static"):
        (homes[profile] / "config.yaml").write_text(
            f"scope_probe: {profile}\n",
            encoding="utf-8",
        )
        (homes[profile] / ".env").write_text(
            f"PROFILE_ROUTING_SCOPE_PROBE={profile}\n",
            encoding="utf-8",
        )
    monkeypatch.delenv("PROFILE_ROUTING_SCOPE_PROBE", raising=False)
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

    # Once dynamic policy rejects the static candidate, every later runtime
    # lookup must honor that finalized default result. In particular, busy
    # policy and both preprocessing/agent scopes must not re-run static routes.
    runner._busy_input_mode = "interrupt"
    runner._busy_input_modes_by_profile = {"static": "queue"}
    assert runner._effective_busy_input_mode(event.source) == "interrupt"
    assert runner._resolve_profile_home_for_source(event.source) == homes["default"]

    def scope_snapshot() -> tuple[Path, str, str | None]:
        from agent.secret_scope import get_secret
        from gateway.run import _load_gateway_config

        home = get_hermes_home()
        return (
            home,
            _load_gateway_config().get("scope_probe"),
            get_secret("PROFILE_ROUTING_SCOPE_PROBE"),
        )

    runtime_scopes: dict[str, tuple[Path, str, str | None]] = {}

    async def capture_preprocessing(**_kwargs):
        runtime_scopes["preprocessing"] = scope_snapshot()
        return "prepared"

    async def capture_agent(*_args, **_kwargs):
        runtime_scopes["agent"] = scope_snapshot()
        return {"final_response": "ok"}

    runner._prepare_inbound_message_text = capture_preprocessing
    runner._run_agent_inner = capture_agent
    assert (
        await runner._prepare_profile_scoped_inbound_message_text(
            event=event,
            source=event.source,
            history=[],
        )
        == "prepared"
    )
    assert await runner._run_agent(
        "hello",
        "context",
        [],
        event.source,
        "session-1",
    ) == {"final_response": "ok"}
    expected_scope = (homes["default"], "default", "default")
    assert runtime_scopes == {
        "preprocessing": expected_scope,
        "agent": expected_scope,
    }

    captured = await _dispatch_through_primary_wrapper(runner, event)
    assert captured["profile"] is None
    assert captured["session_key"] == "agent:main:telegram:dm:chat-1"

    direct = _event(runner)
    direct.source.profile_route_rejected = True
    runner._startup_restore_in_progress = True
    queued: list[MessageEvent] = []
    runner._queue_startup_restore_event = queued.append
    assert await GatewayRunner._handle_message(runner, direct) is None
    assert direct.source.profile is None
    assert direct.source.profile_route_rejected is False
    assert queued == [direct]


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
