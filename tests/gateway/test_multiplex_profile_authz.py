"""Regression tests for multiplex profile-aware own-policy authorization."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


class _PlatformClassSpoof:
    """Object that fools ``isinstance(value, Platform)`` without being an enum."""

    @property
    def __class__(self):
        return Platform


class _RelayEqualPlatformSpoof(_PlatformClassSpoof):
    """Spoof that can also collide with the Relay enum key in a dict lookup."""

    def __hash__(self):
        return hash(Platform.RELAY)

    def __eq__(self, other):
        return other is Platform.RELAY


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "WECOM_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "WECOM_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_transport_lookup_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    native = SimpleNamespace(
        authorization_is_upstream=False,
        enforces_own_access_policy=False,
    )
    discord = object()
    relay = SimpleNamespace(
        authorization_is_upstream=True,
        enforces_own_access_policy=False,
    )
    runner.adapters = {
        Platform.TELEGRAM: native,
        Platform.DISCORD: discord,
        Platform.RELAY: relay,
    }
    runner._profile_adapters = {}
    runner.config = GatewayConfig()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    return runner, native, discord, relay


def _transport_source(transport_platform):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-1",
        chat_id="chat-1",
        chat_type="dm",
        profile=None,
        transport_owner_profile="default",
    )
    source.transport_platform = transport_platform
    return source


@pytest.mark.parametrize(
    ("transport_platform", "expected_name"),
    [
        pytest.param(Platform.DISCORD, "discord", id="exact-discord-enum"),
        pytest.param(Platform.RELAY, "relay", id="exact-relay-enum"),
        pytest.param("discord", "discord", id="valid-discord-string"),
        pytest.param("relay", "relay", id="valid-relay-string"),
    ],
)
def test_adapter_lookup_accepts_only_real_persisted_transport_kinds(
    transport_platform,
    expected_name,
):
    runner, _native, discord, relay = _make_transport_lookup_runner()

    expected = {"discord": discord, "relay": relay}[expected_name]
    assert runner._adapter_for_source(_transport_source(transport_platform)) is expected


@pytest.mark.parametrize(
    "transport_platform",
    [
        pytest.param(None, id="none"),
        pytest.param("future-transport", id="invalid-string"),
        pytest.param(object(), id="arbitrary-object"),
        pytest.param(MagicMock(), id="bare-magic-mock"),
        pytest.param(MagicMock(spec=Platform), id="platform-spec-magic-mock"),
        pytest.param(_PlatformClassSpoof(), id="class-spoofing-proxy"),
        pytest.param(_RelayEqualPlatformSpoof(), id="relay-equal-class-spoof"),
    ],
)
def test_invalid_persisted_transport_kinds_fall_back_to_native_adapter(
    transport_platform,
):
    runner, native, _discord, _relay = _make_transport_lookup_runner()

    assert runner._adapter_for_source(_transport_source(transport_platform)) is native


def test_bare_magic_mock_source_uses_native_platform_fallback():
    runner, native, _discord, _relay = _make_transport_lookup_runner()
    source = MagicMock()
    source.platform = Platform.TELEGRAM
    source.profile = None
    source.transport_owner_profile = None
    source.delivered_via_upstream_relay = False
    source._transport_adapter_ref = None

    assert runner._adapter_for_source(source) is native


def test_spoofed_transport_platform_cannot_gain_relay_authorization(monkeypatch):
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)
    runner, _native, _discord, _relay = _make_transport_lookup_runner()

    assert runner._is_user_authorized(
        _transport_source(_RelayEqualPlatformSpoof())
    ) is False


def test_relay_delivery_marker_preserves_relay_adapter_with_invalid_provenance():
    runner, _native, _discord, relay = _make_transport_lookup_runner()
    source = _transport_source(_PlatformClassSpoof())
    source.delivered_via_upstream_relay = True

    assert runner._adapter_for_source(source) is relay


def _make_multiplex_runner(monkeypatch):
    """Runner with default allowlist WeCom and secondary open-policy WeCom."""
    from gateway.run import GatewayRunner

    _clear_auth_env(monkeypatch)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)

    default_adapter = SimpleNamespace(
        send=AsyncMock(),
        enforces_own_access_policy=True,
        _dm_policy="allowlist",
        _group_policy="pairing",
    )
    secondary_adapter = SimpleNamespace(
        send=AsyncMock(),
        enforces_own_access_policy=True,
        _dm_policy="open",
        _group_policy="open",
    )

    runner.adapters = {Platform.WECOM: default_adapter}
    runner._profile_adapters = {
        "coder": {Platform.WECOM: secondary_adapter},
    }
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    return runner, default_adapter, secondary_adapter


def test_default_profile_still_trusts_own_allowlist(monkeypatch):
    """Default-profile allowlist trust is unchanged when profile is unstamped."""
    runner, _default_adapter, _secondary_adapter = _make_multiplex_runner(monkeypatch)

    source = SessionSource(
        platform=Platform.WECOM,
        user_id="allowed-user",
        chat_id="dm-chat",
        user_name="allowed-user",
        chat_type="dm",
        profile=None,
    )

    assert runner._is_user_authorized(source) is True


def test_active_profile_stamp_resolves_primary_adapter(monkeypatch):
    """A single-profile gateway stamps its active profile but stores adapters as primary."""
    runner, default_adapter, _secondary_adapter = _make_multiplex_runner(monkeypatch)
    runner._active_profile_name = lambda: "dev"

    assert runner._authorization_adapter(Platform.WECOM, profile="dev") is default_adapter


def test_secondary_allowlist_dm_behavior_ignores_unauthorized(monkeypatch):
    """Unauthorized-DM behavior must read the secondary adapter's dm_policy."""
    runner, _default_adapter, secondary_adapter = _make_multiplex_runner(monkeypatch)
    secondary_adapter._dm_policy = "allowlist"

    assert runner._get_unauthorized_dm_behavior(
        Platform.WECOM,
        profile="coder",
    ) == "ignore"
    assert runner._get_unauthorized_dm_behavior(Platform.WECOM) == "ignore"


def test_adapter_auth_check_stamps_secondary_profile(monkeypatch):
    """The adapter auth-check callback must stamp its own secondary profile.

    Regression for the gap where ``_make_adapter_auth_check`` built a
    profile-less ``SessionSource``, so a secondary adapter's external-context
    authorization (e.g. Slack/Discord thread-reply lookups) silently
    resolved the *active* profile's allowlist scope instead of its own.
    """
    from gateway.run import GatewayRunner

    _clear_auth_env(monkeypatch)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)

    captured: dict = {}

    def fake_is_user_authorized(source):
        captured["profile"] = source.profile
        return True

    runner._is_user_authorized = fake_is_user_authorized

    check = runner._make_adapter_auth_check(Platform.WECOM, profile_name="coder")
    assert check("some-user", "dm", "dm-chat") is True
    assert captured["profile"] == "coder"


def test_startup_guard_gateway_allow_all_reads_scope_not_environ(monkeypatch):
    """The GATEWAY_ALLOW_ALL_USERS opt-in check inside the startup guard
    must honor the active profile secret scope (#93522): the default
    profile's env-only opt-in must not leak into a secondary profile that
    never opted in, and a secondary profile's own scoped opt-in must be
    honored."""
    from agent import secret_scope
    from gateway.run import _own_policy_open_startup_violation

    _clear_auth_env(monkeypatch)
    cfg = GatewayConfig(multiplex_profiles=True)
    cfg.platforms = {
        Platform.WECOM: PlatformConfig(enabled=True, extra={"dm_policy": "open"}),
    }

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    try:
        token = secret_scope.set_secret_scope({"SOMETHING_ELSE": "x"})
        try:
            violation = _own_policy_open_startup_violation(cfg)
        finally:
            secret_scope.reset_secret_scope(token)
        assert violation is not None, "default profile's env opt-in must not leak into the scoped secondary profile"

        token = secret_scope.set_secret_scope({"GATEWAY_ALLOW_ALL_USERS": "true"})
        try:
            violation = _own_policy_open_startup_violation(cfg)
        finally:
            secret_scope.reset_secret_scope(token)
        assert violation is None, "the secondary profile's own scoped opt-in must be honored"
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)


def test_secondary_open_policy_fails_startup_guard(monkeypatch):
    """Secondary profiles must pass the same open-policy startup guard."""
    from gateway.run import _own_policy_open_startup_violation

    _clear_auth_env(monkeypatch)

    secondary_cfg = GatewayConfig(multiplex_profiles=True)
    secondary_cfg.platforms = {
        Platform.WECOM: PlatformConfig(
            enabled=True,
            extra={"dm_policy": "open"},
        ),
    }

    violation = _own_policy_open_startup_violation(secondary_cfg)
    assert violation is not None
    assert "wecom" in violation
    assert "open policy" in violation
