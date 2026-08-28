from dataclasses import FrozenInstanceError

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from gateway.config import (
    GatewayConfig,
    ProfileSwitchRule,
    ProfileSwitchingConfig,
)
from gateway.profile_switching.models import ReasonCode, ScopeKey
from gateway.profile_switching.policy import ProfilePolicy


_EXPLICIT_ALLOWLIST_ERROR = (
    "profile_switching.enabled requires an explicit "
    "gateway.multiplex_profile_allowlist; use [] for default-only"
)
_DEFAULT_PRIMARY_ERROR = (
    "profile_switching.enabled requires the default profile gateway; "
    "launch with hermes -p default gateway run"
)
_SENSITIVE_PROFILE_ERROR = (
    "profile 'production' cannot be served, visible, or switched "
    "in Milestone 1; it may only be named in profile_switching.hidden"
)


def test_profile_switching_defaults_disabled():
    cfg = GatewayConfig.from_dict({})
    assert cfg.profile_switching.enabled is False
    assert cfg.profile_switching.picker_ttl_seconds == 300
    assert "profile_switching" not in cfg.to_dict()


def test_nested_profile_switching_parses():
    cfg = GatewayConfig.from_dict({
        "gateway": {
            "multiplex_profiles": True,
            "multiplex_profile_allowlist": ["coder"],
            "profile_switching": {
                "enabled": True,
                "default_visible": ["default", "coder"],
                "hidden": ["production"],
                "admins": {"telegram": [123]},
                "rules": [{
                    "profile": "coder",
                    "users": [123],
                    "chats": ["*"],
                }],
            },
        }
    })
    assert cfg.profile_switching.enabled is True
    assert cfg.profile_switching.admins["telegram"] == ("123",)
    assert cfg.profile_switching.rules[0].profile == "coder"


def test_enabled_switching_requires_multiplex():
    with pytest.raises(ValueError, match="requires gateway.multiplex_profiles"):
        GatewayConfig.from_dict({
            "gateway": {"profile_switching": {"enabled": True}}
        })


def test_enabled_switching_rejects_omitted_nested_allowlist():
    with pytest.raises(ValueError) as exc_info:
        GatewayConfig.from_dict({
            "gateway": {
                "multiplex_profiles": True,
                "profile_switching": {"enabled": True},
            }
        })
    assert str(exc_info.value) == _EXPLICIT_ALLOWLIST_ERROR


def test_enabled_switching_rejects_explicit_null_nested_allowlist():
    with pytest.raises(ValueError) as exc_info:
        GatewayConfig.from_dict({
            "gateway": {
                "multiplex_profiles": True,
                "multiplex_profile_allowlist": None,
                "profile_switching": {"enabled": True},
            }
        })
    assert str(exc_info.value) == _EXPLICIT_ALLOWLIST_ERROR


def test_top_level_null_allowlist_overrides_nested_safe_list_and_is_rejected():
    with pytest.raises(ValueError) as exc_info:
        GatewayConfig.from_dict({
            "multiplex_profile_allowlist": None,
            "gateway": {
                "multiplex_profiles": True,
                "multiplex_profile_allowlist": ["coder"],
                "profile_switching": {"enabled": True},
            },
        })
    assert str(exc_info.value) == _EXPLICIT_ALLOWLIST_ERROR


def test_direct_constructor_rejects_enabled_switching_without_allowlist():
    with pytest.raises(ValueError) as exc_info:
        GatewayConfig(
            multiplex_profiles=True,
            profile_switching=ProfileSwitchingConfig(enabled=True),
        )
    assert str(exc_info.value) == _EXPLICIT_ALLOWLIST_ERROR


def test_named_profile_config_fails_before_gateway_state_is_initialized(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "hermes"
    profile_home = root / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "gateway:\n"
        "  multiplex_profiles: true\n"
        "  multiplex_profile_allowlist: []\n"
        "  profile_switching:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)

    from gateway.run import load_gateway_config_for_runner

    with pytest.raises(ValueError) as exc_info:
        load_gateway_config_for_runner()

    assert str(exc_info.value) == _DEFAULT_PRIMARY_ERROR
    assert not (profile_home / "state.db").exists()
    assert not (profile_home / "state" / "profile-routing.db").exists()


def test_direct_constructor_rejects_feature_on_from_named_profile(
    tmp_path,
    monkeypatch,
):
    profile_home = tmp_path / "hermes" / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    with pytest.raises(ValueError) as exc_info:
        GatewayConfig(
            multiplex_profiles=True,
            multiplex_profile_allowlist=[],
            profile_switching=ProfileSwitchingConfig(enabled=True),
        )

    assert str(exc_info.value) == _DEFAULT_PRIMARY_ERROR


def test_named_profile_feature_off_and_default_profile_feature_on_remain_valid(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "hermes"
    profile_home = root / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    named_feature_off = GatewayConfig(
        multiplex_profiles=True,
        multiplex_profile_allowlist=None,
        profile_switching=ProfileSwitchingConfig(enabled=False),
    )

    monkeypatch.setenv("HERMES_HOME", str(root))
    default_feature_on = GatewayConfig(
        multiplex_profiles=True,
        multiplex_profile_allowlist=[],
        profile_switching=ProfileSwitchingConfig(enabled=True),
    )

    assert named_feature_off.profile_switching.enabled is False
    assert default_feature_on.profile_switching.enabled is True


def test_secondary_context_scope_does_not_override_default_process_ownership(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "hermes"
    profile_home = root / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    home_token = set_hermes_home_override(profile_home)
    try:
        config = GatewayConfig(
            multiplex_profiles=True,
            multiplex_profile_allowlist=[],
            profile_switching=ProfileSwitchingConfig(enabled=True),
        )
    finally:
        reset_hermes_home_override(home_token)

    assert config.profile_switching.enabled is True


def test_runner_revalidates_process_profile_before_session_state(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "hermes"
    profile_home = root / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    config = GatewayConfig(
        multiplex_profiles=True,
        multiplex_profile_allowlist=[],
        profile_switching=ProfileSwitchingConfig(enabled=True),
    )
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    from gateway.run import GatewayRunner

    with pytest.raises(ValueError) as exc_info:
        GatewayRunner(config)

    assert str(exc_info.value) == _DEFAULT_PRIMARY_ERROR
    assert not (profile_home / "state.db").exists()
    assert not (profile_home / "sessions").exists()


def test_symlinked_named_process_home_cannot_alias_the_default_gateway(
    tmp_path,
    monkeypatch,
):
    profile_home = tmp_path / "hermes" / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    launch_home = tmp_path / "current-home"
    launch_home.symlink_to(profile_home, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    with pytest.raises(ValueError) as exc_info:
        GatewayConfig(
            multiplex_profiles=True,
            multiplex_profile_allowlist=[],
            profile_switching=ProfileSwitchingConfig(enabled=True),
        )

    assert str(exc_info.value) == _DEFAULT_PRIMARY_ERROR


def test_nested_symlinked_named_process_home_cannot_alias_default(
    tmp_path,
    monkeypatch,
):
    external_home = tmp_path / "external-coder"
    external_home.mkdir()
    profile_home = tmp_path / "hermes" / "profiles" / "coder"
    profile_home.parent.mkdir(parents=True)
    profile_home.symlink_to(external_home, target_is_directory=True)
    launch_home = tmp_path / "current-home"
    launch_home.symlink_to(profile_home, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    with pytest.raises(ValueError) as exc_info:
        GatewayConfig(
            multiplex_profiles=True,
            multiplex_profile_allowlist=[],
            profile_switching=ProfileSwitchingConfig(enabled=True),
        )

    assert str(exc_info.value) == _DEFAULT_PRIMARY_ERROR


def test_direct_mixed_case_hidden_profile_remains_hidden_to_policy():
    config = GatewayConfig(
        multiplex_profiles=True,
        multiplex_profile_allowlist=["coder"],
        profile_switching=ProfileSwitchingConfig(
            enabled=True,
            hidden=("Coder",),
            rules=(ProfileSwitchRule("coder", users=("user-1",), chats=("*",)),),
        ),
    )
    policy = ProfilePolicy(
        config.profile_switching,
        served_profiles={"default", "coder"},
        existing_profiles={"default", "coder"},
    )

    decision = policy.evaluate(
        "coder",
        ScopeKey("telegram", "telegram:primary", "chat-1", None),
        "user-1",
    )

    assert decision.reason is ReasonCode.PROFILE_HIDDEN


def test_direct_mixed_case_visible_rule_is_usable_by_canonical_profile():
    config = GatewayConfig(
        multiplex_profiles=True,
        multiplex_profile_allowlist=["coder"],
        profile_switching=ProfileSwitchingConfig(
            enabled=True,
            default_visible=("Coder",),
            rules=(ProfileSwitchRule("Coder", users=("user-1",), chats=("*",)),),
        ),
    )
    policy = ProfilePolicy(
        config.profile_switching,
        served_profiles={"default", "coder"},
        existing_profiles={"default", "coder"},
    )

    visible = policy.visible_profiles(
        ScopeKey("telegram", "telegram:primary", "chat-1", None),
        "user-1",
    )

    assert visible == ("coder",)


@pytest.mark.parametrize("profile", ["production", "Production"])
@pytest.mark.parametrize("location", ["allowlist", "visible", "rule"])
def test_direct_constructor_rejects_sensitive_profile_like_parser(location, profile):
    allowlist = ["coder"]
    switching = ProfileSwitchingConfig(enabled=True)
    if location == "allowlist":
        allowlist = [profile]
    elif location == "visible":
        switching = ProfileSwitchingConfig(
            enabled=True,
            default_visible=(profile,),
        )
    else:
        switching = ProfileSwitchingConfig(
            enabled=True,
            rules=(ProfileSwitchRule(profile, users=("user-1",)),),
        )

    with pytest.raises(ValueError) as exc_info:
        GatewayConfig(
            multiplex_profiles=True,
            multiplex_profile_allowlist=allowlist,
            profile_switching=switching,
        )

    assert str(exc_info.value) == _SENSITIVE_PROFILE_ERROR


@pytest.mark.parametrize(
    "switching,error",
    [
        (
            ProfileSwitchingConfig(default_visible="production"),
            "default_visible must be a list",
        ),
        (
            ProfileSwitchingConfig(hidden="production"),
            "hidden must be a list",
        ),
        (
            ProfileSwitchingConfig(
                default_visible=("coder",),
                hidden=("Coder",),
            ),
            "both visible and hidden",
        ),
        (
            ProfileSwitchingConfig(default_visible=("research",)),
            "default_visible profiles are not served",
        ),
        (
            ProfileSwitchingConfig(
                rules=(
                    ProfileSwitchRule("coder"),
                    ProfileSwitchRule("Coder"),
                )
            ),
            "duplicate profile_switching rule",
        ),
        (
            ProfileSwitchingConfig(
                rules=(ProfileSwitchRule("coder", require_confirmation=True),)
            ),
            "require_confirmation=true.*unsupported",
        ),
        (
            ProfileSwitchingConfig(
                rules=(ProfileSwitchRule("coder", threads=("thread-1",)),)
            ),
            "threads requires concrete chat IDs",
        ),
        (
            ProfileSwitchingConfig(rules=(ProfileSwitchRule("coder", users=("",)),)),
            "contains an invalid ID",
        ),
    ],
)
def test_direct_constructor_enforces_shared_policy_invariants(switching, error):
    with pytest.raises(ValueError, match=error):
        GatewayConfig(
            multiplex_profiles=True,
            multiplex_profile_allowlist=["coder"],
            profile_switching=switching,
        )


def test_non_literal_enabled_integer_stays_off_across_roundtrip():
    config = GatewayConfig(
        multiplex_profiles=True,
        profile_switching=ProfileSwitchingConfig(enabled=1),
    )

    serialized = config.to_dict()
    restored = GatewayConfig.from_dict(serialized)

    assert config.profile_switching.enabled is not True
    assert "profile_switching" not in serialized
    assert restored.profile_switching.enabled is False


def test_policy_error_precedes_positive_limit_error_like_legacy_parser():
    with pytest.raises(ValueError, match="duplicate.*coder"):
        GatewayConfig.from_dict({
            "multiplex_profile_allowlist": ["coder"],
            "profile_switching": {
                "picker_ttl_seconds": 0,
                "rules": [
                    {"profile": "coder"},
                    {"profile": "Coder"},
                ],
            },
        })


def test_enabled_switching_accepts_explicit_empty_allowlist_as_default_only():
    cfg = GatewayConfig.from_dict({
        "gateway": {
            "multiplex_profiles": True,
            "multiplex_profile_allowlist": [],
            "profile_switching": {"enabled": True},
        }
    })

    assert cfg.multiplex_profile_allowlist == []
    from gateway.run import _multiplex_profile_homes

    assert [name for name, _home in _multiplex_profile_homes(cfg)] == ["default"]


@pytest.mark.parametrize(
    "raw",
    [
        {"gateway": {"multiplex_profiles": True}},
        {
            "gateway": {
                "multiplex_profiles": True,
                "multiplex_profile_allowlist": None,
            }
        },
    ],
)
def test_feature_off_preserves_omitted_and_null_serve_all_allowlist(raw):
    cfg = GatewayConfig.from_dict(raw)

    assert cfg.profile_switching.enabled is False
    assert cfg.multiplex_profile_allowlist is None


def test_visible_profile_must_be_served():
    with pytest.raises(ValueError, match="not served"):
        GatewayConfig.from_dict({
            "gateway": {
                "multiplex_profiles": True,
                "multiplex_profile_allowlist": ["coder"],
                "profile_switching": {
                    "enabled": True,
                    "default_visible": ["research"],
                },
            }
        })


def test_visible_profile_must_be_served_before_switching_is_enabled():
    with pytest.raises(ValueError, match="not served"):
        GatewayConfig.from_dict({
            "multiplex_profile_allowlist": ["coder"],
            "profile_switching": {
                "default_visible": ["research"],
            },
        })


def test_unknown_profile_switching_key_is_rejected():
    with pytest.raises(ValueError, match="unknown profile_switching keys"):
        GatewayConfig.from_dict({
            "gateway": {
                "multiplex_profiles": True,
                "profile_switching": {"enabled": True, "enabeld": True},
            }
        })


def test_default_profile_is_implicitly_served():
    cfg = GatewayConfig.from_dict({
        "gateway": {
            "multiplex_profiles": True,
            "multiplex_profile_allowlist": [],
            "profile_switching": {
                "enabled": True,
                "default_visible": ["default"],
            },
        }
    })
    assert cfg.profile_switching.default_visible == ("default",)


def test_hidden_profile_may_be_unserved():
    cfg = GatewayConfig.from_dict({
        "gateway": {
            "multiplex_profiles": True,
            "multiplex_profile_allowlist": ["coder"],
            "profile_switching": {
                "enabled": True,
                "hidden": ["production"],
            },
        }
    })
    assert cfg.profile_switching.hidden == ("production",)


def test_visible_and_hidden_profiles_must_not_overlap():
    with pytest.raises(ValueError, match="both visible and hidden"):
        GatewayConfig.from_dict({
            "gateway": {
                "multiplex_profiles": True,
                "profile_switching": {
                    "enabled": True,
                    "default_visible": ["Coder"],
                    "hidden": ["coder"],
                },
            }
        })


@pytest.mark.parametrize(
    "key,value",
    [
        ("picker_ttl_seconds", 0),
        ("audit_retention_days", -1),
        ("audit_max_rows", True),
        ("audit_max_rows", "many"),
    ],
)
def test_positive_limits_reject_invalid_values(key, value):
    with pytest.raises(ValueError, match=key):
        GatewayConfig.from_dict({"profile_switching": {key: value}})


def test_admin_and_rule_ids_are_coerced_to_strings():
    cfg = GatewayConfig.from_dict({
        "profile_switching": {
            "admins": {"telegram": [123, "456"]},
            "rules": [
                {
                    "profile": "coder",
                    "users": [123],
                    "chats": [-100456],
                    "threads": [789],
                    "require_confirmation": False,
                }
            ],
        }
    })
    assert cfg.profile_switching.admins == {"telegram": ("123", "456")}
    assert cfg.profile_switching.rules[0].users == ("123",)
    assert cfg.profile_switching.rules[0].chats == ("-100456",)
    assert cfg.profile_switching.rules[0].threads == ("789",)
    assert cfg.profile_switching.rules[0].require_confirmation is False


@pytest.mark.parametrize("value", [True, "true", 1])
def test_require_confirmation_true_is_rejected_for_milestone_one(value):
    with pytest.raises(ValueError, match="require_confirmation.*unsupported"):
        GatewayConfig.from_dict({
            "profile_switching": {
                "rules": [
                    {
                        "profile": "coder",
                        "require_confirmation": value,
                    }
                ],
            }
        })


@pytest.mark.parametrize("value", ["", "   ", True, False])
@pytest.mark.parametrize("kind", ["users", "chats", "threads", "admins"])
def test_blank_and_boolean_policy_ids_are_rejected(kind, value):
    if kind == "admins":
        raw = {"admins": {"telegram": [value]}}
    else:
        raw = {"rules": [{"profile": "coder", kind: [value]}]}
    with pytest.raises(ValueError, match="invalid ID"):
        GatewayConfig.from_dict({"profile_switching": raw})


def test_duplicate_profile_rules_are_rejected_after_normalization():
    with pytest.raises(ValueError, match="duplicate.*coder"):
        GatewayConfig.from_dict({
            "profile_switching": {
                "rules": [
                    {"profile": "Coder", "users": ["one"]},
                    {"profile": "coder", "users": ["two"]},
                ],
            }
        })


@pytest.mark.parametrize("chats", [[], ["*"]])
def test_thread_rules_require_concrete_non_wildcard_chats(chats):
    with pytest.raises(ValueError, match="threads.*concrete chat"):
        GatewayConfig.from_dict({
            "profile_switching": {
                "rules": [
                    {
                        "profile": "coder",
                        "chats": chats,
                        "threads": ["thread-1"],
                    }
                ],
            }
        })


@pytest.mark.parametrize("profile", ["finance", "security", "production"])
@pytest.mark.parametrize("location", ["allowlist", "visible", "rule"])
def test_high_risk_profiles_are_denied_from_milestone_one_routing(profile, location):
    gateway = {"multiplex_profiles": True, "multiplex_profile_allowlist": ["coder"]}
    switching = {"enabled": True}
    if location == "allowlist":
        gateway["multiplex_profile_allowlist"] = [profile]
    elif location == "visible":
        switching["default_visible"] = [profile]
    else:
        switching["rules"] = [{"profile": profile, "users": ["user-1"]}]
    gateway["profile_switching"] = switching

    with pytest.raises(ValueError, match=f"{profile}.*Milestone 1"):
        GatewayConfig.from_dict({"gateway": gateway})


def test_high_risk_profile_may_only_be_named_hidden():
    cfg = GatewayConfig.from_dict({
        "gateway": {
            "multiplex_profiles": True,
            "multiplex_profile_allowlist": ["coder"],
            "profile_switching": {
                "enabled": True,
                "hidden": ["finance", "security", "production"],
            },
        }
    })
    assert cfg.profile_switching.hidden == ("finance", "security", "production")


def test_feature_off_preserves_legacy_sensitive_profile_multiplex_allowlist():
    cfg = GatewayConfig.from_dict({
        "gateway": {
            "multiplex_profiles": True,
            "multiplex_profile_allowlist": ["finance"],
        }
    })

    assert cfg.profile_switching.enabled is False
    assert cfg.multiplex_profile_allowlist == ["finance"]
    assert "profile_switching" not in cfg.to_dict()


@pytest.mark.parametrize(
    "raw",
    [
        {"profile_switching": {1: True}},
        {"profile_switching": {"rules": [{"profile": "coder", 1: True}]}},
    ],
)
def test_heterogeneous_unknown_keys_raise_value_error(raw):
    with pytest.raises(ValueError, match="unknown profile_switching"):
        GatewayConfig.from_dict(raw)


def test_unknown_rule_key_is_rejected():
    with pytest.raises(ValueError, match="unknown profile_switching rule keys"):
        GatewayConfig.from_dict({
            "profile_switching": {
                "rules": [{"profile": "coder", "user": [123]}],
            }
        })


def test_top_level_profile_switching_takes_precedence_over_nested():
    cfg = GatewayConfig.from_dict({
        "profile_switching": {"picker_ttl_seconds": 42},
        "gateway": {
            "profile_switching": {"picker_ttl_seconds": 99},
        },
    })
    assert cfg.profile_switching.picker_ttl_seconds == 42


def test_profile_switching_roundtrip_and_immutability():
    cfg = GatewayConfig.from_dict({
        "multiplex_profiles": True,
        "multiplex_profile_allowlist": ["coder"],
        "profile_switching": {
            "enabled": True,
            "picker_ttl_seconds": "42",
            "audit_retention_days": 7,
            "audit_max_rows": 500,
            "default_visible": ["Coder"],
            "admins": {"telegram": [123]},
            "rules": [{"profile": "Coder", "users": [123]}],
        },
    })
    restored = GatewayConfig.from_dict(cfg.to_dict())

    assert restored.profile_switching == cfg.profile_switching
    assert restored.profile_switching.default_visible == ("coder",)
    with pytest.raises(FrozenInstanceError):
        restored.profile_switching.enabled = True
