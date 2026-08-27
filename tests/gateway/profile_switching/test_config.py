from dataclasses import FrozenInstanceError

import pytest

from gateway.config import GatewayConfig


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
