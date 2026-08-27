from dataclasses import FrozenInstanceError

import pytest

from gateway.config import GatewayConfig


def test_profile_switching_defaults_disabled():
    cfg = GatewayConfig.from_dict({})
    assert cfg.profile_switching.enabled is False
    assert cfg.profile_switching.picker_ttl_seconds == 300


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
            "rules": [{
                "profile": "coder",
                "users": [123],
                "chats": [-100456],
                "threads": [789],
                "require_confirmation": True,
            }],
        }
    })
    assert cfg.profile_switching.admins == {"telegram": ("123", "456")}
    assert cfg.profile_switching.rules[0].users == ("123",)
    assert cfg.profile_switching.rules[0].chats == ("-100456",)
    assert cfg.profile_switching.rules[0].threads == ("789",)


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
        "profile_switching": {
            "picker_ttl_seconds": "42",
            "audit_retention_days": 7,
            "audit_max_rows": 500,
            "default_visible": ["Coder"],
            "admins": {"telegram": [123]},
            "rules": [{"profile": "Coder", "users": [123]}],
        }
    })
    restored = GatewayConfig.from_dict(cfg.to_dict())

    assert restored.profile_switching == cfg.profile_switching
    assert restored.profile_switching.default_visible == ("coder",)
    with pytest.raises(FrozenInstanceError):
        restored.profile_switching.enabled = True
