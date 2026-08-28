from gateway.config import ProfileSwitchRule, ProfileSwitchingConfig
from gateway.profile_switching.models import ReasonCode, ScopeKey
from gateway.profile_switching.policy import ProfilePolicy


def make_scope(*, chat_id: str = "chat-1", thread_id: str | None = None) -> ScopeKey:
    return ScopeKey("telegram", "telegram:primary", chat_id, thread_id)


def make_config(**kwargs) -> ProfileSwitchingConfig:
    return ProfileSwitchingConfig(**kwargs)


def test_default_visible_allowed_for_matching_user_and_wildcard_chat():
    policy = ProfilePolicy(
        make_config(
            default_visible=("coder",),
            rules=(ProfileSwitchRule("coder", users=("user-1",), chats=("*",)),),
        ),
        served_profiles={"coder"},
        existing_profiles={"coder"},
    )

    assert policy.evaluate("coder", make_scope(), "user-1").reason is ReasonCode.ALLOWED


def test_unknown_profile_denied():
    policy = ProfilePolicy(
        make_config(rules=(ProfileSwitchRule("coder", users=("user-1",), chats=("*",)),)),
        served_profiles={"coder", "unknown"},
        existing_profiles={"coder"},
    )

    assert policy.evaluate("unknown", make_scope(), "user-1").reason is ReasonCode.PROFILE_UNKNOWN


def test_unserved_profile_denied():
    policy = ProfilePolicy(
        make_config(rules=(ProfileSwitchRule("coder", users=("user-1",), chats=("*",)),)),
        served_profiles=set(),
        existing_profiles={"coder"},
    )

    assert policy.evaluate("coder", make_scope(), "user-1").reason is ReasonCode.PROFILE_UNSERVED


def test_hidden_profile_denied_even_when_it_exists():
    policy = ProfilePolicy(
        make_config(
            hidden=("coder",),
            rules=(ProfileSwitchRule("coder", users=("user-1",), chats=("*",)),),
        ),
        served_profiles={"coder"},
        existing_profiles={"coder"},
    )

    assert policy.evaluate("coder", make_scope(), "user-1").reason is ReasonCode.PROFILE_HIDDEN


def test_wrong_user_denied():
    policy = ProfilePolicy(
        make_config(rules=(ProfileSwitchRule("coder", users=("user-1",), chats=("*",)),)),
        served_profiles={"coder"},
        existing_profiles={"coder"},
    )

    assert policy.evaluate("coder", make_scope(), "user-2").reason is ReasonCode.USER_DENIED


def test_wrong_chat_denied():
    policy = ProfilePolicy(
        make_config(rules=(ProfileSwitchRule("coder", users=("user-1",), chats=("chat-1",)),)),
        served_profiles={"coder"},
        existing_profiles={"coder"},
    )

    assert policy.evaluate("coder", make_scope(chat_id="chat-2"), "user-1").reason is ReasonCode.CHAT_DENIED


def test_thread_rule_requires_matching_thread():
    policy = ProfilePolicy(
        make_config(
            rules=(
                ProfileSwitchRule(
                    "coder",
                    users=("user-1",),
                    chats=("*",),
                    threads=("thread-1",),
                ),
            ),
        ),
        served_profiles={"coder"},
        existing_profiles={"coder"},
    )

    assert policy.evaluate("coder", make_scope(thread_id="thread-2"), "user-1").reason is ReasonCode.THREAD_DENIED
    assert policy.evaluate("coder", make_scope(thread_id="thread-1"), "user-1").reason is ReasonCode.ALLOWED


def test_telegram_admin_does_not_bypass_profile_rule():
    policy = ProfilePolicy(
        make_config(
            admins={"telegram": ("admin-1",)},
            rules=(ProfileSwitchRule("coder", users=("user-1",), chats=("*",)),),
        ),
        served_profiles={"coder"},
        existing_profiles={"coder"},
    )

    assert policy.evaluate("coder", make_scope(), "admin-1").reason is ReasonCode.USER_DENIED


def test_visible_profiles_filters_denied_and_preserves_config_order():
    policy = ProfilePolicy(
        make_config(
            default_visible=("writer", "coder", "hidden", "private", "unserved"),
            hidden=("hidden",),
            rules=(
                ProfileSwitchRule("writer", users=("user-1",), chats=("*",)),
                ProfileSwitchRule("coder", users=("user-1",), chats=("chat-1",)),
                ProfileSwitchRule("hidden", users=("user-1",), chats=("*",)),
                ProfileSwitchRule("private", users=("user-2",), chats=("*",)),
                ProfileSwitchRule("unserved", users=("user-1",), chats=("*",)),
            ),
        ),
        served_profiles={"writer", "coder", "hidden", "private"},
        existing_profiles={"writer", "coder", "hidden", "private", "unserved"},
    )

    assert policy.visible_profiles(make_scope(), "user-1") == ("writer", "coder")
