from dataclasses import FrozenInstanceError

import pytest

from gateway.config import Platform
from gateway.profile_switching.models import (
    ProfileBinding,
    ResolutionSource,
    ScopeKey,
    ScopeKind,
)
from gateway.session import SessionSource


def test_scope_key_normalizes_source_identifiers_to_strings():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=-100123,
        thread_id=42,
        user_id=7,
    )
    scope = ScopeKey.from_source(source, account_id="primary")
    assert scope == ScopeKey(
        platform="telegram",
        account_id="primary",
        chat_id="-100123",
        thread_id="42",
    )


def test_scope_key_chat_projection_removes_thread():
    scope = ScopeKey("telegram", "primary", "-100123", "42")
    assert scope.for_kind(ScopeKind.CHAT).thread_id is None


def test_profile_binding_is_immutable():
    binding = ProfileBinding(
        scope=ScopeKey("telegram", "primary", "1", None),
        scope_kind=ScopeKind.CHAT,
        profile_name="coder",
        created_by_user_id="7",
        created_at=1.0,
        updated_at=1.0,
        expires_at=None,
        version=1,
    )
    with pytest.raises(FrozenInstanceError):
        binding.profile_name = "research"


def test_resolution_source_values_are_stable():
    assert [item.value for item in ResolutionSource] == [
        "once", "thread", "chat", "static", "default"
    ]
