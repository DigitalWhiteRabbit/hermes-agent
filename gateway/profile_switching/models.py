from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ScopeKind(str, Enum):
    CHAT = "chat"
    THREAD = "thread"


class ResolutionSource(str, Enum):
    ONCE = "once"
    THREAD = "thread"
    CHAT = "chat"
    STATIC = "static"
    DEFAULT = "default"


class ReasonCode(str, Enum):
    ALLOWED = "allowed"
    FEATURE_DISABLED = "feature_disabled"
    MULTIPLEX_DISABLED = "multiplex_disabled"
    PROFILE_UNKNOWN = "profile_unknown"
    PROFILE_UNSERVED = "profile_unserved"
    PROFILE_HIDDEN = "profile_hidden"
    USER_DENIED = "user_denied"
    CHAT_DENIED = "chat_denied"
    THREAD_DENIED = "thread_denied"
    ACTIVE_TURN = "active_turn"
    DB_UNAVAILABLE = "db_unavailable"
    NO_MATCH = "no_match"


@dataclass(frozen=True)
class ScopeKey:
    platform: str
    account_id: str
    chat_id: str
    thread_id: Optional[str]

    @classmethod
    def from_source(cls, source, *, account_id: str) -> "ScopeKey":
        platform = getattr(getattr(source, "platform", None), "value", None)
        return cls(
            platform=str(platform or ""),
            account_id=str(account_id),
            chat_id=str(getattr(source, "chat_id", "")),
            thread_id=(
                str(source.thread_id)
                if getattr(source, "thread_id", None) not in (None, "")
                else None
            ),
        )

    def for_kind(self, kind: ScopeKind) -> "ScopeKey":
        return self if kind is ScopeKind.THREAD else ScopeKey(
            self.platform, self.account_id, self.chat_id, None
        )


@dataclass(frozen=True)
class ProfileBinding:
    scope: ScopeKey
    scope_kind: ScopeKind
    profile_name: str
    created_by_user_id: str
    created_at: float
    updated_at: float
    expires_at: Optional[float]
    version: int


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: ReasonCode


@dataclass(frozen=True)
class ProfileResolution:
    profile_name: Optional[str]
    source: ResolutionSource
    reason: ReasonCode
    binding: Optional[ProfileBinding] = None
