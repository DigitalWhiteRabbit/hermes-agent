from __future__ import annotations

from collections.abc import Iterable

from gateway.config import ProfileSwitchingConfig
from gateway.profile_switching.models import PolicyDecision, ReasonCode, ScopeKey


class ProfilePolicy:
    """Evaluate profile access from typed configuration without side effects."""

    def __init__(
        self,
        config: ProfileSwitchingConfig,
        *,
        served_profiles: Iterable[str],
        existing_profiles: Iterable[str],
    ) -> None:
        self._config = config
        self._served_profiles = frozenset(served_profiles)
        self._existing_profiles = frozenset(existing_profiles)
        self._rules_by_profile = {rule.profile: rule for rule in config.rules}

    def evaluate(
        self,
        profile_name: str,
        scope: ScopeKey,
        actor_user_id: str,
    ) -> PolicyDecision:
        if profile_name not in self._existing_profiles:
            return PolicyDecision(False, ReasonCode.PROFILE_UNKNOWN)
        if profile_name not in self._served_profiles:
            return PolicyDecision(False, ReasonCode.PROFILE_UNSERVED)
        if profile_name in self._config.hidden:
            return PolicyDecision(False, ReasonCode.PROFILE_HIDDEN)
        rule = self._rules_by_profile.get(profile_name)
        if rule is None:
            return PolicyDecision(False, ReasonCode.USER_DENIED)
        if actor_user_id not in rule.users:
            return PolicyDecision(False, ReasonCode.USER_DENIED)
        if "*" not in rule.chats and scope.chat_id not in rule.chats:
            return PolicyDecision(False, ReasonCode.CHAT_DENIED)
        if rule.threads and (scope.thread_id or "") not in rule.threads:
            return PolicyDecision(False, ReasonCode.THREAD_DENIED)
        return PolicyDecision(True, ReasonCode.ALLOWED)

    def visible_profiles(
        self,
        scope: ScopeKey,
        actor_user_id: str,
    ) -> tuple[str, ...]:
        return tuple(
            profile_name
            for profile_name in self._config.default_visible
            if self.evaluate(profile_name, scope, actor_user_id).allowed
        )
