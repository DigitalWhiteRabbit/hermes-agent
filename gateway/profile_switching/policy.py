from __future__ import annotations

from collections.abc import Callable, Iterable

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
        profile_exists_now: Callable[[str], bool] | None = None,
        profile_served_now: Callable[[str], bool] | None = None,
    ) -> None:
        self._config = config
        self._served_profiles = frozenset(served_profiles)
        self._existing_profiles = frozenset(existing_profiles)
        self._profile_exists_now = profile_exists_now
        self._profile_served_now = profile_served_now
        self._rules_by_profile = {rule.profile: rule for rule in config.rules}

    @staticmethod
    def _current(check: Callable[[str], bool] | None, profile_name: str) -> bool:
        if check is None:
            return True
        try:
            return check(profile_name) is True
        except Exception:
            return False

    def evaluate(
        self,
        profile_name: str,
        scope: ScopeKey,
        actor_user_id: str,
    ) -> PolicyDecision:
        if profile_name not in self._existing_profiles or not self._current(
            self._profile_exists_now, profile_name
        ):
            return PolicyDecision(False, ReasonCode.PROFILE_UNKNOWN)
        if profile_name not in self._served_profiles or not self._current(
            self._profile_served_now, profile_name
        ):
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
