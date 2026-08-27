from __future__ import annotations

from pathlib import Path

from hermes_cli.profiles import normalize_profile_name, validate_profile_name

from .models import (
    ProfileBinding,
    ProfileResolution,
    ReasonCode,
    ScopeKey,
    ScopeKind,
)
from .policy import ProfilePolicy
from .resolver import ProfileResolver
from .store import ProfileBindingChanged, ProfileRoutingStore


_CLEAR_RETRY_LIMIT = 3


class ProfileSwitchError(RuntimeError):
    """Base class for rejected profile-routing mutations."""


class ProfileSwitchDenied(ProfileSwitchError):
    def __init__(self, reason: ReasonCode) -> None:
        self.reason = reason
        super().__init__(f"profile switch denied: {reason.value}")


class ProfileSwitchBusy(ProfileSwitchError):
    def __init__(self) -> None:
        self.reason = ReasonCode.ACTIVE_TURN
        super().__init__("profile switch denied while a turn is active")


class InvalidProfileScope(ProfileSwitchError):
    def __init__(self) -> None:
        self.reason = ReasonCode.THREAD_DENIED
        super().__init__("thread profile scope requires a thread id")


class ProfileSwitchingService:
    """Validate, authorize, and persist profile-routing operations."""

    def __init__(
        self,
        store: ProfileRoutingStore,
        policy: ProfilePolicy,
    ) -> None:
        self._store = store
        self._policy = policy
        self._resolver = ProfileResolver(store, policy)

    @classmethod
    def from_gateway_config(
        cls,
        config,
        *,
        db_path: Path,
    ) -> "ProfileSwitchingService":
        """Build the routing service from the gateway's authoritative profiles."""
        from gateway.run import _multiplex_profile_homes
        from hermes_cli.profiles import list_profile_names, profile_exists

        existing_profiles = {
            name for name in list_profile_names() if profile_exists(name)
        }
        existing_profiles.add("default")
        served_profiles = {
            name for name, _home in _multiplex_profile_homes(config)
        }
        served_profiles.add("default")
        store = ProfileRoutingStore(Path(db_path))
        policy = ProfilePolicy(
            config.profile_switching,
            served_profiles=served_profiles,
            existing_profiles=existing_profiles,
        )
        return cls(store, policy)

    @staticmethod
    def _profile_name(profile_name: str) -> str:
        normalized = normalize_profile_name(profile_name)
        validate_profile_name(normalized)
        return normalized

    def _audit(
        self,
        *,
        actor_user_id: str,
        scope: ScopeKey,
        old_profile: str | None,
        new_profile: str | None,
        scope_kind: ScopeKind | None,
        source: str,
        result: str,
        reason_code: ReasonCode,
    ) -> None:
        self._store.append_audit(
            actor_user_id=actor_user_id,
            scope=scope,
            old_profile=old_profile,
            new_profile=new_profile,
            scope_kind=scope_kind,
            source=source,
            result=result,
            reason_code=reason_code,
        )

    def _require_scope(
        self,
        *,
        scope: ScopeKey,
        scope_kind: ScopeKind,
        actor_user_id: str,
        new_profile: str | None,
        source: str,
    ) -> None:
        if scope_kind is not ScopeKind.THREAD or scope.thread_id:
            return
        self._audit(
            actor_user_id=actor_user_id,
            scope=scope,
            old_profile=None,
            new_profile=new_profile,
            scope_kind=scope_kind,
            source=source,
            result="denied",
            reason_code=ReasonCode.THREAD_DENIED,
        )
        raise InvalidProfileScope

    def _authorize(
        self,
        *,
        profile_name: str,
        scope: ScopeKey,
        scope_kind: ScopeKind,
        actor_user_id: str,
        source: str,
        old_profile: str | None,
        new_profile: str | None,
    ) -> None:
        decision = self._policy.evaluate(profile_name, scope, actor_user_id)
        if decision.allowed:
            return
        self._audit(
            actor_user_id=actor_user_id,
            scope=scope,
            old_profile=old_profile,
            new_profile=new_profile,
            scope_kind=scope_kind,
            source=source,
            result="denied",
            reason_code=decision.reason,
        )
        raise ProfileSwitchDenied(decision.reason)

    def _require_idle(
        self,
        *,
        active_turn: bool,
        scope: ScopeKey,
        scope_kind: ScopeKind,
        actor_user_id: str,
        source: str,
        new_profile: str | None,
    ) -> None:
        if not active_turn:
            return
        self._audit(
            actor_user_id=actor_user_id,
            scope=scope,
            old_profile=None,
            new_profile=new_profile,
            scope_kind=scope_kind,
            source=source,
            result="denied",
            reason_code=ReasonCode.ACTIVE_TURN,
        )
        raise ProfileSwitchBusy

    def set_profile(
        self,
        *,
        scope: ScopeKey,
        scope_kind: ScopeKind,
        actor_user_id: str,
        profile_name: str,
        active_turn: bool,
    ) -> ProfileBinding:
        profile_name = self._profile_name(profile_name)
        scope_kind = ScopeKind(scope_kind)
        self._require_scope(
            scope=scope,
            scope_kind=scope_kind,
            actor_user_id=actor_user_id,
            new_profile=profile_name,
            source="command",
        )
        self._authorize(
            profile_name=profile_name,
            scope=scope,
            scope_kind=scope_kind,
            actor_user_id=actor_user_id,
            source="command",
            old_profile=None,
            new_profile=profile_name,
        )
        self._require_idle(
            active_turn=active_turn,
            scope=scope,
            scope_kind=scope_kind,
            actor_user_id=actor_user_id,
            source="command",
            new_profile=profile_name,
        )
        return self._store.set_binding_with_audit(
            scope,
            scope_kind,
            profile_name=profile_name,
            created_by_user_id=actor_user_id,
            source="command",
        )

    def set_once(
        self,
        *,
        scope: ScopeKey,
        actor_user_id: str,
        profile_name: str,
        active_turn: bool,
    ) -> ProfileBinding:
        profile_name = self._profile_name(profile_name)
        scope_kind = ScopeKind.THREAD if scope.thread_id else ScopeKind.CHAT
        self._authorize(
            profile_name=profile_name,
            scope=scope,
            scope_kind=scope_kind,
            actor_user_id=actor_user_id,
            source="once",
            old_profile=None,
            new_profile=profile_name,
        )
        self._require_idle(
            active_turn=active_turn,
            scope=scope,
            scope_kind=scope_kind,
            actor_user_id=actor_user_id,
            source="once",
            new_profile=profile_name,
        )
        return self._store.set_once_with_audit(
            scope,
            profile_name=profile_name,
            created_by_user_id=actor_user_id,
            source="once",
        )

    def clear(
        self,
        *,
        scope: ScopeKey,
        scope_kind: ScopeKind,
        actor_user_id: str,
        active_turn: bool,
    ) -> bool:
        scope_kind = ScopeKind(scope_kind)
        self._require_scope(
            scope=scope,
            scope_kind=scope_kind,
            actor_user_id=actor_user_id,
            new_profile=None,
            source="clear",
        )
        self._require_idle(
            active_turn=active_turn,
            scope=scope,
            scope_kind=scope_kind,
            actor_user_id=actor_user_id,
            source="clear",
            new_profile=None,
        )
        for _attempt in range(_CLEAR_RETRY_LIMIT):
            old_binding = self._store.get_binding(scope, scope_kind)
            if old_binding is not None:
                self._authorize(
                    profile_name=old_binding.profile_name,
                    scope=scope,
                    scope_kind=scope_kind,
                    actor_user_id=actor_user_id,
                    source="clear",
                    old_profile=old_binding.profile_name,
                    new_profile=None,
                )
            try:
                return self._store.clear_binding_with_audit(
                    scope,
                    scope_kind,
                    expected_version=(old_binding.version if old_binding else None),
                    expected_profile=(
                        old_binding.profile_name if old_binding else None
                    ),
                    actor_user_id=actor_user_id,
                    source="clear",
                )
            except ProfileBindingChanged:
                continue
        raise ProfileSwitchError(
            "profile binding kept changing during authorized clear"
        )

    def resolve(
        self,
        *,
        scope: ScopeKey,
        actor_user_id: str,
        consume_once: bool,
        static_profile: str | None,
    ) -> ProfileResolution:
        return self._resolver.resolve(
            scope,
            actor_user_id=actor_user_id,
            consume_once=consume_once,
            static_profile=static_profile,
        )

    def note_session(
        self,
        *,
        scope: ScopeKey,
        profile_name: str,
        session_id: str,
    ) -> None:
        profile_name = self._profile_name(profile_name)
        self._store.record_session(scope, profile_name, session_id)
