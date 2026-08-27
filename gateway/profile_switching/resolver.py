from __future__ import annotations

from collections.abc import Callable

from .models import (
    ProfileBinding,
    ProfileResolution,
    ReasonCode,
    ResolutionSource,
    ScopeKey,
    ScopeKind,
)
from .policy import ProfilePolicy
from .store import ProfileRoutingStore, ProfileRoutingStoreUnavailable


class ProfileResolver:
    """Resolve the highest-priority profile candidate allowed by policy."""

    def __init__(
        self,
        store: ProfileRoutingStore,
        policy: ProfilePolicy,
    ) -> None:
        self._store = store
        self._policy = policy

    def resolve(
        self,
        scope: ScopeKey,
        *,
        actor_user_id: str,
        consume_once: bool,
        static_profile: str | None,
        on_diagnostic: Callable[[ResolutionSource | None, str | None, ReasonCode], None]
        | None = None,
    ) -> ProfileResolution:
        def emit(
            source: ResolutionSource | None,
            profile_name: str | None,
            reason: ReasonCode,
        ) -> None:
            if on_diagnostic is None:
                return
            try:
                on_diagnostic(source, profile_name, reason)
            except Exception:
                # Observability is never part of the authorization decision.
                pass

        candidates: list[
            tuple[ResolutionSource, ProfileBinding | None, str | None]
        ] = []
        db_unavailable = False
        try:
            if consume_once:
                once = self._store.claim_once(scope)
                if once is not None:
                    candidates.append((ResolutionSource.ONCE, once, once.profile_name))
            thread = self._store.get_binding(scope, ScopeKind.THREAD)
            chat = self._store.get_binding(
                scope.for_kind(ScopeKind.CHAT), ScopeKind.CHAT
            )
            candidates.extend([
                (
                    ResolutionSource.THREAD,
                    thread,
                    thread.profile_name if thread else None,
                ),
                (
                    ResolutionSource.CHAT,
                    chat,
                    chat.profile_name if chat else None,
                ),
            ])
        except ProfileRoutingStoreUnavailable:
            candidates = []
            db_unavailable = True
            emit(None, None, ReasonCode.DB_UNAVAILABLE)

        candidates.append((ResolutionSource.STATIC, None, static_profile))
        for source, binding, profile_name in candidates:
            if not profile_name:
                continue
            decision = self._policy.evaluate(profile_name, scope, actor_user_id)
            if decision.allowed:
                return ProfileResolution(
                    profile_name,
                    source,
                    (ReasonCode.DB_UNAVAILABLE if db_unavailable else decision.reason),
                    binding,
                )
            emit(source, profile_name, decision.reason)
        return ProfileResolution(
            None,
            ResolutionSource.DEFAULT,
            (ReasonCode.DB_UNAVAILABLE if db_unavailable else ReasonCode.NO_MATCH),
        )
