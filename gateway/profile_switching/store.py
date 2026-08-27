from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .migrations import run_migrations
from .models import ProfileBinding, ReasonCode, ScopeKey, ScopeKind


logger = logging.getLogger(__name__)
_SQLITE_HEADER = b"SQLite format 3\x00"
_BUSY_TIMEOUT_MS = 250


class ProfileRoutingStoreUnavailable(RuntimeError):
    """The profile routing database could not complete an operation."""


class ProfileBindingChanged(RuntimeError):
    """The binding changed after it was read for authorization."""


class ProfileRoutingStore:
    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path)
        self._clock = clock
        self.quarantined_paths: tuple[Path, ...] = ()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self._has_invalid_sqlite_header():
                self.quarantined_paths = self._quarantine_corrupt_files()
            self._initialize_schema()
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            if self._is_corruption_error(exc) and self.path.exists():
                try:
                    self.quarantined_paths = self._quarantine_corrupt_files()
                    self._initialize_schema()
                    logger.error(
                        "Quarantined corrupt profile routing database and "
                        "initialized a replacement: %s",
                        self.path,
                    )
                    return
                except (OSError, sqlite3.Error, TypeError, ValueError) as retry_exc:
                    exc = retry_exc
            raise ProfileRoutingStoreUnavailable(
                f"profile routing database is unavailable: {self.path}"
            ) from exc

    def _initialize_schema(self) -> None:
        with self._connection() as conn:
            run_migrations(conn)

    def _has_invalid_sqlite_header(self) -> bool:
        try:
            if not self.path.is_file() or self.path.stat().st_size == 0:
                return False
            with self.path.open("rb") as handle:
                return handle.read(len(_SQLITE_HEADER)) != _SQLITE_HEADER
        except OSError:
            return False

    @staticmethod
    def _is_corruption_error(exc: BaseException) -> bool:
        if isinstance(exc, (TypeError, ValueError)):
            # During initialization these arise only while decoding persisted
            # schema metadata (for example a non-integer schema version).
            return True
        code = getattr(exc, "sqlite_errorcode", None)
        corrupt_codes = {
            getattr(sqlite3, "SQLITE_CORRUPT", -1),
            getattr(sqlite3, "SQLITE_NOTADB", -2),
        }
        if code in corrupt_codes:
            return True
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "database disk image is malformed",
                "file is not a database",
                "malformed database schema",
            )
        )

    def _quarantine_corrupt_files(self) -> tuple[Path, ...]:
        """Recoverably rename the DB and sidecars; never delete them."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        moved: list[Path] = []
        for candidate in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        ):
            if not candidate.exists():
                continue
            destination = candidate.with_name(f"{candidate.name}.corrupt-{stamp}")
            suffix = 1
            while destination.exists():
                destination = candidate.with_name(
                    f"{candidate.name}.corrupt-{stamp}-{suffix}"
                )
                suffix += 1
            candidate.replace(destination)
            moved.append(destination)
        if moved:
            logger.error(
                "Quarantined corrupt profile routing artifacts: %s",
                ", ".join(str(item) for item in moved),
            )
        return tuple(moved)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            timeout=_BUSY_TIMEOUT_MS / 1000,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
            return conn
        except BaseException:
            conn.close()
            raise

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect()
            yield conn
        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    def _sql_thread_id(thread_id: str | None) -> str:
        return thread_id or ""

    @staticmethod
    def _domain_thread_id(thread_id: str) -> str | None:
        return thread_id or None

    @staticmethod
    def _binding_scope(scope: ScopeKey, scope_kind: ScopeKind) -> ScopeKey:
        return (
            scope if scope_kind is ScopeKind.THREAD else scope.for_kind(ScopeKind.CHAT)
        )

    @classmethod
    def _binding_from_row(
        cls,
        row: sqlite3.Row,
        *,
        scope_kind: ScopeKind | None = None,
    ) -> ProfileBinding:
        resolved_kind = scope_kind or ScopeKind(row["scope_kind"])
        created_at = float(row["created_at"])
        return ProfileBinding(
            scope=ScopeKey(
                platform=row["platform"],
                account_id=row["account_id"],
                chat_id=row["chat_id"],
                thread_id=cls._domain_thread_id(row["thread_id"]),
            ),
            scope_kind=resolved_kind,
            profile_name=row["profile_name"],
            created_by_user_id=row["created_by_user_id"],
            created_at=created_at,
            updated_at=(
                float(row["updated_at"]) if "updated_at" in row.keys() else created_at
            ),
            expires_at=(
                float(row["expires_at"]) if row["expires_at"] is not None else None
            ),
            version=int(row["version"]) if "version" in row.keys() else 1,
        )

    def _decode_binding(
        self,
        row: sqlite3.Row,
        *,
        scope_kind: ScopeKind | None = None,
    ) -> ProfileBinding:
        try:
            return self._binding_from_row(row, scope_kind=scope_kind)
        except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
            raise self._unavailable(exc) from exc

    @contextmanager
    def _immediate(self, conn: sqlite3.Connection) -> Iterator[None]:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    def get_binding(
        self,
        scope: ScopeKey,
        scope_kind: ScopeKind,
    ) -> ProfileBinding | None:
        scope_kind = ScopeKind(scope_kind)
        key = self._binding_scope(scope, scope_kind)
        try:
            with self._connection() as conn:
                row = self._select_binding(conn, key, scope_kind)
        except sqlite3.Error as exc:
            raise self._unavailable(exc) from exc
        if row is None:
            return None
        binding = self._decode_binding(row)
        if binding.expires_at is not None and binding.expires_at <= self._clock():
            return None
        return binding

    def set_binding(
        self,
        scope: ScopeKey,
        scope_kind: ScopeKind,
        *,
        profile_name: str,
        created_by_user_id: str,
        expires_at: float | None = None,
    ) -> ProfileBinding:
        scope_kind = ScopeKind(scope_kind)
        key = self._binding_scope(scope, scope_kind)
        now = self._clock()
        try:
            with self._connection() as conn, self._immediate(conn):
                row = self._upsert_binding(
                    conn,
                    key,
                    scope_kind,
                    profile_name=profile_name,
                    created_by_user_id=created_by_user_id,
                    now=now,
                    expires_at=expires_at,
                )
        except sqlite3.Error as exc:
            raise self._unavailable(exc) from exc
        return self._decode_binding(row)

    def set_binding_with_audit(
        self,
        scope: ScopeKey,
        scope_kind: ScopeKind,
        *,
        profile_name: str,
        created_by_user_id: str,
        source: str,
        expires_at: float | None = None,
    ) -> ProfileBinding:
        scope_kind = ScopeKind(scope_kind)
        key = self._binding_scope(scope, scope_kind)
        now = self._clock()
        try:
            with self._connection() as conn, self._immediate(conn):
                old_row = self._select_binding(conn, key, scope_kind)
                row = self._upsert_binding(
                    conn,
                    key,
                    scope_kind,
                    profile_name=profile_name,
                    created_by_user_id=created_by_user_id,
                    now=now,
                    expires_at=expires_at,
                )
                self._insert_audit(
                    conn,
                    actor_user_id=created_by_user_id,
                    scope=scope,
                    old_profile=(old_row["profile_name"] if old_row else None),
                    new_profile=profile_name,
                    scope_kind=scope_kind,
                    source=source,
                    result="allowed",
                    reason_code=ReasonCode.ALLOWED,
                )
        except sqlite3.Error as exc:
            raise self._unavailable(exc) from exc
        return self._decode_binding(row)

    def _select_binding(
        self,
        conn: sqlite3.Connection,
        key: ScopeKey,
        scope_kind: ScopeKind,
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT * FROM profile_bindings
            WHERE platform = ? AND account_id = ? AND chat_id = ?
              AND scope_kind = ? AND thread_id = ?
            """,
            (
                key.platform,
                key.account_id,
                key.chat_id,
                scope_kind.value,
                self._sql_thread_id(key.thread_id),
            ),
        ).fetchone()

    def _upsert_binding(
        self,
        conn: sqlite3.Connection,
        key: ScopeKey,
        scope_kind: ScopeKind,
        *,
        profile_name: str,
        created_by_user_id: str,
        now: float,
        expires_at: float | None,
    ) -> sqlite3.Row:
        conn.execute(
            """
            INSERT INTO profile_bindings (
                platform, account_id, chat_id, thread_id, scope_kind,
                profile_name, created_by_user_id, created_at, updated_at,
                expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(platform, account_id, chat_id, scope_kind, thread_id)
            DO UPDATE SET
                profile_name = excluded.profile_name,
                created_by_user_id = excluded.created_by_user_id,
                updated_at = excluded.updated_at,
                expires_at = excluded.expires_at,
                version = profile_bindings.version + 1
            """,
            (
                key.platform,
                key.account_id,
                key.chat_id,
                self._sql_thread_id(key.thread_id),
                scope_kind.value,
                profile_name,
                created_by_user_id,
                now,
                now,
                expires_at,
            ),
        )
        row = self._select_binding(conn, key, scope_kind)
        assert row is not None
        return row

    def clear_binding(self, scope: ScopeKey, scope_kind: ScopeKind) -> bool:
        scope_kind = ScopeKind(scope_kind)
        key = self._binding_scope(scope, scope_kind)
        try:
            with self._connection() as conn, self._immediate(conn):
                cursor = conn.execute(
                    """
                    DELETE FROM profile_bindings
                    WHERE platform = ? AND account_id = ? AND chat_id = ?
                      AND scope_kind = ? AND thread_id = ?
                    """,
                    (
                        key.platform,
                        key.account_id,
                        key.chat_id,
                        scope_kind.value,
                        self._sql_thread_id(key.thread_id),
                    ),
                )
                removed = cursor.rowcount
        except sqlite3.Error as exc:
            raise self._unavailable(exc) from exc
        return removed > 0

    def clear_binding_with_audit(
        self,
        scope: ScopeKey,
        scope_kind: ScopeKind,
        *,
        expected_version: int | None,
        expected_profile: str | None,
        actor_user_id: str,
        source: str,
    ) -> bool:
        scope_kind = ScopeKind(scope_kind)
        key = self._binding_scope(scope, scope_kind)
        try:
            with self._connection() as conn, self._immediate(conn):
                row = self._select_binding(conn, key, scope_kind)
                expired = (
                    row is not None
                    and row["expires_at"] is not None
                    and float(row["expires_at"]) <= self._clock()
                )
                current_version = (
                    int(row["version"]) if row is not None and not expired else None
                )
                current_profile = (
                    str(row["profile_name"])
                    if row is not None and not expired
                    else None
                )
                if (current_version, current_profile) != (
                    expected_version,
                    expected_profile,
                ):
                    raise ProfileBindingChanged(
                        "profile binding changed during authorized clear"
                    )
                removed = False
                if row is not None:
                    cursor = conn.execute(
                        """
                        DELETE FROM profile_bindings
                        WHERE platform = ? AND account_id = ? AND chat_id = ?
                          AND scope_kind = ? AND thread_id = ? AND version = ?
                        """,
                        (
                            key.platform,
                            key.account_id,
                            key.chat_id,
                            scope_kind.value,
                            self._sql_thread_id(key.thread_id),
                            int(row["version"]),
                        ),
                    )
                    removed = cursor.rowcount > 0
                    if not removed:
                        raise ProfileBindingChanged(
                            "profile binding changed during authorized clear"
                        )
                self._insert_audit(
                    conn,
                    actor_user_id=actor_user_id,
                    scope=scope,
                    old_profile=(row["profile_name"] if row else None),
                    new_profile=None,
                    scope_kind=scope_kind,
                    source=source,
                    result="allowed",
                    reason_code=ReasonCode.ALLOWED,
                )
        except sqlite3.Error as exc:
            raise self._unavailable(exc) from exc
        return removed

    def set_once(
        self,
        scope: ScopeKey,
        *,
        profile_name: str,
        created_by_user_id: str,
        expires_at: float | None = None,
    ) -> ProfileBinding:
        now = self._clock()
        try:
            with self._connection() as conn, self._immediate(conn):
                row = self._upsert_once(
                    conn,
                    scope,
                    profile_name=profile_name,
                    created_by_user_id=created_by_user_id,
                    now=now,
                    expires_at=expires_at,
                )
        except sqlite3.Error as exc:
            raise self._unavailable(exc) from exc
        return self._decode_binding(row, scope_kind=self._once_scope_kind(scope))

    def set_once_with_audit(
        self,
        scope: ScopeKey,
        *,
        profile_name: str,
        created_by_user_id: str,
        source: str,
        expires_at: float | None = None,
    ) -> ProfileBinding:
        now = self._clock()
        scope_kind = self._once_scope_kind(scope)
        try:
            with self._connection() as conn, self._immediate(conn):
                old_row = self._select_once(conn, scope)
                row = self._upsert_once(
                    conn,
                    scope,
                    profile_name=profile_name,
                    created_by_user_id=created_by_user_id,
                    now=now,
                    expires_at=expires_at,
                )
                self._insert_audit(
                    conn,
                    actor_user_id=created_by_user_id,
                    scope=scope,
                    old_profile=(old_row["profile_name"] if old_row else None),
                    new_profile=profile_name,
                    scope_kind=scope_kind,
                    source=source,
                    result="allowed",
                    reason_code=ReasonCode.ALLOWED,
                )
        except sqlite3.Error as exc:
            raise self._unavailable(exc) from exc
        return self._decode_binding(row, scope_kind=scope_kind)

    def _upsert_once(
        self,
        conn: sqlite3.Connection,
        scope: ScopeKey,
        *,
        profile_name: str,
        created_by_user_id: str,
        now: float,
        expires_at: float | None,
    ) -> sqlite3.Row:
        conn.execute(
            """
            INSERT INTO profile_once (
                platform, account_id, chat_id, thread_id, profile_name,
                created_by_user_id, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(platform, account_id, chat_id, thread_id)
            DO UPDATE SET
                profile_name = excluded.profile_name,
                created_by_user_id = excluded.created_by_user_id,
                created_at = excluded.created_at,
                expires_at = excluded.expires_at
            """,
            (
                scope.platform,
                scope.account_id,
                scope.chat_id,
                self._sql_thread_id(scope.thread_id),
                profile_name,
                created_by_user_id,
                now,
                expires_at,
            ),
        )
        row = self._select_once(conn, scope)
        assert row is not None
        return row

    def claim_once(self, scope: ScopeKey) -> ProfileBinding | None:
        try:
            with self._connection() as conn, self._immediate(conn):
                row = self._select_once(conn, scope)
                conn.execute(
                    """
                    DELETE FROM profile_once
                    WHERE platform = ? AND account_id = ? AND chat_id = ?
                      AND thread_id = ?
                    """,
                    self._scope_values(scope),
                )
        except sqlite3.Error as exc:
            raise self._unavailable(exc) from exc
        if row is None:
            return None
        binding = self._decode_binding(row, scope_kind=self._once_scope_kind(scope))
        if binding.expires_at is not None and binding.expires_at <= self._clock():
            return None
        return binding

    def _select_once(
        self,
        conn: sqlite3.Connection,
        scope: ScopeKey,
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT * FROM profile_once
            WHERE platform = ? AND account_id = ? AND chat_id = ?
              AND thread_id = ?
            """,
            self._scope_values(scope),
        ).fetchone()

    @staticmethod
    def _once_scope_kind(scope: ScopeKey) -> ScopeKind:
        return ScopeKind.THREAD if scope.thread_id else ScopeKind.CHAT

    def _scope_values(self, scope: ScopeKey) -> tuple[str, str, str, str]:
        return (
            scope.platform,
            scope.account_id,
            scope.chat_id,
            self._sql_thread_id(scope.thread_id),
        )

    def record_session(
        self,
        scope: ScopeKey,
        profile_name: str,
        session_id: str,
    ) -> None:
        try:
            with self._connection() as conn, self._immediate(conn):
                conn.execute(
                    """
                    INSERT INTO profile_session_bindings (
                        platform, account_id, chat_id, thread_id, profile_name,
                        session_id, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(platform, account_id, chat_id, thread_id, profile_name)
                    DO UPDATE SET
                        session_id = excluded.session_id,
                        updated_at = excluded.updated_at
                    """,
                    (
                        *self._scope_values(scope),
                        profile_name,
                        session_id,
                        self._clock(),
                    ),
                )
        except sqlite3.Error as exc:
            raise self._unavailable(exc) from exc

    def get_session(self, scope: ScopeKey, profile_name: str) -> str | None:
        try:
            with self._connection() as conn:
                row = conn.execute(
                    """
                    SELECT session_id FROM profile_session_bindings
                    WHERE platform = ? AND account_id = ? AND chat_id = ?
                      AND thread_id = ? AND profile_name = ?
                    """,
                    (*self._scope_values(scope), profile_name),
                ).fetchone()
        except sqlite3.Error as exc:
            raise self._unavailable(exc) from exc
        return None if row is None else str(row["session_id"])

    def append_audit(
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
        try:
            with self._connection() as conn, self._immediate(conn):
                self._insert_audit(
                    conn,
                    actor_user_id=actor_user_id,
                    scope=scope,
                    old_profile=old_profile,
                    new_profile=new_profile,
                    scope_kind=scope_kind,
                    source=source,
                    result=result,
                    reason_code=reason_code,
                )
        except sqlite3.Error as exc:
            raise self._unavailable(exc) from exc

    def _insert_audit(
        self,
        conn: sqlite3.Connection,
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
        conn.execute(
            """
            INSERT INTO profile_switch_audit (
                created_at, actor_user_id, platform, account_id, chat_id,
                thread_id, old_profile, new_profile, scope_kind, source,
                result, reason_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._clock(),
                actor_user_id,
                scope.platform,
                scope.account_id,
                scope.chat_id,
                scope.thread_id,
                old_profile,
                new_profile,
                scope_kind.value if scope_kind is not None else None,
                source,
                result,
                reason_code.value,
            ),
        )

    def prune(
        self,
        *,
        audit_before: float,
        audit_max_rows: int | None = None,
    ) -> tuple[int, int]:
        if audit_max_rows is not None and audit_max_rows < 0:
            raise ValueError("audit_max_rows must not be negative")
        try:
            with self._connection() as conn, self._immediate(conn):
                audit_removed = conn.execute(
                    "DELETE FROM profile_switch_audit WHERE created_at < ?",
                    (audit_before,),
                ).rowcount
                if audit_max_rows is not None:
                    audit_removed += conn.execute(
                        """
                        DELETE FROM profile_switch_audit
                        WHERE id IN (
                            SELECT id FROM profile_switch_audit
                            ORDER BY created_at DESC, id DESC
                            LIMIT -1 OFFSET ?
                        )
                        """,
                        (audit_max_rows,),
                    ).rowcount
                nonces_removed = conn.execute(
                    "DELETE FROM profile_picker_nonces WHERE expires_at <= ?",
                    (self._clock(),),
                ).rowcount
        except sqlite3.Error as exc:
            raise self._unavailable(exc) from exc
        return audit_removed, nonces_removed

    def _unavailable(self, exc: BaseException) -> ProfileRoutingStoreUnavailable:
        return ProfileRoutingStoreUnavailable(
            f"profile routing database is unavailable: {self.path}"
        )
