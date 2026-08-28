from __future__ import annotations

import sqlite3


LATEST_SCHEMA_VERSION = 1

_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """
        CREATE TABLE IF NOT EXISTS profile_bindings (
            platform TEXT NOT NULL,
            account_id TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT NOT NULL DEFAULT '',
            scope_kind TEXT NOT NULL CHECK(scope_kind IN ('chat', 'thread')),
            profile_name TEXT NOT NULL,
            created_by_user_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            expires_at REAL,
            version INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY(platform, account_id, chat_id, scope_kind, thread_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS profile_once (
            platform TEXT NOT NULL,
            account_id TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT NOT NULL DEFAULT '',
            profile_name TEXT NOT NULL,
            created_by_user_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL,
            PRIMARY KEY(platform, account_id, chat_id, thread_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS profile_session_bindings (
            platform TEXT NOT NULL,
            account_id TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT NOT NULL DEFAULT '',
            profile_name TEXT NOT NULL,
            session_id TEXT NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY(platform, account_id, chat_id, thread_id, profile_name)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS profile_switch_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            actor_user_id TEXT NOT NULL,
            platform TEXT NOT NULL,
            account_id TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT,
            old_profile TEXT,
            new_profile TEXT,
            scope_kind TEXT,
            source TEXT NOT NULL,
            result TEXT NOT NULL,
            reason_code TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS profile_picker_nonces (
            nonce_hash TEXT PRIMARY KEY,
            platform TEXT NOT NULL,
            account_id TEXT NOT NULL,
            actor_user_id TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT,
            message_id TEXT NOT NULL,
            action TEXT NOT NULL,
            profile_name TEXT,
            expires_at REAL NOT NULL,
            used_at REAL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_profile_audit_created_at
        ON profile_switch_audit(created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_profile_nonce_expires_at
        ON profile_picker_nonces(expires_at)
        """,
    ),
}


def run_migrations(conn: sqlite3.Connection) -> None:
    """Apply every missing schema migration in one immediate transaction."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        current_version = int(row[0]) if row is not None else 0
        if current_version > LATEST_SCHEMA_VERSION:
            raise sqlite3.DatabaseError(
                "profile routing schema is newer than this Hermes version: "
                f"{current_version} > {LATEST_SCHEMA_VERSION}"
            )

        for version in range(current_version + 1, LATEST_SCHEMA_VERSION + 1):
            for statement in _MIGRATIONS[version]:
                conn.execute(statement)
            conn.execute(
                """
                INSERT INTO schema_meta(key, value) VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(version),),
            )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
