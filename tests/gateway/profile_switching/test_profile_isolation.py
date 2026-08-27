from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent import secret_scope
from agent.prompt_builder import (
    build_skills_system_prompt,
    clear_skills_system_prompt_cache,
    load_soul_md,
)
from gateway.config import (
    GatewayConfig,
    Platform,
    ProfileSwitchRule,
    ProfileSwitchingConfig,
)
from gateway.platforms.base import MessageEvent
from gateway.profile_switching.models import ScopeKey, ScopeKind
from gateway.profile_switching.policy import ProfilePolicy
from gateway.profile_switching.service import ProfileSwitchingService
from gateway.profile_switching.store import ProfileRoutingStore
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource, SessionStore
from hermes_constants import get_hermes_home
from hermes_cli.config import load_config
from hermes_cli.env_loader import reset_secret_source_cache
from tools.memory_tool import MemoryStore


_PROFILE_SENTINELS = {
    "default": "DEFAULT_SENTINEL",
    "coder": "CODER_SENTINEL",
    "copywriter": "COPYWRITER_SENTINEL",
    "research": "RESEARCH_SENTINEL",
}
_NAMED_PROFILES = tuple(name for name in _PROFILE_SENTINELS if name != "default")
_ACTOR_ID = "user-1"


@dataclass(frozen=True)
class _ProfileFixture:
    root: Path
    homes: dict[str, Path]
    workspaces: dict[str, Path]
    runner: GatewayRunner
    service: ProfileSwitchingService
    store: ProfileRoutingStore
    scope: ScopeKey
    primary_adapter: object


@dataclass(frozen=True)
class _RuntimeSnapshot:
    home: Path
    config_sentinel: str
    gateway_config_sentinel: str
    soul: str | None
    memory_entries: tuple[str, ...]
    skills_prompt: str
    credential: str | None
    workspace: Path
    workspace_marker: str


def _policy(*, allowed_profiles: tuple[str, ...]) -> ProfilePolicy:
    config = ProfileSwitchingConfig(
        enabled=True,
        rules=tuple(
            ProfileSwitchRule(
                profile,
                users=(_ACTOR_ID,),
                chats=("*",),
            )
            for profile in allowed_profiles
        ),
    )
    return ProfilePolicy(
        config,
        served_profiles=tuple(_PROFILE_SENTINELS),
        existing_profiles=tuple(_PROFILE_SENTINELS),
    )


@pytest.fixture
def profile_fixture(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    homes = {"default": root}
    homes.update({name: root / "profiles" / name for name in _NAMED_PROFILES})
    workspaces: dict[str, Path] = {}

    for profile, home in homes.items():
        sentinel = _PROFILE_SENTINELS[profile]
        workspace = home / "workspace"
        skill_dir = home / "skills" / "isolation" / f"{profile}-sentinel"
        (home / "memories").mkdir(parents=True, exist_ok=True)
        workspace.mkdir(parents=True, exist_ok=True)
        skill_dir.mkdir(parents=True, exist_ok=True)
        workspaces[profile] = workspace
        (home / "config.yaml").write_text(
            "\n".join(
                (
                    "profile_probe:",
                    f"  sentinel: {sentinel}",
                    "terminal:",
                    f"  cwd: {json.dumps(str(workspace))}",
                    "skills:",
                    "  project_discovery: false",
                    "",
                )
            ),
            encoding="utf-8",
        )
        (home / ".env").write_text(
            "\n".join(
                (
                    f"PROFILE_API_KEY={sentinel}_API_KEY",
                    f"{profile.upper()}_ONLY_SECRET={sentinel}_ONLY_SECRET",
                    "",
                )
            ),
            encoding="utf-8",
        )
        (home / "SOUL.md").write_text(f"{sentinel}_SOUL\n", encoding="utf-8")
        (home / "memories" / "MEMORY.md").write_text(
            f"{sentinel}_MEMORY\n",
            encoding="utf-8",
        )
        (skill_dir / "SKILL.md").write_text(
            "\n".join(
                (
                    "---",
                    f"name: {profile}-sentinel",
                    f"description: {sentinel} skill.",
                    "---",
                    f"# {sentinel}",
                    "",
                )
            ),
            encoding="utf-8",
        )
        (workspace / "profile.txt").write_text(
            f"{sentinel}_WORKSPACE\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("PROFILE_API_KEY", "PROCESS_GLOBAL_API_KEY")
    monkeypatch.setenv(
        "COPYWRITER_ONLY_SECRET",
        "PROCESS_GLOBAL_COPYWRITER_SECRET",
    )
    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    reset_secret_source_cache()
    clear_skills_system_prompt_cache(clear_snapshot=False)

    store = ProfileRoutingStore(root / "state" / "profile-routing.db")
    service = ProfileSwitchingService(
        store,
        _policy(allowed_profiles=tuple(_PROFILE_SENTINELS)),
    )
    config = GatewayConfig(
        multiplex_profiles=True,
        multiplex_profile_allowlist=list(_NAMED_PROFILES),
        write_sessions_json=False,
        profile_switching=ProfileSwitchingConfig(enabled=True),
    )
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner._profile_switching_service = service
    primary_adapter = object()
    runner.adapters = {Platform.TELEGRAM: primary_adapter}
    runner._profile_adapters = {}
    scope = ScopeKey("telegram", "telegram:primary", "chat-1", None)

    try:
        yield _ProfileFixture(
            root=root,
            homes=homes,
            workspaces=workspaces,
            runner=runner,
            service=service,
            store=store,
            scope=scope,
            primary_adapter=primary_adapter,
        )
    finally:
        clear_skills_system_prompt_cache(clear_snapshot=False)
        reset_secret_source_cache()
        secret_scope.set_multiplex_active(previous_multiplex)


def _set_chat_profile(fixture: _ProfileFixture, profile: str) -> None:
    fixture.service.set_profile(
        scope=fixture.scope,
        scope_kind=ScopeKind.CHAT,
        actor_user_id=_ACTOR_ID,
        profile_name=profile,
        active_turn=False,
    )


async def _resolve_next_turn(fixture: _ProfileFixture) -> MessageEvent:
    event = MessageEvent(
        text="profile isolation probe",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            chat_type="dm",
            user_id=_ACTOR_ID,
        ),
    )
    await fixture.runner._resolve_dynamic_profile_for_event(event)
    return event


def _runtime_snapshot(home: Path) -> _RuntimeSnapshot:
    from agent.secret_scope import get_secret
    from gateway.run import _load_gateway_config

    with _profile_runtime_scope(home):
        config = load_config()
        gateway_config = _load_gateway_config()
        memory = MemoryStore()
        memory.load_from_disk()
        workspace = Path(config["terminal"]["cwd"])
        return _RuntimeSnapshot(
            home=get_hermes_home(),
            config_sentinel=config["profile_probe"]["sentinel"],
            gateway_config_sentinel=gateway_config["profile_probe"]["sentinel"],
            soul=load_soul_md(),
            memory_entries=tuple(memory.memory_entries),
            skills_prompt=build_skills_system_prompt(
                skills_dir_override=home / "skills"
            ),
            credential=get_secret("PROFILE_API_KEY"),
            workspace=workspace,
            workspace_marker=(workspace / "profile.txt").read_text(
                encoding="utf-8"
            ).strip(),
        )


@pytest.mark.asyncio
async def test_dynamic_switch_changes_session_key_namespace(profile_fixture):
    _set_chat_profile(profile_fixture, "coder")
    coder = await _resolve_next_turn(profile_fixture)
    _set_chat_profile(profile_fixture, "copywriter")
    copywriter = await _resolve_next_turn(profile_fixture)

    assert profile_fixture.runner._session_key_for_source(coder.source) == (
        "agent:coder:telegram:dm:chat-1"
    )
    assert profile_fixture.runner._session_key_for_source(copywriter.source) == (
        "agent:copywriter:telegram:dm:chat-1"
    )
    for event in (coder, copywriter):
        assert event.source.transport_owner_profile == "default"
        assert event.source.transport_platform is Platform.TELEGRAM
        assert event.source._profile_transport_account_id == "telegram:primary"
        assert (
            profile_fixture.runner._adapter_for_source(event.source)
            is profile_fixture.primary_adapter
        )
        assert profile_fixture.runner._adapter_profile_for_source(event.source) is None


@pytest.mark.asyncio
async def test_runtime_scope_reads_only_selected_profile_config(profile_fixture):
    snapshots: dict[str, _RuntimeSnapshot] = {}
    for profile in ("coder", "copywriter"):
        _set_chat_profile(profile_fixture, profile)
        event = await _resolve_next_turn(profile_fixture)
        assert event.source.profile == profile
        resolved_home = profile_fixture.runner._resolve_profile_home_for_source(
            event.source
        )
        snapshots[profile] = _runtime_snapshot(resolved_home)

    for profile, snapshot in snapshots.items():
        sentinel = _PROFILE_SENTINELS[profile]
        assert snapshot.home == profile_fixture.homes[profile]
        assert snapshot.config_sentinel == sentinel
        assert snapshot.gateway_config_sentinel == sentinel
        assert snapshot.soul == f"{sentinel}_SOUL"
        assert snapshot.memory_entries == (f"{sentinel}_MEMORY",)
        assert sentinel in snapshot.skills_prompt
        assert snapshot.credential == f"{sentinel}_API_KEY"
        assert snapshot.workspace == profile_fixture.workspaces[profile]
        assert snapshot.workspace_marker == f"{sentinel}_WORKSPACE"

        sibling_sentinels = {
            other_sentinel
            for other, other_sentinel in _PROFILE_SENTINELS.items()
            if other != profile
        }
        selected_state = "\n".join(
            (
                snapshot.config_sentinel,
                snapshot.gateway_config_sentinel,
                snapshot.soul or "",
                *snapshot.memory_entries,
                snapshot.skills_prompt,
                snapshot.credential or "",
                snapshot.workspace_marker,
            )
        )
        assert all(sibling not in selected_state for sibling in sibling_sentinels)


@pytest.mark.asyncio
async def test_selected_profile_cannot_read_sibling_secret_scope(profile_fixture):
    from agent.secret_scope import get_secret
    from hermes_cli.runtime_provider import _getenv

    _set_chat_profile(profile_fixture, "coder")
    event = await _resolve_next_turn(profile_fixture)
    assert event.source.profile == "coder"
    resolved_home = profile_fixture.runner._resolve_profile_home_for_source(
        event.source
    )

    with _profile_runtime_scope(resolved_home):
        assert get_secret("PROFILE_API_KEY") == "CODER_SENTINEL_API_KEY"
        assert get_secret("COPYWRITER_ONLY_SECRET") is None
        assert _getenv("COPYWRITER_ONLY_SECRET") == ""

    with pytest.raises(secret_scope.UnscopedSecretError):
        get_secret("COPYWRITER_ONLY_SECRET")


@pytest.mark.asyncio
async def test_return_to_profile_reuses_its_own_session_only(profile_fixture):
    session_store = SessionStore(
        sessions_dir=profile_fixture.root / "sessions",
        config=profile_fixture.runner.config,
    )

    async def turn(profile: str):
        _set_chat_profile(profile_fixture, profile)
        event = await _resolve_next_turn(profile_fixture)
        resolved_home = profile_fixture.runner._resolve_profile_home_for_source(
            event.source
        )
        with _profile_runtime_scope(resolved_home):
            entry = session_store.get_or_create_session(event.source)
        profile_fixture.service.note_session(
            scope=profile_fixture.scope,
            profile_name=profile,
            session_id=entry.session_id,
        )
        return entry

    first_coder = await turn("coder")
    copywriter = await turn("copywriter")
    second_coder = await turn("coder")

    assert first_coder.session_key == "agent:coder:telegram:dm:chat-1"
    assert copywriter.session_key == "agent:copywriter:telegram:dm:chat-1"
    assert first_coder.session_id != copywriter.session_id
    assert second_coder.session_id == first_coder.session_id
    assert second_coder.session_id != copywriter.session_id
    assert profile_fixture.store.get_session(profile_fixture.scope, "coder") == (
        first_coder.session_id
    )
    assert profile_fixture.store.get_session(
        profile_fixture.scope,
        "copywriter",
    ) == copywriter.session_id


@pytest.mark.asyncio
async def test_concurrent_coder_and_copywriter_turns_restore_contextvars(
    profile_fixture,
):
    from agent.secret_scope import current_secret_scope, get_secret

    entered = {profile: asyncio.Event() for profile in ("coder", "copywriter")}
    release = asyncio.Event()

    async def concurrent_turn(profile: str):
        home = profile_fixture.homes[profile]
        sentinel = _PROFILE_SENTINELS[profile]
        with _profile_runtime_scope(home):
            entered[profile].set()
            await release.wait()
            config = load_config()
            await asyncio.sleep(0)
            return (
                get_hermes_home(),
                config["profile_probe"]["sentinel"],
                get_secret("PROFILE_API_KEY"),
                load_soul_md(),
            )

    coder_task = asyncio.create_task(concurrent_turn("coder"))
    copywriter_task = asyncio.create_task(concurrent_turn("copywriter"))
    await asyncio.gather(*(event.wait() for event in entered.values()))

    assert get_hermes_home() == profile_fixture.root
    assert current_secret_scope() is None
    release.set()
    coder, copywriter = await asyncio.gather(coder_task, copywriter_task)

    assert coder == (
        profile_fixture.homes["coder"],
        "CODER_SENTINEL",
        "CODER_SENTINEL_API_KEY",
        "CODER_SENTINEL_SOUL",
    )
    assert copywriter == (
        profile_fixture.homes["copywriter"],
        "COPYWRITER_SENTINEL",
        "COPYWRITER_SENTINEL_API_KEY",
        "COPYWRITER_SENTINEL_SOUL",
    )
    assert get_hermes_home() == profile_fixture.root
    assert current_secret_scope() is None


@pytest.mark.asyncio
async def test_policy_revocation_invalidates_existing_binding_next_turn(
    profile_fixture,
):
    _set_chat_profile(profile_fixture, "coder")
    allowed_turn = await _resolve_next_turn(profile_fixture)
    assert allowed_turn.source.profile == "coder"
    assert profile_fixture.runner._session_key_for_source(allowed_turn.source) == (
        "agent:coder:telegram:dm:chat-1"
    )

    revoked_service = ProfileSwitchingService(
        profile_fixture.store,
        _policy(allowed_profiles=("default", "copywriter", "research")),
    )
    profile_fixture.runner._profile_switching_service = revoked_service
    revoked_turn = await _resolve_next_turn(profile_fixture)

    stored = profile_fixture.store.get_binding(profile_fixture.scope, ScopeKind.CHAT)
    assert stored is not None and stored.profile_name == "coder"
    assert revoked_turn.source.profile is None
    assert revoked_turn.source.transport_owner_profile == "default"
    assert profile_fixture.runner._session_key_for_source(revoked_turn.source) == (
        "agent:main:telegram:dm:chat-1"
    )
    assert (
        profile_fixture.runner._adapter_for_source(revoked_turn.source)
        is profile_fixture.primary_adapter
    )
