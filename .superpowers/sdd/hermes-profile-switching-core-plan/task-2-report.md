# Task 2 report: immutable profile-switching domain types

## Implementation

Added dependency-free immutable domain models in `gateway/profile_switching/models.py`: `ScopeKind`, `ResolutionSource`, `ReasonCode`, and frozen `ScopeKey`, `ProfileBinding`, `PolicyDecision`, and `ProfileResolution`. `ScopeKey.from_source` consumes `SessionSource` through attribute access, normalizes identifiers to strings, and `for_kind(ScopeKind.CHAT)` removes the thread ID. Exported all public types from `gateway/profile_switching/__init__.py`.

## Files

- `gateway/profile_switching/models.py`
- `gateway/profile_switching/__init__.py`
- `tests/gateway/profile_switching/test_models.py`

## TDD evidence

RED command: `uv run pytest tests/gateway/profile_switching/test_models.py -q`

RED output: `ModuleNotFoundError: No module named 'gateway.profile_switching'` during collection; this was expected because the production package did not yet exist (pytest exited 2).

GREEN/Ruff command: `uv run pytest tests/gateway/profile_switching/test_models.py -q && uv run ruff check gateway/profile_switching tests/gateway/profile_switching`

GREEN/Ruff output: `.... [100%]` / `4 passed in 0.11s` and `All checks passed!` (exit 0).

## Self-review

Reviewed the implementation against the brief, verified frozen dataclasses, stable enum values, string normalization, chat projection, public exports, and no imports of `gateway.run` or Telegram SDK types. `git diff --check` passed.

## Concerns

None identified within the requested scope. `for_kind` follows the specified behavior and treats non-`THREAD` kinds as chat projection.
