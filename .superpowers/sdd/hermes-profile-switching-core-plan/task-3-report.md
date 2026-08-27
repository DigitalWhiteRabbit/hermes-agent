# Task 3 report: typed profile-switching configuration

## Implementation

Added frozen `ProfileSwitchRule` and `ProfileSwitchingConfig` dataclasses to `gateway/config.py` and added `GatewayConfig.profile_switching`. The parser accepts only the documented keys, coerces admin and rule IDs to strings, normalizes and validates profile names, rejects malformed positive limits and rule keys, and enforces the profile visibility safety invariants.

The effective multiplex setting and normalized allowlist are resolved before parsing profile switching. `default` is always considered served; an absent allowlist retains the existing serve-all meaning; visible profiles must be served; hidden profiles may remain unserved; visible/hidden overlap is rejected; and `enabled: true` requires multiplex mode.

## Files

- `gateway/config.py`
- `tests/gateway/profile_switching/test_config.py`
- `tests/gateway/test_config.py`
- `.superpowers/sdd/hermes-profile-switching-core-plan/task-3-report.md`

## TDD evidence

Initial RED command:

```text
HERMES_TEST_FILE_RETRIES=0 scripts/run_tests.sh tests/gateway/profile_switching/test_config.py -q
```

Initial RED output (exit 1):

```text
Discovered 1 test files (~13 tests) under ['tests/gateway/profile_switching/test_config.py']; running with -j 20
[100.0% |    13/~13 | ✓ 0 | ✗16] ✗ tests/gateway/profile_switching/test_config.py (16✗, 0.5s)
=== Summary: 1 files, 0 tests passed, 16 failed (100% complete) in 0.5s (20 workers) ===
```

The failures were the intended missing-feature failures: `GatewayConfig` had no `profile_switching` attribute, and the unsafe inputs did not raise `ValueError`.

During self-review, the unconditional “visible profiles must be served” ruling received its own second RED cycle.

Second RED command:

```text
HERMES_TEST_FILE_RETRIES=0 scripts/run_tests.sh tests/gateway/profile_switching/test_config.py -q
```

Second RED output (exit 1):

```text
[100.0% |    14/~14 | ✓16 | ✗ 1] ✗ tests/gateway/profile_switching/test_config.py (16✓ 1✗, 0.5s)
FAILED tests/gateway/profile_switching/test_config.py::test_visible_profile_must_be_served_before_switching_is_enabled
=== Summary: 1 files, 16 tests passed, 1 failed (100% complete) in 0.5s (20 workers) ===
```

Final GREEN command:

```text
HERMES_TEST_FILE_RETRIES=0 scripts/run_tests.sh tests/gateway/profile_switching/test_config.py tests/gateway/test_config.py -q && uv run ruff check gateway/config.py tests/gateway/profile_switching/test_config.py && git diff --check
```

Final GREEN output (exit 0):

```text
[ 18.4% |    14/~76 | ✓17 | ✗ 0] ✓ tests/gateway/profile_switching/test_config.py (17✓, 0.3s)
[100.0% |    76/~76 | ✓79 | ✗ 0] ✓ tests/gateway/test_config.py (62✓, 3.5s)
=== Summary: 2 files, 79 tests passed, 0 failed (100% complete) in 3.5s (20 workers) ===
All checks passed!
```

The first combined existing-config run exposed a missing local optional dependency (`aiohttp`) rather than a product failure: the pre-existing Slack bridge test could not load the Slack plugin. I restored the repository's declared test/runtime extras with `uv sync --extra dev --extra slack`, then reran the exact combined suite above successfully. No dependency or lock files changed.

## Serialization and backward compatibility

- `GatewayConfig.to_dict()` emits a complete `profile_switching` mapping, and `GatewayConfig.from_dict()` round-trips it back to equal typed values.
- `default_visible`, `hidden`, and admin IDs serialize as lists. Rules serialize through `asdict`; their tuple ID fields are accepted by the parser on round-trip.
- A missing `profile_switching` section produces `enabled=False`, TTL 300, retention 30 days, and 10,000 maximum audit rows. No existing gateway behavior is activated.
- Top-level `profile_switching` wins by key presence; nested `gateway.profile_switching` is used only when the top-level key is absent. `load_gateway_config()` carries both forms through the same precedence bridge.
- The new `GatewayConfig` field is appended after existing fields to avoid changing prior positional field indexes.

## Self-review

Reviewed the implementation line by line against the task brief and mutation-checked each relevant branch against a test:

- removing the default-disabled field or defaults breaks the default test;
- skipping nested parsing, ID coercion, profile normalization, or rule construction breaks parse/round-trip tests;
- removing multiplex, served-profile, overlap, hidden-profile, implicit-default, positive-limit, unknown-key, or rule-key handling breaks a dedicated safety test;
- reversing top-level precedence breaks the precedence test;
- dropping the real YAML loader bridge breaks the temp-`HERMES_HOME` regression test;
- removing `frozen=True` breaks the immutability test.

Confirmed that the change adds no command UI, database initialization, environment setting, or runtime switching behavior. `git diff --check` is clean.

## Concerns

None identified within the requested scope. An absent multiplex allowlist intentionally retains the established “serve all profiles” behavior, so there is no finite served set to validate against in that mode.
