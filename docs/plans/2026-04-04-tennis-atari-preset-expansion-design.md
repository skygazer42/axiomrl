# Tennis Atari Preset Expansion Design

**Date:** 2026-04-04

**Goal:** Add `Agent57` and `EfficientZero` as first-class Tennis benchmark options without changing trainer code or the existing Atari benchmark protocol.

## Scope

This change is intentionally narrow:

- add one Tennis config for `Agent57`
- add one Tennis config for `EfficientZero`
- add one zoo preset file for each config
- register both presets in the existing Tennis benchmark manifest
- add focused tests that prove the new packaged presets resolve and inherit benchmark defaults

This change does not modify trainer implementations, environment wrapper code, CLI semantics, or the existing Breakout benchmark manifest.

## Design

The new Tennis presets should mirror the established pattern already used by `rainbow_dqn_tennis` and `r2d2_tennis`:

- a root config under `configs/<algo>/tennis_atari.yaml`
- a packaged asset mirror under `src/axiomrl/assets/configs/<algo>/tennis_atari.yaml`
- a zoo preset under `zoo/atari/<preset>.yaml`
- a packaged asset mirror under `src/axiomrl/assets/zoo/atari/<preset>.yaml`
- an entry in `zoo/atari/tennis_benchmark.yaml`

Both configs will be copied from the existing Breakout Atari configs for the same algorithms and adapted only where Tennis already has established conventions:

- `env_id` becomes `ALE/Tennis-v5`
- training and evaluation environment settings follow the Tennis protocol already used by existing Tennis configs
- `training.repeat_action_probability` is `0.0`
- `evaluation.repeat_action_probability` is `0.25`
- training uses clipped rewards and evaluation does not

## Rationale

This preserves consistency with the rest of the repository:

- Tennis remains driven by manifest-backed zoo presets.
- Packaged asset resolution continues to work outside the repository root.
- Existing test coverage patterns remain valid.

It also avoids overreaching. Adding `MuZero`, `DreamerV3`, or broader Tennis benchmark restructuring can be done later, but the current need is to expose the two highest-priority new candidates with minimal risk.

## Testing

The most valuable regression tests are configuration-level:

- `load_config("zoo/atari/agent57_tennis.yaml")` should resolve outside the repo root.
- `load_config("zoo/atari/efficientzero_tennis.yaml")` should resolve outside the repo root.
- both presets should inherit the Tennis benchmark protocol defaults from `tennis_benchmark.yaml`
- the Tennis benchmark manifest should include both new preset names

No training smoke test is required for this change because the feature is manifest/config wiring, not new runtime behavior.
