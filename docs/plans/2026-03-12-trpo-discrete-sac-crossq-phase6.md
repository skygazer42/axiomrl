# TRPO, Discrete SAC, And CrossQ Phase 6 Plan

**Date:** 2026-03-12

## Status Update

Current implementation status in the repository:

- `TRPO` task is now implemented in a narrow v1 form:
  - discrete actions only
  - flat vector observations only
  - MLP actor-critic only
  - shared checkpoint / eval / resume / predict wiring included
  - starter configs and unexecuted tests added
- `Discrete SAC` task is now implemented in a narrow v1 form:
  - discrete actions only
  - flat vector observations only
  - categorical actor with twin Q critics
  - shared replay-buffer trainer plus checkpoint / eval / resume / predict
    wiring included
  - starter configs and unexecuted tests added
- `CrossQ` task is now implemented in a narrow v1 form:
  - continuous actions only
  - flat vector observations only
  - SAC-style actor with BatchNorm-backed critics
  - no target network in the v1 learner path
  - shared replay-buffer trainer plus checkpoint / eval / resume / predict
    wiring included
  - starter configs and unexecuted tests added

## Goal

Move `rl_training` from the classical offline RL consolidation wave
(`BC` / `AWAC` / `BCQ` / `BEAR` / `HER`) into the next mainstream package
gap set:

1. `TRPO` for recognizable on-policy completeness
2. `Discrete SAC` for a modern discrete actor-critic baseline
3. `CrossQ` for a lower-tuning modern continuous-control baseline

This phase should still protect the existing package shape:

- reuse the shared config / checkpoint / managed API surface
- avoid introducing a second runtime style
- keep first releases narrow and readable
- add benchmarkable starter configs alongside code paths

## Why This Wave Next

The package already covers:

- on-policy: `PPO`, `A2C`, `RecurrentPPO`
- value-based discrete RL: `DQN` family, `C51`, `QR-DQN`, `IQN`
- continuous-control off-policy RL: `DDPG`, `TD3`, `SAC`, `REDQ`, `TQC`
- offline RL: `BC`, `AWAC`, `IQL`, `CQL`, `TD3+BC`, `BCQ`, `BEAR`
- goal-conditioned replay: `HER`

The next mainstream gaps are therefore no longer “more offline names”. They are:

- a trust-region baseline users still expect (`TRPO`)
- a discrete actor-critic option users compare against modern DQN variants
  (`Discrete SAC`)
- a lower-tuning continuous-control addition that reflects the newer package
  landscape (`CrossQ`)

## Scope Constraints

### TRPO

First version should be:

- discrete-action only
- flat vector observations only
- MLP policy/value network only
- no recurrent state
- no image support in v1

### Discrete SAC

First version should be:

- discrete-action only
- flat vector observations only
- online replay-buffer trainer only
- categorical actor with twin Q critics
- checkpoint / eval / predict support through the normal registry

### CrossQ

First version should be:

- continuous-control only
- flat vector observations only
- online off-policy trainer only
- reuse as much SAC-style model structure as possible
- keep normalization / regularization choices explicit in config

## Recommended Execution Order

### Task 1: `TRPO`

**Status:** Implemented in v1. Keep as the reference for the remaining Phase 6
surface wiring pattern.

**Files:**

- Create: `src/rl_training/algorithms/trpo.py`
- Create: `src/rl_training/runtime/trpo_trainer.py`
- Modify: `src/rl_training/algorithms/__init__.py`
- Modify: `src/rl_training/experiment/registry.py`
- Modify: `src/rl_training/api/algorithms.py`
- Modify: `src/rl_training/api/__init__.py`
- Modify: `src/rl_training/__init__.py`
- Create: `configs/trpo/cartpole.yaml`
- Create: `src/rl_training/assets/configs/trpo/cartpole.yaml`
- Create: `tests/test_trpo_update.py`
- Create: `tests/test_trpo_trainer_smoke.py`

**Implementation notes:**

- start from the current PPO data path and rollout stack
- add conjugate-gradient + line-search updates in the smallest viable form
- preserve the current public train / eval / resume workflow

### Task 2: `Discrete SAC`

**Status:** Implemented in v1. The package now has a modern discrete
actor-critic baseline alongside value-based DQN-family baselines.

**Files:**

- Create: `src/rl_training/models/mlp_discrete_sac.py`
- Create: `src/rl_training/algorithms/discrete_sac.py`
- Create: `src/rl_training/runtime/discrete_sac_trainer.py`
- Modify: `src/rl_training/models/__init__.py`
- Modify: `src/rl_training/algorithms/__init__.py`
- Modify: `src/rl_training/experiment/registry.py`
- Modify: `src/rl_training/api/algorithms.py`
- Modify: `src/rl_training/api/__init__.py`
- Modify: `src/rl_training/__init__.py`
- Create: `configs/discrete_sac/cartpole.yaml`
- Create: `src/rl_training/assets/configs/discrete_sac/cartpole.yaml`
- Create: `tests/test_discrete_sac_update.py`
- Create: `tests/test_discrete_sac_trainer_smoke.py`

**Implementation notes:**

- actor outputs a categorical policy over discrete actions
- critics estimate action-values for all discrete actions
- evaluation and prediction should return integer actions through the standard
  workflow helpers

### Task 3: `CrossQ`

**Status:** Implemented in v1. This release stays close to the current
continuous-control package surface while making the normalization and update
choices explicit in config.

**Files:**

- Create: `src/rl_training/algorithms/crossq.py`
- Create: `src/rl_training/runtime/crossq_trainer.py`
- Modify: `src/rl_training/algorithms/__init__.py`
- Modify: `src/rl_training/experiment/registry.py`
- Modify: `src/rl_training/api/algorithms.py`
- Modify: `src/rl_training/api/__init__.py`
- Modify: `src/rl_training/__init__.py`
- Create: `configs/crossq/pendulum.yaml`
- Create: `src/rl_training/assets/configs/crossq/pendulum.yaml`
- Create: `tests/test_crossq_update.py`
- Create: `tests/test_crossq_trainer_smoke.py`

**Implementation notes:**

- keep the first version close to the current SAC runtime
- avoid introducing a new collector / learner architecture
- bias toward readability over paper-maximal details in v1

### Task 4: Product Surface And Docs

**Files:**

- Modify: `README.md`
- Modify: `tests/test_package_api_exports.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_experiment_manager.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_package_smoke.py`
- Modify: `tests/test_checkpoint_workflows.py`
- Modify: `docs/plans/2026-03-12-rl-expansion-roadmap-design.md`
- Modify: `docs/plans/2026-03-12-rl-yearly-sourcebook-design.md`

**Implementation notes:**

- document where `TRPO`, `Discrete SAC`, and `CrossQ` sit in the package
- keep the “core vs contrib vs zoo” split explicit
- extend starter config references and package asset assertions

## Non-Goals For Phase 6

- no distributed actor-learner runtime
- no world-model stack
- no multi-agent support
- no image-support expansion for `TRPO` or `Discrete SAC` in the first pass
- no attempt to merge all continuous-control algorithms under one giant meta-implementation

## Exit Criteria

Phase 6 should only be considered complete when:

- each algorithm has a starter config under both repo configs and packaged assets
- each algorithm is reachable through root exports, managed API, registry,
  checkpoint evaluation, resume, and prediction
- public-surface tests are present for each new algorithm
- README and roadmap docs reflect the new package surface
- test execution remains deferred until explicitly allowed by the user
