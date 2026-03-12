# Mainstream RL Package Design

**Date:** 2026-03-12

Related documents:

- `docs/plans/2026-03-09-rl-package-roadmap-design.md`
- `docs/plans/2026-03-09-rl-training-package.md`
- `docs/plans/2026-03-12-atari-recurrent-ppo-phase1.md`
- `docs/plans/2026-03-12-bcq-bear-phase5.md`
- `docs/plans/2026-03-12-trpo-discrete-sac-crossq-phase6.md`
- `docs/plans/2026-03-12-drqv2-phase7.md`
- `docs/plans/2026-03-12-crr-phase8.md`
- `docs/plans/2026-03-12-rebrac-phase9.md`
- `docs/plans/2026-03-12-calql-phase10.md`
- `docs/plans/2026-03-12-xql-phase11.md`
- `docs/plans/2026-03-12-edac-phase12.md`
- `docs/plans/2026-03-12-rlpd-phase13.md`
- `docs/plans/2026-03-12-awr-phase14.md`
- `docs/plans/2026-03-12-marwil-phase15.md`

## Goal

Define how `rl_training` should evolve from a growing RL algorithm collection
into an easy-to-adopt mainstream reinforcement learning package.

This document answers a practical question:

> What product shape gives `rl_training` the best chance of becoming a widely
> used RL deep learning package?

## External Product Anchors

The direction in this document is based on the public positioning of the main
projects users already treat as reference points:

- Stable-Baselines3 keeps a stable core API with strong ergonomics and readable
  training workflows.
- `sb3-contrib` isolates more advanced or less battle-tested algorithms such as
  recurrent PPO from the core stability promise.
- RL Baselines3 Zoo turns presets, benchmark configs, and run scripts into a
  first-class product surface rather than leaving them as scattered examples.
- CleanRL proves that highly readable reference scripts and benchmark visibility
  materially improve adoption.
- Tianshou and TorchRL show that collector, environment, and runtime boundaries
  matter as much as algorithm count once a library moves beyond toy scale.

References:

- https://github.com/DLR-RM/stable-baselines3
- https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
- https://github.com/DLR-RM/rl-baselines3-zoo
- https://github.com/vwxyzjn/cleanrl
- https://github.com/thu-ml/tianshou
- https://github.com/pytorch/rl

## Product Thesis

`rl_training` should not try to win by adding the largest number of algorithm
names. The fastest path to mainstream adoption is to make the package easy to
install, easy to run, easy to trust, and easy to reproduce.

That means the package should optimize for:

- stable training and evaluation entrypoints
- predictable public API design
- environment-specific presets that work without hand-tuning
- clear documentation and reference scripts
- benchmark visibility for common tasks
- modular internal boundaries so new capability layers do not collapse the core

Popularity is treated here as an outcome of usability, coverage, and
reproducibility, not as a branding exercise.

## Proposed Product Shape

The package should evolve into three product layers:

### 1. Core

The existing `rl_training` package remains the stable, documented surface for
mainstream algorithms and common workflows:

- train / eval / resume / checkpoint
- typed config
- environment factories
- rollout and replay data systems
- stable public API objects such as `PPO`, `DQN`, and `SAC`

Core should bias toward algorithms and workflows that are broadly used and
operationally easy to explain.

### 2. Contrib

A new `rl_training.contrib` layer should hold algorithms or execution styles
that are valuable, but add extra state or edge cases that would otherwise
complicate the core contract.

The first `contrib` algorithm should be `RecurrentPPO`.

This mirrors the mainline ecosystem pattern: the package can support stronger
capabilities without forcing every stable path to absorb recurrent state
management, sequence masking, and hidden-state checkpoint semantics.

### 3. Zoo

A new `zoo/` product layer should become the home for:

- benchmark-ready presets
- environment-family hyperparameter bundles
- reproducible run commands
- result manifests and summary tables
- example run recipes for documentation and CI smoke checks

The goal is to stop treating examples and configs as secondary artifacts.

## Phase Strategy

The recommended roadmap is intentionally narrow.

### Phase 1A: Atari and CNN Infrastructure

Build the pieces that make the package look and feel like a mainstream RL
library rather than a tabular-classic-control trainer:

- Atari environment wrappers and preprocessing transforms
- pixel-observation support in environment factories
- CNN feature extractors, starting with a `NatureCNN` baseline
- trainer compatibility for image observations in DQN and PPO paths
- smoke tests and configs for Atari training

This is the first missing layer that users expect from a serious RL package.

### Phase 1B: Recurrent PPO

Add `RecurrentPPO` as the first new headline algorithm.

Why this algorithm first:

- it is an established mainstream extension
- it complements Atari and partial-observability use cases
- it adds new capability instead of duplicating existing DQN-family variants
- it can be built on top of the current PPO mental model

The first version should be deliberately narrow:

- LSTM-based actor-critic only
- discrete-action focus first
- no distributed execution
- no attempt to generalize all existing policies to recurrence on day one

### Phase 1C: Zoo and Benchmark Productization

Turn the new Atari path into a product surface:

- named Atari presets
- reproducible benchmark scripts
- run summaries and reference metrics
- docs that point users to stable entry commands

Without this layer, new capability remains invisible and difficult to trust.

### Phase 1D: Packaging and Documentation Polish

Close the productization gap:

- add installable CLI entrypoints in `pyproject.toml`
- document recommended train / eval / resume flows
- explain the difference between `core`, `contrib`, and `zoo`
- add a short "start here" guide for classic control and Atari

## Algorithm Roadmap After Phase 1

After Atari, recurrent PPO, and zoo are stable, the next algorithms should be
selected based on product leverage instead of novelty.

Status update on **March 12, 2026**:

- `HER`, `BC`, `AWAC`, `BCQ`, and `BEAR` are now part of the active package
  expansion wave
- `CRR` has now landed as another offline actor-critic baseline on the same
  dataset / checkpoint / API surface
- `Cal-QL` has now landed as a calibrated `CQL` follow-on on the same offline
  SAC-family runtime lane
- `EDAC` has now landed as an ensemble-diversified offline actor-critic
  follow-on on the current `REDQ`-style runtime lane
- `RLPD` has now landed as a prior-data offline-to-online follow-on on the
  current `SAC` runtime lane
- `AWR` has now landed as a narrow return-weighted offline actor/value baseline
  on top of the current offline dataset and returns-to-go processing surface
- `MARWIL` has now landed as a narrow weighted offline imitation / RL bridge
  on top of the same actor/value and returns-to-go package surface
- `XQL` has now landed as an extreme-value `IQL` follow-on on the same offline
  actor / critic / value runtime lane
- `ReBRAC` has now landed as the first 2023 offline follow-on on top of the
  existing `TD3+BC` runtime lane
- shared offline data loading, reward presets, and schedule / budget controls
  are no longer only roadmap items
- `TRPO`, `Discrete SAC`, `CrossQ`, and `DrQ-v2` have now moved onto the
  package surface
- package-facing `EDAC` tests have been added but remain intentionally
  unexecuted until explicitly requested
- package-facing `RLPD` tests have also been added but remain intentionally
  unexecuted until explicitly requested
- package-facing `AWR` tests have also been added but remain intentionally
  unexecuted until explicitly requested
- package-facing `MARWIL` tests have also been added but remain intentionally
  unexecuted until explicitly requested
- the next mainstream gaps have shifted past this wave and toward stronger
  validation coverage plus newer follow-on baselines

Recommended order:

1. stronger preset, benchmark, and validation coverage for the offline and
   pixel-control waves.
2. `IMPALA` / `APPO` once the runtime story is ready.
3. newer model-based or sequence-model families after infrastructure matures.

Algorithms explicitly deferred:

- `IMPALA`, `Ape-X`, `R2D2`: require a new distributed runtime story.
- `Dreamer` and model-based RL: require world-model infrastructure and change
  the package identity too early.
- `QMIX`, `MAPPO`, and broader MARL: belong after the single-agent zoo and
  evaluation discipline are mature.

## Architectural Constraints

To keep the package coherent, the following rules should be treated as
non-negotiable:

1. Do not fork full trainers just because observations change from vector to
   pixels. Observation encoding must be a composable model concern whenever
   possible.
2. Do not push recurrent state handling into every policy in the core package.
   Keep recurrence explicit and initially isolated to `contrib`.
3. Do not add more algorithm names if benchmark presets and docs for existing
   algorithms are still missing.
4. Do not start distributed RL before single-process Atari training, evaluation,
   and reproducibility are stable.
5. Do not let `zoo` become a second runtime implementation. It should curate
   configs, scripts, and results, not compete with the core API.

## Success Criteria

Phase 1 should be considered successful only if all of the following are true:

- a user can install the package and run Atari DQN and PPO from documented
  commands
- pixel observations work through the standard env and trainer stack
- `RecurrentPPO` can be trained, checkpointed, resumed, and evaluated through a
  documented public surface
- the repository contains benchmark presets and reproducible run recipes under
  `zoo/`
- README and package docs clearly explain the stable core vs `contrib` split

## Non-Goals

This roadmap does not attempt to maximize research novelty in the next phase.
It intentionally favors adoption drivers over breadth:

- no distributed rollout system in Phase 1
- no multi-agent support in Phase 1
- no model-based RL in Phase 1
- no broad new continuous-control family expansion before Atari is productized

## Recommended Next Step

Execute a focused implementation plan for:

- Atari wrappers and pixel-observation support
- CNN feature extraction
- `RecurrentPPO`
- `contrib` package boundaries
- `zoo` presets, scripts, and benchmark docs
- packaging and CLI polish

That plan is captured in `docs/plans/2026-03-12-atari-recurrent-ppo-phase1.md`.
