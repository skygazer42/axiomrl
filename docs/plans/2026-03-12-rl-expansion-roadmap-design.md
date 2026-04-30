# RL Expansion Roadmap Design

**Date context:** This roadmap is written on **March 12, 2026**. Years 2025
and 2026 are treated as partially consolidated frontier years, not fully stable
package commitments.

## Why This Document Exists

The repository already has a credible `core + contrib + zoo` baseline for:

- `PPO` / `A2C`
- `DQN` and several value-based variants
- `SAC` / `TD3` / `DDPG` / `TQC` / `REDQ`
- `IQL` / `CQL` / `TD3+BC`
- Atari CNN pipelines
- `RecurrentPPO`

Companion planning documents for the current expansion wave:

- `docs/plans/2026-03-12-her-goal-replay-design.md`
- `docs/plans/2026-03-12-her-goal-replay-phase4.md`
- `docs/plans/2026-03-12-rl-yearly-sourcebook-design.md`
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

The package has also moved beyond “just more trainers” and now includes shared
infrastructure for:

- file-backed offline datasets (`random`, `.npz`, `.pt`, Minari)
- mixed offline datasets through `dataset_mix`
- reward scaling / shifting / clipping for datasets and environments
- named reward presets for common RL transforms
- offline and online evaluation cadence controls
- early stopping callbacks
- `BC`, `AWR`, `AWAC`, `MARWIL`, `BCQ`, `BEAR`, and `HER`
- `CRR`
- `Cal-QL`
- `XQL`
- `EDAC`
- `RLPD`
- `ReBRAC`

The next roadmap therefore needs to do two things at once:

1. keep expanding algorithm coverage toward mainstream expectations
2. protect the package shape so new algorithms reuse shared runtime, data, and
   product surfaces instead of fragmenting them

## Ecosystem Baseline From Current Libraries

Current mainstream RL libraries still converge on a relatively small set of
algorithms that matter in practice:

- Stable-Baselines3 keeps a compact core around `A2C`, `DDPG`, `DQN`, `HER`,
  `PPO`, `SAC`, and `TD3`.
- SB3 Contrib keeps specialized extensions such as `TRPO`, `RecurrentPPO`,
  `QR-DQN`, `TQC`, `CrossQ`, and `ARS` outside the stability-critical core.
- RLlib’s current public algorithm page still centers `PPO`, `APPO`,
  `IMPALA`, `DQN/Rainbow`, `SAC`, `DreamerV3`, `BC`, `CQL`, and `MARWIL`.
- d3rlpy remains a strong signal for offline demand with `BC`, `BCQ`, `BEAR`,
  `CRR`, `CQL`, `Cal-QL`, `IQL`, `AWR`, `AWAC`, `ReBRAC`, and related batch-RL baselines.

This gives `axiomrl` a practical target surface:

- `BCQ` and `BEAR` are now part of the current offline core wave
- `TRPO` has now landed in a narrow vector-observation v1 release
- `Discrete SAC` has now landed as the first modern discrete actor-critic core
  baseline
- `CrossQ` has now landed as a lower-tuning continuous-control v1 baseline
- `DrQ-v2` has now landed as a first pixel-observation continuous-control v1
  baseline
- `CRR` has now landed as a low-friction offline follow-on that reuses the
  current `AWAC/CQL/IQL` infrastructure
- `Cal-QL` has now landed as a calibrated 2022 offline follow-on that reuses
  the current `CQL` infrastructure plus returns-to-go processing
- `EDAC` has now landed as an ensemble-diversified 2022 offline follow-on that
  reuses the current multi-critic continuous actor-critic lane
- `RLPD` has now landed as a prior-data 2022 offline-to-online follow-on that
  reuses the current `SAC` actor-critic lane plus the offline dataset stack
- `AWR` has now landed as a low-friction offline actor/value baseline that
  reuses the current returns-to-go processing and actor/value model family
- `MARWIL` has now landed as a low-friction RLlib-style offline imitation
  baseline that reuses the same actor/value lane plus running advantage scaling
- `XQL` has now landed as an `IQL`-adjacent offline follow-on that reuses the
  current actor / critic / value infrastructure
- `ReBRAC` has now landed as a low-friction 2023 offline follow-on that reuses
  the current `TD3+BC` infrastructure
- `IMPALA` / `APPO` only after collector-learner orchestration is redesigned
- `DreamerV3` / `TD-MPC(2)` only after a world-model runtime exists

## Planning Rule For The Yearly Sourcebook

The yearly sourcebook below is a package-planning sourcebook, not a promise to
implement every item immediately.

Two caveats matter:

1. Early years such as 2014 do not contain six equally canonical deep RL
   algorithms. In those years, package-relevant foundational recipes are
   included alongside named algorithms.
2. Years 2025 and 2026 are incomplete as of **March 12, 2026**, so they are
   represented as watchlists of active families rather than fixed commitments.

## Yearly Algorithm Sourcebook (2014-2026)

### 2014

- `DQN`
- `Deterministic Policy Gradient` as the practical continuous-control precursor
- replay-buffer Q-learning as a reusable deep RL runtime pattern
- target-network stabilization as a reusable value-learning pattern
- convolutional pixel-control Q-learning as the first serious Atari recipe
- deterministic continuous-control actor-critic as a future package lane

### 2015

- `TRPO`
- `Double DQN`
- `Prioritized Experience Replay`
- `DDPG`
- `GAE`
- trust-region policy-gradient baselines for stable on-policy training

### 2016

- `A3C`
- `A2C` as the synchronous production-friendly variant
- `Dueling DQN`
- `NAF`
- `ACER`
- `UNREAL`

### 2017

- `PPO`
- `C51`
- `NoisyNet DQN`
- `Rainbow DQN`
- `HER`
- `ACKTR`

### 2018

- `SAC`
- `TD3`
- `QR-DQN`
- `IQN`
- `IMPALA`
- `Ape-X`

### 2019

- `BCQ`
- `BEAR`
- `MPO`
- `Dreamer`
- `R2D2`
- `MuZero`

### 2020

- `CQL`
- `AWAC`
- `DrQ`
- `CURL`
- `AWR` as a practical advantage-weighted imitation / RL bridge
- `PPG`

### 2021

- `IQL`
- `REDQ`
- `TD3+BC`
- `DrQ-v2`
- `Decision Transformer`
- `CRR`

### 2022

- `TD-MPC`
- `Cal-QL`
- `EDAC`
- `XQL`
- `RLPD`
- `Discrete SAC` as a mainstream package target

### 2023

- `DreamerV3`
- `TD-MPC2`
- `Diffusion-QL`
- `ReBRAC`
- offline-to-online hybrids built around `AWAC` / `IQL` / `TD-MPC`
- sequence-model RL follow-ons to `Decision Transformer`

### 2024

- `CrossQ`
- state-space / Mamba-style sequence-model control
- stronger `DreamerV3` deployment and scaling recipes
- stronger `TD-MPC2` implementation wave
- offline-to-online actor-critic hybrids becoming product-relevant
- world-model planning stacks becoming package-relevant rather than purely research-only

### 2025

- `DreamerV3` robustness / exploration extensions
- `CrossQ` stabilization and scaling follow-ons
- policy-constrained `TD-MPC` / `TD-MPC2` variants
- larger-model offline actor-critic distillation and fine-tuning families
- sequence / state-space control models becoming more practical
- stronger sim-to-real and transfer-oriented actor-critic variants

### 2026

- treat **2026 as watchlist-only on March 12, 2026**
- hybrid continuous-control actor-critic / Q-learning variants
- larger sequence-model control policies
- world-model transfer stacks with stronger evaluation discipline
- offline-to-online curriculum and adaptation families
- benchmark and validation stacks becoming as important as the algorithm itself

## Recommended Implementation Order

Chronological order is a bad implementation order. The package should instead
move by leverage.

### Wave A: Offline And Goal-Conditioned Consolidation

Already landed:

1. `BC`
2. `AWAC`
3. `HER`

Why this wave first:

- it forces real data loading instead of synthetic-only paths
- it adds imitation and sparse-reward coverage without redesigning the runtime
- it proves the package can support non-trivial training regimes through the
  same config / checkpoint / API surfaces

### Wave B: Canonical Batch RL

Now landed:

1. `BCQ`
2. `BEAR`
3. shared offline mixing / schedule / budget utilities

Why this wave mattered:

- `BCQ` and `BEAR` are still the most recognizable classical offline RL
  baselines users expect after `CQL`, `IQL`, and `TD3+BC`
- both stress policy-constraint machinery, generative action support, and
  offline evaluation discipline
- adding them together encourages a shared offline support layer instead of two
  isolated trainers

Execution handoff for this wave now lives in:

- `docs/plans/2026-03-12-bcq-bear-phase5.md`

### Wave C: Mainstream On-Policy Completeness

Recommended after Wave B:

1. `TRPO`
2. `Discrete SAC` if discrete-control demand is strong
3. `CrossQ` as a low-friction modern continuous-control addition
4. stronger benchmark presets for Atari and classic control

Why:

- `TRPO` remains a recognizable mainstream baseline and clarifies trust-region
  support in the package
- `Discrete SAC` fills a practical product gap more often than more obscure
  research algorithms do

### Wave D: Scaled Actor-Learner RL

Only after the runtime is redesigned:

1. `IMPALA`
2. `APPO`
3. possibly `R2D2`

Why deferred:

- these require a new sample-collection / learner-orchestration story
- adding them prematurely would produce a misleading algorithm count without a
  credible runtime

### Wave E: World Models

Only after a dedicated world-model runtime exists:

1. `DreamerV3`
2. `TD-MPC`
3. `TD-MPC2`

Why deferred:

- they need latent-dynamics models, imagination rollouts, planning loops,
  sequence replay, and different evaluation conventions
- they are large product bets, not “just one more trainer”

## Shared Infrastructure That Should Land Before More Algorithm Names

The next package gaps are now more important than another bare trainer.

### Data Processing

- demo / offline dataset mixing rather than single-source datasets only
- later: trajectory slicing, sequence windows, and prioritized offline sampling
- later: normalization-stat caching for train / eval parity
- later: goal-conditioned dataset utilities beyond `HER future`

### Reward Handling

- reward preset loading for common continuous-control tasks
- reward decomposition hooks for multi-term reward logging
- later: normalization and per-component reward metrics

### Training Controls

- budget rules for offline epochs vs gradient steps
- shared schedule utilities for behavior-cloning weight, constraint strength,
  entropy, and exploration
- stronger early-stopping rules beyond reward threshold / no-improvement
- max-episode / dataset-pass guards for offline trainers

### Product Surfaces

- more packaged configs, not just ad-hoc examples
- benchmark manifests and reference runs for new algorithms
- docs that explain which algorithms are `core`, which should stay `contrib`,
  and which are frontier watchlist only

## Immediate Phase 5 Scope

The next code batch should focus on `BCQ` / `BEAR` readiness instead of another
random algorithm count bump:

1. add a detailed Phase 5 plan for `BCQ` + `BEAR`
   Status: completed in `docs/plans/2026-03-12-bcq-bear-phase5.md`
2. add shared offline schedule / budget utilities
3. add one constrained offline actor baseline first (`BCQ`)
4. add one support-matching baseline second (`BEAR`)
5. then expand presets, docs, and benchmark recipes around the offline wave

## Research Sources Used

Current ecosystem docs:

- Stable-Baselines3 docs: https://stable-baselines3.readthedocs.io/en/master/
- SB3 Contrib docs: https://sb3-contrib.readthedocs.io/en/master/
- RLlib algorithms docs: https://docs.ray.io/en/latest/rllib/rllib-algorithms.html
- d3rlpy algorithms docs: https://d3rlpy.readthedocs.io/en/v0.41/references/algos.html

Representative primary papers:

- DPG: https://proceedings.mlr.press/v32/silver14.html
- TRPO: https://proceedings.mlr.press/v37/schulman15.html
- PPO: https://arxiv.org/abs/1707.06347
- SAC: https://proceedings.mlr.press/v80/haarnoja18b.html
- BCQ: https://arxiv.org/abs/1812.02900
- BEAR: https://arxiv.org/abs/1906.00949
- AWR: https://arxiv.org/abs/1910.00177
- CQL: https://arxiv.org/abs/2006.04779
- AWAC: https://arxiv.org/abs/2006.09359
- CRR: https://arxiv.org/abs/2006.15134
- IQL: https://arxiv.org/abs/2110.06169
- Cal-QL: https://arxiv.org/abs/2303.05479
- EDAC: https://arxiv.org/abs/2110.01548
- RLPD: https://arxiv.org/abs/2208.07544
- REDQ: https://arxiv.org/abs/2101.05982
- ReBRAC: https://arxiv.org/abs/2305.09836
- DrQ-v2: https://arxiv.org/abs/2107.09645
- Decision Transformer: https://arxiv.org/abs/2106.01345
- XQL: https://arxiv.org/abs/2301.02328
- TD-MPC: https://proceedings.mlr.press/v162/hansen22a.html
- DreamerV3: https://arxiv.org/abs/2301.04104
- TD-MPC2: https://arxiv.org/abs/2310.16828
- CrossQ: https://openreview.net/forum?id=1vARvraAjo
