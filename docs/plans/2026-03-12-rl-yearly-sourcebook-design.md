# RL Yearly Sourcebook Design (2014-2026)

**Date context:** This sourcebook is written on **March 12, 2026**.

## Purpose

This document turns the user's request into a package-planning artifact:

- organize mainstream RL algorithm demand from **2014 to 2026**
- map that history to what a serious package should implement next
- separate **stable package targets** from **frontier watchlist items**

It is not a promise to implement every item. It is a prioritization sourcebook.

## How To Read This Document

The yearly lists below are intentionally pragmatic rather than academically
pure. I am making two explicit inferences:

1. For **2014-2016**, there were fewer canonical deep RL algorithms than in
   later years, so some entries are technique families that were essential to
   package design, not only standalone trainer names.
2. For **2024-2026**, the field is still consolidating, so some entries are
   package-relevant algorithm directions or implementation waves rather than
   universally settled standards.

Popularity is estimated from three signals:

- whether current mainstream libraries still surface the family
- whether the method remains a common benchmark or comparison baseline
- whether users building a general-purpose RL package still ask for it

## Ecosystem Popularity Signals On March 12, 2026

Current public library surfaces still cluster around the same anchors:

- **Stable-Baselines3** keeps a compact mainstream core around `A2C`, `DDPG`,
  `DQN`, `HER`, `PPO`, `SAC`, and `TD3`.
- **SB3 Contrib** continues to surface specialized but expected extensions such
  as `TRPO`, `QR-DQN`, `TQC`, `CrossQ`, `ARS`, and `RecurrentPPO`.
- **RLlib** still treats `PPO`, `DQN/Rainbow`, `SAC`, `IMPALA`, `APPO`,
  `DreamerV3`, `BC`, `CQL`, `IQL`, and `MARWIL` as current public algorithm choices.
- **d3rlpy** remains one of the clearest signals for offline and imitation
  demand, with `BC`, `BCQ`, `BEAR`, `CRR`, `CQL`, `AWR`, `AWAC`, `IQL`, and
  `ReBRAC`.

That means a serious package should optimize for:

- mainstream on-policy coverage
- strong off-policy continuous control coverage
- credible offline RL baselines
- sparse-reward and goal-conditioned workflows
- only then world-model and frontier sequence-model work

## Yearly Algorithm Intake Sourcebook

### 2014: Foundation Year

This year does **not** honestly offer six equally canonical standalone deep RL
algorithms. The package-relevant intake list is therefore a mix of algorithms
and the stabilization bundles that made deep RL viable.

- `DQN`
- `Deterministic Policy Gradient`
- replay-buffered deep value learning
- target-network stabilized deep Q-learning
- CNN-based pixel-control value networks

**Package meaning:** 2014 is why every serious RL package needs replay buffers,
target networks, image encoders, and robust Q-learning infrastructure.

### 2015: First Major Expansion

- `TRPO`
- `Double DQN`
- `Prioritized Experience Replay`
- `DDPG`
- `GAE`
- `A3C`

**Package meaning:** this is the first year where the package story clearly
splits into on-policy trust-region work, scalable actor-critic training, and
better value-based replay methods.

### 2016: Stabilization And Richer DQN Variants

- `A2C`
- `Dueling DQN`
- `NAF`
- `ACER`
- `UNREAL`
- `DRQN`

**Package meaning:** a package that only supports vanilla DQN/PPO misses a lot
of the representational and recurrent ideas users still expect from the
mid-2010s deep RL lineage.

### 2017: The Breakout Mainstream Year

- `PPO`
- `C51`
- `QR-DQN`
- `NoisyNet DQN`
- `Rainbow DQN`
- `HER`

**Package meaning:** this is the strongest single year for algorithms that are
still expected in general-purpose libraries.

### 2018: Strong Continuous Control And Distributed Scale

- `SAC`
- `TD3`
- `IQN`
- `IMPALA`
- `Ape-X`
- `MPO`

**Package meaning:** continuous-control quality jumped sharply here, and
distributed learner-actor designs became hard to ignore.

### 2019: Offline RL Becomes A First-Class Package Concern

- `BCQ`
- `BEAR`
- `Dreamer`
- `R2D2`
- `AWR`
- `TQC`

**Package meaning:** if the package wants to look serious beyond online control,
this is the year that forces offline RL, recurrent replay, and stronger value
distribution baselines onto the roadmap.

### 2020: Practical Offline And Data-Efficient RL

- `CQL`
- `AWAC`
- `DrQ`
- `CURL`
- `PPG`
- `MOPO`

**Package meaning:** this year strongly reinforces the need for offline dataset
loading, reward transforms, evaluation cadence, and imitation-to-online ramps.

### 2021: Strong Offline Baselines And Sequence Modeling

- `IQL`
- `REDQ`
- `TD3+BC`
- `CRR`
- `Decision Transformer`
- `DrQ-v2`

**Package meaning:** this is where a package has to decide whether it will only
ship actor-critic families or also support sequence-model and data-efficient
pixel-control baselines.

### 2022: Model Predictive RL And Conservative Dataset Refinement

- `TD-MPC`
- `Cal-QL`
- `EDAC`
- `RLPD`
- `Diffuser`
- `Discrete SAC` as a practical package target

**Package meaning:** by this point, dataset quality control, stronger ensemble
critics, and planning-oriented world models become package-relevant.

### 2023: World Models Return To The Center

- `DreamerV3`
- `TD-MPC2`
- `XQL`
- `ReBRAC`
- sequence-model RL follow-ons to `Decision Transformer`
- offline-to-online hybrid stacks around `AWAC`, `IQL`, and `TD-MPC`

**Package meaning:** this is the earliest year where a world-model track becomes
realistic for a mainstream package, but only if the runtime and experiment stack
is already mature.

### 2024: Simplification And Frontier Implementation Wave

- `CrossQ`
- state-space / `Decision Mamba` style control models
- larger-scale `DreamerV3` implementation adoption
- larger-scale `TD-MPC2` implementation adoption
- world-model scaling stacks becoming package-relevant
- `APPO` / `IMPALA` renewed production relevance

**Package meaning:** the package should now think about what can actually be
maintained, not only what is paper-popular. Simpler low-tuning families such as
`CrossQ` matter a lot.

### 2025: Frontier Watchlist, Not Stable Core

These are package watchlist items rather than fixed promises:

- stronger world-model robustness variants
- lower-tuning offline-to-online actor-critic hybrids
- scalable sparse-goal and robotics relabeling stacks
- state-space / sequence decision-model variants
- simplified critic-regularized continuous-control families
- safety- and constraint-aware continuous-control baselines

**Package meaning:** treat 2025 as a design watchlist year, not a commit-to-six
algorithms year.

### 2026: Current Watchlist On March 12, 2026

This year is still too early to lock into a canonical set of package targets.
The correct move is to keep a watchlist:

- `CrossQ`-style low-tuning continuous-control implementations
- stronger world-model training and evaluation stacks
- hybrid model-based plus offline RL pipelines
- richer sparse-goal dataset and replay tooling
- sequence / state-space control models that prove reproducible
- benchmark and evaluation infrastructure that makes frontier methods comparable

**Package meaning:** in 2026, the stability bottleneck is often tooling and
reproducibility, not the lack of another trainer name.

## What This Means For `rl_training`

From the package's current state on **March 12, 2026**, the immediate offline
wave is no longer hypothetical. `AWR`, `MARWIL`, `BCQ`, `BEAR`, offline dataset mixing,
reward presets, schedule / budget controls, `CRR`, `Cal-QL`, `EDAC`, `RLPD`,
`XQL`, and `ReBRAC` have been moved onto the current package surface.

`DrQ-v2` has also now moved onto the package surface in a narrow v1 form for
pixel observations and continuous actions.

For the current `AWR` wave, coverage files have also been added but test
execution is still intentionally deferred until explicitly requested.

For the current `MARWIL` wave, coverage files have also been added but test
execution is still intentionally deferred until explicitly requested.

For the current `EDAC` wave, coverage files have been added but test execution
is still intentionally deferred until explicitly requested.

For the current `RLPD` wave, coverage files have also been added but test
execution is still intentionally deferred until explicitly requested.

That means the best next intake order is now:

1. stronger benchmark / validation presets for the offline and pixel-control waves
2. `IMPALA` / `APPO`
3. `DreamerV3` / `TD-MPC2`

The point is not to follow history literally. It is to add the algorithms that
users still compare against while maturing the shared package surface each wave
needs.

## Recommended Packaging Split

- **Core now:** `PPO`, `A2C`, `DQN` family, `DDPG`, `SAC`, `TD3`, `REDQ`,
  `TQC`, `IQL`, `CQL`, `Cal-QL`, `EDAC`, `RLPD`, `XQL`, `TD3+BC`, `BC`, `AWR`,
  `AWAC`, `MARWIL`, `BCQ`, `BEAR`, `CRR`, `ReBRAC`, `HER`, `TRPO`, `Discrete SAC`,
  `CrossQ`, `DrQ-v2`
- **Core next:** stronger preset / validation coverage, then practical follow-on
  baselines around distributed actor-learners and stronger pixel-control
  validation
- **Contrib or staged entry:** `IMPALA`, `APPO`
- **Frontier track after infra matures:** `DreamerV3`, `TD-MPC2`, sequence-model RL

## Research Anchors Used

Mainstream library surfaces:

- Stable-Baselines3 docs: https://stable-baselines3.readthedocs.io/en/master/
- SB3 Contrib docs: https://sb3-contrib.readthedocs.io/en/master/
- RLlib algorithms docs: https://docs.ray.io/en/latest/rllib/rllib-algorithms.html
- d3rlpy algorithm references: https://d3rlpy.readthedocs.io/en/v0.41/references/algos.html

Canonical papers and primary sources:

- Deterministic Policy Gradient: https://proceedings.mlr.press/v32/silver14.html
- DQN: https://www.nature.com/articles/nature14236
- TRPO: https://proceedings.mlr.press/v37/schulman15.html
- DDPG: https://arxiv.org/abs/1509.02971
- PPO: https://arxiv.org/abs/1707.06347
- HER: https://arxiv.org/abs/1707.01495
- SAC: https://proceedings.mlr.press/v80/haarnoja18b.html
- TD3: https://proceedings.mlr.press/v80/fujimoto18a.html
- BCQ: https://arxiv.org/abs/1812.02900
- BEAR: https://arxiv.org/abs/1906.00949
- AWR: https://arxiv.org/abs/1910.00177
- AWAC: https://arxiv.org/abs/2006.09359
- CRR: https://arxiv.org/abs/2006.15134
- IQL: https://arxiv.org/abs/2110.06169
- Cal-QL: https://arxiv.org/abs/2303.05479
- EDAC: https://arxiv.org/abs/2110.01548
- RLPD: https://arxiv.org/abs/2208.07544
- DrQ-v2: https://arxiv.org/abs/2107.09645
- ReBRAC: https://arxiv.org/abs/2305.09836
- Decision Transformer: https://arxiv.org/abs/2106.01345
- TD-MPC: https://proceedings.mlr.press/v162/hansen22a.html
- DreamerV3: https://arxiv.org/abs/2301.04104
- TD-MPC2: https://arxiv.org/abs/2310.16828
- CrossQ: https://openreview.net/forum?id=PczQtTsTIX
