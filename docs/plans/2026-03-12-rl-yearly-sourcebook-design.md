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

- `DQN` - already implemented
- `Deterministic Policy Gradient` - still deferred as a separately named historical trainer
- replay-buffered deep value learning - already implemented as core runtime infrastructure
- target-network stabilized deep Q-learning - already implemented across the DQN family
- CNN-based pixel-control value networks - already implemented through the Atari/CNN lane

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

- `A2C` - already implemented
- `Dueling DQN` - already implemented
- `NAF` - implemented in the March 13, 2026 yearly-gap batch
- `DRQN` - implemented on March 13, 2026 as a narrow v1 recurrent replay baseline for discrete vector observations
- `ACER` - deferred until the runtime can support a more faithful on/off-policy actor-critic lane
- `UNREAL` - deferred until auxiliary-task and recurrent state handling are stronger

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
- `OpenAI ES` - implemented on March 13, 2026 as a narrow synchronous v1 vector-observation continuous-action evolution-strategy baseline

**Package meaning:** this is the strongest single year for algorithms that are
still expected in general-purpose libraries.

### 2018: Strong Continuous Control And Distributed Scale

- `ARS` - implemented on March 13, 2026 as a narrow synchronous v1 vector-observation continuous-action parameter-search baseline
- `SAC` - already implemented
- `TD3` - already implemented
- `IQN` - already implemented
- `D4PG` - implemented in the March 13, 2026 yearly-gap batch as a narrow non-distributed baseline
- `PETS` - implemented on March 13, 2026 as a narrow single-environment vector-observation continuous-action ensemble-dynamics plus CEM MPC planning baseline
- `IMPALA` - implemented on March 13, 2026 as a narrow synchronous v1 vector-observation discrete-action baseline with `V-trace` updates, without distributed actors or learner queues
- `Ape-X` - deferred until distributed replay and actor orchestration exist
- `MPO` - still deferred because a narrow approximation would risk misleading naming

**Package meaning:** continuous-control quality jumped sharply here, and
distributed learner-actor designs became hard to ignore.

### 2019: Offline RL Becomes A First-Class Package Concern

- `BCQ`
- `BEAR`
- `Dreamer`
- `R2D2` - implemented on March 13, 2026 as a narrow non-distributed v1 with prioritized recurrent replay and n-step returns
- `AWR`
- `TQC`

**Package meaning:** if the package wants to look serious beyond online control,
this is the year that forces offline RL, recurrent replay, and stronger value
distribution baselines onto the roadmap.

### 2020: Practical Offline And Data-Efficient RL

- `CQL`
- `AWAC`
- `DrQ` - implemented on March 13, 2026 as a narrow v1 pixel-observation continuous-control baseline with random-shift augmentation and SAC-style stochastic updates
- `CURL` - implemented on March 13, 2026 as a narrow v1 pixel-observation continuous-control follow-on with auxiliary contrastive representation updates on top of the existing DrQ-style lane
- `PPG` - implemented on March 13, 2026 as a narrow v1 vector-observation discrete-action on-policy follow-on with PPO-style policy phases and periodic auxiliary distillation phases
- `MOPO` - implemented on March 13, 2026 as a narrow v1 offline vector-observation continuous-action baseline with learned ensemble dynamics, short uncertainty-penalized synthetic rollouts, and SAC-style policy updates

**Package meaning:** this year strongly reinforces the need for offline dataset
loading, reward transforms, evaluation cadence, and imitation-to-online ramps.

### 2021: Strong Offline Baselines And Sequence Modeling

- `IQL`
- `REDQ`
- `TD3+BC`
- `CRR`
- `Decision Transformer` - implemented on March 13, 2026 as a narrow v1 offline vector-observation continuous-action baseline with fixed-length masked trajectory windows and return-conditioned transformer policy updates
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

## What This Means For `axiomrl`

From the package's current state on **March 13, 2026**, the immediate offline
wave is no longer hypothetical. `AWR`, `MARWIL`, `BCQ`, `BEAR`, offline dataset mixing,
reward presets, schedule / budget controls, `CRR`, `Cal-QL`, `EDAC`, `RLPD`,
`XQL`, and `ReBRAC` have been moved onto the current package surface.

`DrQ` has now landed as a narrow v1 pixel-observation continuous-control
baseline that reuses the package pixel wrappers but keeps the scope honest:
continuous actions only, fixed entropy coefficient, and no auxiliary contrastive
losses.

`DrQ-v2` has also now moved onto the package surface in a narrow v1 form for
pixel observations and continuous actions.

`CURL` has now landed as a narrow v1 pixel-observation continuous-control
representation-learning follow-on that reuses the package pixel wrappers and
DrQ-style replay runtime while adding a contrastive auxiliary encoder loss.

`PPG` has now landed as a narrow v1 on-policy follow-on in the discrete vector
control lane, reusing PPO-style rollouts but adding periodic auxiliary value
regression and policy-distillation phases.

`NAF` has now landed as a historical continuous-control value baseline on top
of the existing replay-buffer actor-critic lane.

`DRQN` has now landed as a narrow v1 recurrent replay baseline for discrete
vector-observation environments, without claiming Atari/image support yet.

`R2D2` has now landed as a narrow non-distributed v1 follow-on that adds
prioritized recurrent replay and n-step targets on top of the new recurrent
value-learning lane, without claiming burn-in or distributed actors yet.

`MOPO` has now landed as a narrow v1 model-based offline RL baseline for
vector observations and continuous actions, using learned ensemble dynamics,
uncertainty-penalized short synthetic rollouts, and SAC-style policy updates
without claiming terminal-model learning, MPC planning, or online collection.

`PETS` has now landed as a narrow v1 online model-based control baseline for
vector observations and continuous actions, using learned ensemble dynamics and
cross-entropy MPC planning without claiming image observations, latent world
models, or distributed actor infrastructure.

`D4PG` has now landed as a narrow non-distributed 2018 follow-on that reuses
the current deterministic actor path plus categorical distributional critics.

`ARS` has now landed as a narrow synchronous v1 continuous-control baseline
that trains a deterministic MLP policy through mirrored parameter
perturbations, without claiming observation normalization, parallel workers, or
distributed search orchestration.

`OpenAI ES` has now landed as a narrow synchronous v1 search-based follow-on
that reuses the same deterministic policy lane but updates parameters through
centered-rank mirrored perturbation utilities, without claiming distributed
gradient aggregation or population infrastructure.

`IMPALA` has now landed as a narrow synchronous v1 vector-observation
discrete-action baseline that reuses the current actor-critic lane while
adding `V-trace` policy/value targets, without claiming distributed
actor-learner infrastructure yet.

`APPO` has now landed as a narrow synchronous v1 follow-on that keeps the same
vector-observation discrete-action scope while combining `V-trace` targets
with PPO-style clipped policy updates against stored behavior-policy
log-probabilities, without claiming distributed actors or asynchronous learner
queues yet.

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
2. stronger benchmark / validation presets for the actor-critic distributed-style lane
3. stronger benchmark / validation presets for the new search-based continuous-control lane
4. `DreamerV3` / `TD-MPC2`

The point is not to follow history literally. It is to add the algorithms that
users still compare against while maturing the shared package surface each wave
needs.

## Recommended Packaging Split

- **Core now:** `PPO`, `PPG`, `A2C`, `DQN` family, `DDPG`, `NAF`, `DRQN`, `R2D2`, `D4PG`, `SAC`, `TD3`, `REDQ`,
  `TQC`, `IQL`, `CQL`, `Cal-QL`, `EDAC`, `RLPD`, `XQL`, `TD3+BC`, `BC`, `AWR`,
  `AWAC`, `MARWIL`, `BCQ`, `BEAR`, `CRR`, `ReBRAC`, `HER`, `TRPO`, `IMPALA`, `APPO`, `Discrete SAC`,
  `CrossQ`, `DrQ`, `CURL`, `DrQ-v2`, `ARS`, `OpenAI ES`
- **Core next:** stronger preset / validation coverage, then practical follow-on
  baselines around distributed actor-learners and stronger pixel-control
  validation
- **Contrib or staged entry:** future async/distributed `APPO` refresh if the runtime grows beyond synchronous rollouts
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
- DrQ: https://arxiv.org/abs/2004.13649
- DrQ-v2: https://arxiv.org/abs/2107.09645
- ReBRAC: https://arxiv.org/abs/2305.09836
- Decision Transformer: https://arxiv.org/abs/2106.01345
- TD-MPC: https://proceedings.mlr.press/v162/hansen22a.html
- DreamerV3: https://arxiv.org/abs/2301.04104
- TD-MPC2: https://arxiv.org/abs/2310.16828
- CrossQ: https://openreview.net/forum?id=PczQtTsTIX
