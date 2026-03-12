# RL Training

RL Training is being built as a real reinforcement learning training package,
not a single-algorithm demo or a lightweight toy scaffold. The target is a
Python-first library that can grow toward the capability level of mature RL
packages while staying modular, readable, and practical to extend.

The package is intended to provide a stable core API for multiple algorithm
families, including on-policy and off-policy methods. The long-term direction
includes support for algorithms such as PPO, DQN, Double DQN, Dueling DQN,
TRPO, Noisy DQN, Prioritized DQN, Rainbow DQN, C51-DQN, N-Step DQN, QR-DQN,
DDPG, SAC, Discrete SAC, CrossQ, DrQ-v2, TD3, BC, AWR, AWAC, MARWIL, BCQ,
BEAR, CRR, IQL, CQL, Cal-QL, EDAC, RLPD, XQL, ReBRAC, and related training patterns,
together with reusable abstractions for policies, rollout buffers, replay buffers, collectors,
trainers, evaluators, and experiment management.

From the product side, RL Training is meant to cover the full workflow needed
for serious experimentation and training operations:

- vectorized environment support
- on-policy and off-policy data handling
- checkpointing, resume, and run directory management
- evaluation, logging, and TensorBoard integration
- structured configuration and CLI-oriented experiment control
- clean extension points for future async, distributed, and offline RL modes

The repository has already moved past a single PPO vertical slice. The current
state includes multiple discrete and continuous-control algorithms, Atari image
pipelines, a `contrib` layer for recurrent PPO, a `zoo/` layer for packaged
presets, and the first offline / goal-conditioned expansion wave through
`BC`, `AWR`, `AWAC`, `MARWIL`, `BCQ`, `BEAR`, `CRR`, `Cal-QL`, `EDAC`,
`RLPD`, `XQL`, `ReBRAC`, and `HER`, plus the next on-policy expansion through `TRPO`, and the first pixel-based
continuous-control expansion through `DrQ-v2`.

The next milestone is no longer “make it look like a package”. It is to make
the package behave like a credible mainstream RL toolkit under broader training
regimes by widening the offline wave that now includes `BCQ`, `BEAR`, `CRR`,
`AWR`, `MARWIL`, `Cal-QL`, `EDAC`, `RLPD`, `XQL`, plus the newer `ReBRAC` follow-on,
alongside stronger offline data-mixing and reward-preset flows, and clearer training budget /
early-stopping controls.

The design draws from Stable-Baselines3, RL Baselines3 Zoo, CleanRL, and
Tianshou: a stable algorithm core, modular runtime boundaries, a dedicated
experiment layer, and readable reference implementations. The goal is to become
a serious RL package with a credible growth path, rather than stopping at a
minimal training example.

## Current Direction

The repository already includes multiple classic-control and continuous-control
algorithms behind a shared `train / eval / resume` workflow. The current
expansion path is aimed at a more mainstream package shape:

- a stable `core + contrib + zoo` package layout
- Atari-ready environment wrappers and image observations
- vector-observation `TRPO` alongside CNN-backed `PPO` and `DQN` training paths
- a discrete actor-critic lane through `Discrete SAC`
- a lower-tuning continuous-control lane through `CrossQ`
- a pixel-observation continuous-control lane through `DrQ-v2`
- a `contrib` layer for algorithms such as recurrent PPO
- a `zoo/` layer for presets, benchmark manifests, and run recipes
- offline dataset loading and reward transforms that work across trainers
- goal-conditioned replay relabeling through `HER`
- canonical offline RL baselines including `AWR`, `MARWIL`, `BCQ`, `BEAR`, `CRR`,
  `Cal-QL`, `EDAC`, `RLPD`, `XQL`, and `ReBRAC`

## Install

Core package:

```bash
pip install -e .
```

Atari extras:

```bash
pip install -e ".[atari]"
```

Offline dataset extras:

```bash
pip install -e ".[offline]"
```

The packaged CLI bundles the repository's documented `configs/` and `zoo/`
YAML assets, so preset paths continue to resolve after installation.

## Start Here

For stable classic-control and continuous-control training, stay on the core
surface:

```bash
rl-training train --config configs/ppo/cartpole.yaml
rl-training train --config configs/crossq/pendulum.yaml
rl-training train --config configs/crr/pendulum.yaml
rl-training train --config configs/awr/pendulum.yaml
rl-training train --config configs/marwil/pendulum.yaml
rl-training train --config configs/cal_ql/pendulum.yaml
rl-training train --config configs/edac/pendulum.yaml
rl-training train --config configs/rlpd/pendulum.yaml
rl-training train --config configs/xql/pendulum.yaml
rl-training train --config configs/rebrac/pendulum.yaml
rl-training train --config configs/drqv2/pendulum_pixels.yaml
rl-training train --config configs/discrete_sac/cartpole.yaml
rl-training train --config configs/trpo/cartpole.yaml
rl-training eval --checkpoint runs/<run-id>/checkpoints/step_<n>.pt
rl-training resume --checkpoint runs/<run-id>/checkpoints/step_<n>.pt
```

For recurrent or higher-complexity extensions, stay explicit and go through
`rl_training.contrib`:

```bash
rl-training train --config configs/recurrent_ppo/breakout_atari.yaml
rl-training eval --checkpoint runs/<run-id>/checkpoints/step_<n>.pt
rl-training resume --checkpoint runs/<run-id>/checkpoints/step_<n>.pt
```

For curated benchmark recipes and reproducible launch commands, go through the
zoo layer:

```bash
rl-training zoo --format commands
rl-training train --config zoo/atari/dqn_breakout.yaml
```

## CLI

After installing the package, the CLI entrypoint is:

```bash
rl-training train --config configs/ppo/cartpole.yaml
rl-training train --config configs/crossq/pendulum.yaml
rl-training train --config configs/crr/pendulum.yaml
rl-training train --config configs/awr/pendulum.yaml
rl-training train --config configs/marwil/pendulum.yaml
rl-training train --config configs/cal_ql/pendulum.yaml
rl-training train --config configs/edac/pendulum.yaml
rl-training train --config configs/rlpd/pendulum.yaml
rl-training train --config configs/xql/pendulum.yaml
rl-training train --config configs/rebrac/pendulum.yaml
rl-training train --config configs/drqv2/pendulum_pixels.yaml
rl-training train --config configs/discrete_sac/cartpole.yaml
rl-training train --config configs/trpo/cartpole.yaml
rl-training eval --checkpoint runs/<run-id>/checkpoints/step_<n>.pt
rl-training resume --checkpoint runs/<run-id>/checkpoints/step_<n>.pt
rl-training zoo --format commands
```

The module form works as well:

```bash
python -m rl_training train --config configs/dqn/cartpole.yaml
```

Training and resume commands print the resolved `run_dir`, `checkpoint_path`,
and final metrics so the CLI can be used directly in shell workflows.

## Atari Presets

Atari configs use `env_kwargs.wrappers.atari` to keep wrapper settings outside
the raw `gym.make(...)` constructor kwargs.

Examples:

```bash
rl-training train --config configs/dqn/breakout_atari.yaml
rl-training train --config configs/ppo/breakout_atari.yaml
rl-training train --config configs/recurrent_ppo/breakout_atari.yaml
```

Reference scripts:

- `examples/dqn_breakout_atari_reference.py`
- `examples/ppo_breakout_atari_reference.py`
- `examples/recurrent_ppo_breakout_atari_reference.py`

Installed package module form:

```bash
python -m rl_training.examples.dqn_breakout_atari_reference --total-timesteps 1024
python -m rl_training.examples.ppo_breakout_atari_reference --total-timesteps 1024
python -m rl_training.examples.recurrent_ppo_breakout_atari_reference --total-timesteps 1024
```

The first benchmark manifest lives at `zoo/atari/benchmark.yaml`.

Zoo preset files can also be passed directly to the CLI:

```bash
rl-training train --config zoo/atari/dqn_breakout.yaml
```

To enumerate the current zoo commands:

```bash
rl-training zoo --format commands
rl-training-zoo --format commands
python -m rl_training zoo --format commands
```

## Pixel Control

`DrQ-v2` uses generic pixel wrappers rather than an algorithm-specific
environment fork. For classic Gymnasium control tasks, keep render settings in
`env_kwargs` and pixel preprocessing in `env_kwargs.wrappers.pixels`.

Example starter config:

```yaml
algo: drqv2
env_id: Pendulum-v1
seed: 7
total_timesteps: 5000
output_dir: runs
eval_episodes: 5
algo_kwargs:
  buffer_capacity: 50000
  batch_size: 128
  learning_starts: 256
  features_dim: 256
  actor_hidden_sizes: [256, 256]
  critic_hidden_sizes: [256, 256]
  learning_rate: 0.0001
  gamma: 0.99
  tau: 0.01
  policy_delay: 2
  augmentation_pad: 4
  exploration_noise: 0.1
  exploration_noise_clip: 0.3
env_kwargs:
  render_mode: rgb_array
  wrappers:
    pixels:
      resize_shape: [84, 84]
      frame_stack: 3
      channel_first: true
```

The packaged starter preset for this path is
`configs/drqv2/pendulum_pixels.yaml`.

## Offline Data

Offline trainers can now pull transitions from more than the built-in random
dataset generator. The package-level direction is:

- `algo: bc` for the first supervised imitation baseline on top of this data path
- `algo: awr` for return-weighted offline actor regression on the same dataset path
- `algo: awac` for advantage-weighted offline actor-critic on the same data path
- `algo: marwil` for RLlib-style weighted offline imitation on the same actor/value data path
- `algo: bcq` for constrained offline policy improvement on the same data path
- `algo: bear` for support-matching offline actor-critic on the same data path
- `algo: crr` for critic-regularized offline policy regression on the same data path
- `algo: cal_ql` for calibrated conservative Q-learning on the same data path
- `algo: edac` for ensemble-diversified offline actor-critic on the same REDQ-style data path
- `algo: rlpd` for prior-data offline-to-online actor-critic on top of the same `SAC` lane
- `algo: xql` for extreme value regression on the same offline actor-critic path
- `algo: rebrac` for behavior-regularized TD3-style offline actor-critic on the same data path
- `algo: her` for goal-conditioned replay relabeling on sparse-reward tasks
- `dataset_kind: random` for synthetic smoke data
- `dataset_kind: npz` for NumPy transition dumps
- `dataset_kind: pt` for PyTorch dictionary payloads
- `dataset_kind: minari` for Farama Minari datasets when the optional extra is installed
- `dataset_mix` for blending multiple offline sources into one training dataset
- optional `next_actions` payloads for behavior-regularized offline learners
- optional `returns_to_go` payloads for calibrated conservative learners

Minimal `AWR` example:

```yaml
algo: awr
env_id: Pendulum-v1
seed: 7
total_timesteps: 20000
output_dir: runs/awr-pendulum
eval_episodes: 5
algo_kwargs:
  dataset_kind: npz
  dataset_path: data/pendulum_medium.npz
  batch_size: 256
  hidden_sizes: [256, 256]
  learning_rate: 0.0003
  gamma: 0.99
  beta: 1.0
  max_weight: 20.0
```

Minimal `MARWIL` example:

```yaml
algo: marwil
env_id: Pendulum-v1
seed: 7
total_timesteps: 20000
output_dir: runs/marwil-pendulum
eval_episodes: 5
algo_kwargs:
  dataset_kind: npz
  dataset_path: data/pendulum_medium.npz
  batch_size: 256
  hidden_sizes: [256, 256]
  learning_rate: 0.0003
  gamma: 0.99
  beta: 1.0
  vf_coeff: 1.0
  moving_average_sqd_adv_norm_start: 100.0
  moving_average_sqd_adv_norm_update_rate: 0.01
```

Example:

```yaml
algo: iql
env_id: Pendulum-v1
seed: 7
total_timesteps: 20000
output_dir: runs/iql-pendulum
eval_episodes: 5
algo_kwargs:
  dataset_kind: npz
  dataset_path: data/pendulum_medium.npz
  normalize_dataset_actions: true
  reward_scale: 0.1
  reward_shift: 0.0
  reward_clip_min: -10.0
  reward_clip_max: 10.0
  batch_size: 256
  hidden_sizes: [256, 256]
```

For Minari datasets:

```yaml
algo_kwargs:
  dataset_kind: minari
  dataset_id: hopper-medium-v0
  dataset_download: true
```

For mixed offline datasets:

```yaml
algo_kwargs:
  dataset_mix:
    - kind: npz
      dataset_path: data/pendulum_medium.npz
      weight: 0.7
    - kind: pt
      dataset_path: data/pendulum_expert.pt
      weight: 0.3
  dataset_mix_size: 200000
  dataset_mix_seed: 17
```

Packaged starter configs now include `configs/bc/pendulum.yaml`,
`configs/awr/pendulum.yaml`, `configs/awac/pendulum.yaml`,
`configs/marwil/pendulum.yaml`,
`configs/bcq/pendulum.yaml`,
`configs/bear/pendulum.yaml`, `configs/cal_ql/pendulum.yaml`,
`configs/crr/pendulum.yaml`, `configs/edac/pendulum.yaml`, `configs/rlpd/pendulum.yaml`,
`configs/xql/pendulum.yaml`,
`configs/rebrac/pendulum.yaml`, `configs/her/point_goal.yaml`, and
`configs/drqv2/pendulum_pixels.yaml`.

The first packaged `CRR` path is intentionally narrow: vector observations,
continuous `Box` actions, and offline datasets only. Pixel observations,
sequence models, and distributed runtime variants stay out of the v1 scope.

The first packaged `AWR` path is also intentionally narrow: vector
observations, continuous `Box` actions, and offline datasets only, with value
regression against discounted returns-to-go and exponentiated
advantage-weighted behavior cloning in the actor update.

The first packaged `MARWIL` path is likewise intentionally narrow: vector
observations, continuous `Box` actions, and offline datasets only, with value
regression against discounted returns-to-go and RLlib-style advantage-weighted
behavior cloning normalized by a running squared-advantage scale.

The first packaged `ReBRAC` path is also intentionally narrow: vector
observations, continuous `Box` actions, and offline datasets only, implemented
as a behavior-regularized follow-on to the existing `TD3+BC` lane.

When an offline dataset carries `next_actions`, the package now preserves that
field through loading, mixing, and batch sampling so behavior-regularized
algorithms can regularize next-state policy targets without a separate data
pipeline.

The first packaged `Cal-QL` path is likewise intentionally narrow: vector
observations, continuous `Box` actions, and offline datasets only, implemented
as a calibrated follow-on to the existing `CQL` lane.

When an offline dataset carries `returns_to_go`, the package now preserves
that field through loading, mixing, and batch sampling. `Cal-QL` also derives
discounted returns-to-go automatically from the processed reward stream when a
dataset does not provide them explicitly.

The first packaged `EDAC` path is likewise intentionally narrow: vector
observations, continuous `Box` actions, and offline datasets only, implemented
as an ensemble-diversified follow-on to the existing `REDQ` lane with a fixed
entropy coefficient in v1.

The first packaged `RLPD` path is likewise intentionally narrow: vector
observations, continuous `Box` actions, single-process online collection, and
prior data loaded from the existing offline dataset stack, implemented as an
offline-to-online follow-on to the existing `SAC` lane through offline
pretraining plus mixed offline/online update batches.

The first packaged `XQL` path is likewise intentionally narrow: vector
observations, continuous `Box` actions, and offline datasets only, implemented
as an extreme value follow-on to the existing `IQL` lane.

For the current `AWR` wave, package-facing tests have been added but are still
intentionally not executed until test execution is explicitly requested.

For the current `MARWIL` wave, package-facing tests have also been added but are
still intentionally not executed until test execution is explicitly requested.

For the current `EDAC` wave, package-facing tests have been added but are still
intentionally not executed until test execution is explicitly requested.

For the current `RLPD` wave, package-facing tests have also been added but are
still intentionally not executed until test execution is explicitly requested.

## Goal-Conditioned Training

The first packaged `HER` path targets goal-conditioned continuous control
through a DDPG backend plus future-goal replay relabeling.

The repository also ships a built-in sparse-reward point-goal environment so
the package has a stable goal-conditioned reference task without external
robotics dependencies:

```yaml
algo: her
env_id: RL-PointGoal1D-v0
seed: 7
total_timesteps: 20000
output_dir: runs/her-point-goal
eval_episodes: 5
algo_kwargs:
  buffer_capacity: 50000
  batch_size: 256
  learning_starts: 256
  her_ratio: 0.8
  goal_selection_strategy: future
  eval_interval: 1000
```

Programmatic usage goes through the same package API surface:

```python
from rl_training import HER, TrainConfig

algo = HER(
    TrainConfig(
        algo="her",
        env_id="RL-PointGoal1D-v0",
        seed=7,
        total_timesteps=20000,
        output_dir="runs/her-point-goal",
    )
)
result = algo.learn()
```

## Reward Wrappers

Generic reward transforms are configured under `env_kwargs.wrappers.reward`,
separate from Atari preprocessing:

```yaml
env_kwargs:
  wrappers:
    reward:
      scale: 0.1
      shift: 0.0
      clip: [-1.0, 1.0]
```

That wrapper path is intended for reusable reward shaping such as scaling,
shifting, and clipping without changing trainer code.

Named presets are also supported for common RL transforms:

```yaml
env_kwargs:
  wrappers:
    reward:
      preset: sign_clip
```

Current presets include `sign_clip`, `clip_1`, and `sparse_goal_zero_one`.

## Training Controls

Trainers now support evaluation cadence and early-stopping rules through
`algo_kwargs`:

```yaml
algo_kwargs:
  eval_interval: 1000
  early_stopping:
    metric: eval_return_mean
    mode: max
    patience: 5
    min_delta: 1.0
    min_steps: 5000
```

The current first-class early-stopping path is wired into `A2C`, `PPO`,
`TRPO`, `Discrete SAC`, `CrossQ`, `DrQ-v2`,
`RecurrentPPO`, `DQN`, `DDPG`, `SAC`, `TD3`, `REDQ`, `TQC`, `BC`, `AWR`, `AWAC`, `MARWIL`,
`BCQ`, `BEAR`, `CRR`, `Cal-QL`, `EDAC`, `RLPD`, `XQL`, `ReBRAC`, `HER`, `IQL`, `CQL`, and `TD3+BC`.

Offline trainers also support simple update-budget and learning-rate schedule
controls:

```yaml
algo_kwargs:
  max_epochs: 100
  max_updates: 1000
  warmup_steps: 100
  learning_rate_schedule:
    type: cosine
    start: 1.0
    end: 0.1
```

The current schedule kinds are `constant`, `linear`, and `cosine`. Metrics now
track `epoch`, `update_count`, `lr_scale`, and the resolved `learning_rate`
for the trainer loop.

## Contrib

Higher-complexity algorithms can live under `rl_training.contrib` without
changing the stable core API surface.

The root package keeps that split explicit: stable algorithms stay under
`rl_training`, while recurrent and experimental extensions are accessed through
`rl_training.contrib`.

```python
from rl_training.contrib import RecurrentPPO
from rl_training.experiment.config import TrainConfig

algo = RecurrentPPO(TrainConfig(...))
result = algo.learn()
```

## Package Layout

Use the package layers deliberately:

- `core`: stable train / eval / resume workflows for mainstream algorithms and
  environments that fit the main runtime.
- `contrib`: algorithms such as `RecurrentPPO` that add state semantics or
  experimental complexity without forcing those concerns into every core path.
- `zoo`: named presets, manifests, and benchmark launch recipes that sit on top
  of the runtime instead of reimplementing it.

That split is what lets the repository grow toward more algorithms without
turning into a pile of disconnected scripts.
