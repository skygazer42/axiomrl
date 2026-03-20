# Configuration Schema

This document describes the *core* YAML schema that powers `axiomrl train`.
Algorithm-specific knobs live under `algo_kwargs` and vary by trainer; use the
files under `configs/` (and `zoo/`) as the canonical examples.

## Two supported config shapes

### 1) Full training config (direct `TrainConfig`)

A full config provides all required top-level keys:

```yaml
algo: ppo
env_id: CartPole-v1
seed: 42
total_timesteps: 100000
output_dir: runs
num_envs: 8
eval_episodes: 5
tags:
  - demo
algo_kwargs:
  learning_rate: 0.0003
env_kwargs: {}
benchmark: {}
```

### 2) Linked preset config (`config:` include)

Zoo presets can be lightweight overlays that point at another config via
`config:`. The loader resolves the linked config first, then applies the preset
overlay and manifest defaults.

Example (`zoo/atari/dqn_breakout.yaml`):

```yaml
name: dqn_breakout
config: configs/dqn/breakout_atari.yaml
algorithm: dqn
env_id: ALE/Breakout-v5
```

The final resolved payload must still satisfy the full `TrainConfig` schema
after resolution.

## Core `TrainConfig` keys

Required:

- `algo` (str, non-empty) — algorithm id, e.g. `ppo`, `dqn`.
- `env_id` (str, non-empty) — Gymnasium environment id.
- `seed` (int, >= 0)
- `total_timesteps` (int, >= 1)
- `output_dir` (str path) — base directory where run directories are created.

Optional:

- `execution_backend` (str, default: `local_sync`)
- `device` (str, default: `auto`)
- `num_envs` (int, default: `1`, must be >= 1)
- `eval_episodes` (int, default: `5`, must be >= 1)
- `log_interval` (int, default: `1`, must be >= 1)
- `checkpoint_interval` (int, default: `1`, must be >= 1)
- `tags` (list[str], default: `[]`) — free-form run tags.
- `algo_kwargs` (mapping, default: `{}`) — algorithm-specific parameters.
- `env_kwargs` (mapping, default: `{}`) — environment factory/wrapper parameters.
- `benchmark` (mapping, default: `{}`) — optional benchmarking metadata.

### `benchmark` (common keys)

`benchmark` is an intentionally flexible mapping that is:

- used to drive multi-seed sweeps (via `benchmark.seeds`)
- recorded into run artifacts (`metadata.json`)
- consumed by Zoo reporting/leaderboard commands

Common keys used by the runtime include:

- `seeds` (list[int]) — run a sweep and write `benchmark-summary.json`.
- `best_metric` (str, default: `eval_return_mean`) — best-checkpoint metric name.
- `best_metric_mode` (`max` | `min`, default: `max`)
- `score_normalization` (mapping | false) — when enabled, adds
  `eval_human_normalized_score` metrics.
- `suite`, `preset_name`, `protocol_name` (str) — Zoo/benchmark identity fields.

## CLI overrides

`axiomrl train` supports a few common overrides without editing YAML:

- `--output-dir <path>` overrides `output_dir`
- `--execution-backend <name>` overrides `execution_backend`
- `--total-timesteps <int>` overrides `total_timesteps`
- `--num-envs <int>` overrides `num_envs`
- `--eval-episodes <int>` overrides `eval_episodes`
- `--seeds 1,2,3` runs a benchmark sweep by setting `benchmark.seeds`

## Inspect resolved config

Use `axiomrl config --config <path>` to print the resolved `TrainConfig` payload
(including linked preset and manifest defaults). Add `--format yaml` if you
prefer YAML output.
