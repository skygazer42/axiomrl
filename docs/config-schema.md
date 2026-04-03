# Configuration Schema

This document describes the *core* YAML schema that powers `axiomrl train`.
Algorithm-specific knobs live under `algo_kwargs` and vary by trainer; use the
files under `configs/` (and `zoo/`) as the canonical examples.

`axiomrl tune` uses a separate study schema that wraps one resolved training
config plus a search space.

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

## Study config (`axiomrl tune`)

Study configs are YAML mappings with these top-level keys:

- `base_config` (required) — path to a standard train config.
- `output_dir` (optional) — root directory that will contain `<study_name>/`.
- `study` (required mapping) — execution metadata.
- `search_space` (required mapping) — tunable parameter definitions.

Example:

```yaml
base_config: configs/ppo/cartpole.yaml
output_dir: runs/studies

study:
  name: ppo_cartpole_tune
  backend: native
  sampler: grid
  objective:
    metric: global_step
    mode: max

search_space:
  total_timesteps:
    type: int
    low: 32000
    high: 64000
    step: 32000
  algo_kwargs.learning_rate:
    type: categorical
    values: [0.0003, 0.001]
```

### `study` keys

- `name` (required) — study directory name.
- `backend` (`native` | `optuna`, default: `native`)
- `sampler`
  - `native`: `random` or `grid`
  - `optuna`: `random` or `tpe`
- `num_trials`
  - required for `native/random` and `optuna`
  - not allowed for `native/grid`
- `seed` (int, default: `0`)
- `fail_fast` (bool, default: `false`)
- `objective.metric` (required str)
- `objective.mode` (required `min` | `max`)

### Search space rules

Supported search spec types:

- `categorical` with `values`
- `int` with `low`, `high`, and optional `step`
- `float` with `low`, `high`, and optional `step` / `log`

Supported target paths:

- top-level scalars: `total_timesteps`, `num_envs`, `eval_episodes`,
  `log_interval`, `checkpoint_interval`, `device`, `execution_backend`
- nested mappings: `algo_kwargs.*`, `env_kwargs.*`, `benchmark.*`

Explicitly forbidden:

- `algo`
- `env_id`
- `seed`
- `output_dir`
- `tags`
- `benchmark.seeds`

### Tune CLI overrides

`axiomrl tune` supports:

- `--config <study.yaml>` — start a study from config
- `--resume-study <study_dir>` — reload an existing study snapshot and continue
  any missing trial indices
- `--output-dir <path>` — override the study root when starting from `--config`
- `--backend native|optuna` — override the configured backend when starting from
  `--config`

### Study report CLI

`axiomrl tune-report` reads an existing study directory without running any
missing trials:

- `--study-dir <path>` — required study directory containing `study.json`
- `--report-output text|json|csv` — render human-readable text or
  machine-readable exports
- `--status all|completed|failed` — keep every trial, only successful trials,
  or only failed trials
- `--objective-at-least <value>` — keep only visible trials whose
  `objective_value` is greater than or equal to the given numeric threshold
- `--objective-at-most <value>` — keep only visible trials whose
  `objective_value` is less than or equal to the given numeric threshold
- `--duration-at-least <seconds>` — keep only visible trials whose derived
  wall-clock duration is greater than or equal to the given numeric threshold
- `--duration-at-most <seconds>` — keep only visible trials whose derived
  wall-clock duration is less than or equal to the given numeric threshold
- `--frontier-only` — keep only visible completed timed trials that remain on
  the objective-vs-duration Pareto frontier after the other filters are applied
- `--param <key=value>` — keep only visible trials whose serialized parameter
  value exactly matches the given value; repeat for AND-style filtering
- `--error <text>` — keep only visible trials whose error text exactly matches
  the given value
- `--error-contains <text>` — keep only visible trials whose error text
  contains the given case-insensitive substring; use only one error filter at a
  time
- `--error-type <name>` — keep only visible trials whose derived exception type
  matches the given value case-insensitively
- `--focus-param <name>` — expose one visible parameter's value buckets as a
  dedicated ordered summary block; text and CSV outputs also surface this block
  in human-readable / flattened forms
- `--focus-sort-by best-objective-value|mean-objective-value|completion-rate|incumbent-updates|mean-duration-seconds|value`
  — choose the ordering used inside the focused parameter bucket summary,
  including best-so-far update contribution and fastest mean wall-clock bucket
  duration
- `--focus-top-k <n>` — keep only the strongest `n` focused parameter buckets
  after the focused ordering is applied; focus-only flags require
  `--focus-param`
- `--sort-by trial-index|objective-value|duration-seconds` — order visible
  trials by index, objective metric, or trial wall-clock duration
- `--descending` — reverse the selected ordering
- `--top-k <n>` — keep only the first `n` visible trials after filtering and
  sorting
- `--export-configs-dir <path>` — export the current visible completed trials as
  ranked YAML configs plus a `manifest.json`
- `--output <path>` — also write the rendered report to disk

JSON study reports now expose selection analytics as
`selected_status_counts`, `selected_objective_summary`,
`selected_parameter_summaries`, `selected_parameter_incumbent_summaries`, and
`selected_parameter_value_summaries`.
Reports also expose `selected_duration_summary`, and each visible trial row also
includes `duration_seconds` when timestamps are available. When objective
threshold filters are active, trials with missing objective values are excluded
from the visible slice. When duration threshold filters are active, trials with
missing or invalid timestamps are excluded from the visible slice.
Reports also expose `selected_incumbent_trace`, a visible-slice best-so-far
timeline ordered by `trial_index`, plus per-trial
`selected_incumbent_trial_index`, `selected_incumbent_objective_value`, and
`selected_is_incumbent_update` fields.
Reports also expose `selected_incumbent_update_summary`, which rolls visible
incumbent updates up into update counts, first/latest update locations,
improvement magnitudes versus the previous incumbent, and visible trial spacing
between updates. Per-trial rows also expose
`selected_incumbent_update_improvement` and
`selected_incumbent_trials_since_previous_update`.
Reports also expose `selected_incumbent_staleness_summary`, which rolls the
visible incumbent timeline up into latest/max incumbent age over both visible
trial count and wall-clock seconds. Per-trial rows also expose
`selected_incumbent_age_trials` and `selected_incumbent_age_seconds`.
Reports also expose `selected_objective_duration_frontier`, which contains the
visible completed timed trials that are nondominated on objective value and
wall-clock duration, plus a per-trial `is_objective_duration_frontier` flag.
Failed visible trials are also grouped into `selected_error_summaries` so one
error string can report its count, visible/failed shares, and trial indices.
Reports also expose `selected_error_type_summaries` so failures can be grouped
by exception class with the distinct matched full error texts per type.
They also expose `selected_best_trial_index`, `selected_best_objective_value`, and
per-trial `selected_best_objective_delta`. When the serialized study config
contains a discrete search space, parameter summaries also include
`candidate_count` and `coverage_ratio`. Parameter value summaries also include
completion/failure rates, per-bucket timing fields (`timed_trials`,
`untimed_trials`, `min_duration_seconds`, `max_duration_seconds`,
`mean_duration_seconds`, `median_duration_seconds`), plus
`incumbent_updates` and `latest_incumbent_trial_index`, and rank fields keyed by
best and mean objective value. Reports also expose
`selected_parameter_effect_leaderboard`, which ranks parameters by the visible
spread in their bucket-level best/mean objective values and surfaces the
strongest/weakest observed values for each parameter. When `--focus-param` is used, reports also expose
`focused_parameter_name` plus `focused_parameter_value_summary`. A top-level
`search_efficiency_summary` rollup summarizes failure rate, best-vs-center
deltas, coverage extremes, and visible-slice convergence fields such as
`selected_trials_until_best`, `completed_trials_until_best`, and
`time_to_best_seconds`. Text and CSV renderers also expose these convergence
metrics as flattened `search_efficiency_*` fields. When config export is
requested, payloads also include `config_export_summary`. When focused ordering
is customized or truncated, `report_filters` also records `focus_sort_by` and
`focus_top_k`.
`selected_parameter_incumbent_summaries` rolls those value-level incumbent
updates up to the parameter level, including `contributing_values`,
`top_incumbent_value`, and `latest_incumbent_value`.
`selected_parameter_incumbent_leaderboard` exposes the same parameter-level
data as a sorted list for dashboards and spreadsheet workflows.
