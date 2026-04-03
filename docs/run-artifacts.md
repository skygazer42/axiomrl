# Run Artifacts & Metadata

This document describes what AxiomRL writes to disk during training, how to
interpret those artifacts, and which other tools consume them.

## Run directory layout

Each training run creates one run directory under `TrainConfig.output_dir` with
an id like:

```
<algo>__<env_id>__seed<seed>__<timestamp>
```

`env_id` is sanitized for filesystem safety (for example, `ALE/Breakout-v5`
becomes `ALE-Breakout-v5` in the directory name), while `metadata.json` still
records the original `env_id`.

Example:

```
runs/ppo__CartPole-v1__seed42__20260321-120102-123456
```

Within the run directory, the runtime writes:

- `metadata.json` — machine-readable run metadata and latest/best metrics.
- `config.yaml` — the serialized `TrainConfig` used for the run.
- `checkpoints/` — periodic checkpoint files, plus `best.pt` when configured.
- `tensorboard/` — TensorBoard event files.

## `metadata.json`

The metadata file is created at run start and updated as training progresses.
It is also the primary input for `axiomrl zoo --format report|leaderboard`
commands (they scan `runs/*/metadata.json`).

### Core identity fields (written at run start)

- `algo` (str)
- `env_id` (str)
- `seed` (int)
- `output_dir` (str) — absolute path to the run directory
- `benchmark` (object) — benchmark settings copied from `TrainConfig.benchmark`

### Reproducibility / context fields (written at run start)

- `created_at_utc` (str) — ISO timestamp
- `command` (list[str]) — argv captured from the training process
- `system` (object) — basic platform info (python version, platform string)
- `versions` (object) — dependency versions (e.g. torch/gymnasium/numpy)
- `git` (object)
  - `commit` (str | null) — best-effort `git rev-parse HEAD`
  - `is_dirty` (bool | null) — best-effort `git status --porcelain` check

### Progress fields (updated during training)

- `latest_checkpoint_path` (str)
- `latest_metrics` (object) — latest logged metrics snapshot
- `best_checkpoint` (object, optional) — best-checkpoint alias metadata

### Example

```json
{
  "algo": "ppo",
  "env_id": "CartPole-v1",
  "seed": 42,
  "output_dir": "F:/pythonproject/axiomrl/runs/ppo__CartPole-v1__seed42__...",
  "created_at_utc": "2026-03-21T04:01:02.123456+00:00",
  "command": ["axiomrl", "train", "--config", "configs/ppo/cartpole.yaml"],
  "system": { "python_version": "3.11.7", "platform": "Windows-10-10.0.22631-SP0" },
  "versions": { "torch": "2.10.0", "gymnasium": "1.2.3", "numpy": "2.4.3" },
  "git": { "commit": "7dedac1...", "is_dirty": false },
  "benchmark": {},
  "latest_checkpoint_path": "F:/.../checkpoints/step_640.pt",
  "latest_metrics": { "eval_return_mean": 123.4 }
}
```

## `config.yaml` (serialized TrainConfig)

Despite the `.yaml` filename, this file is currently stored as JSON for
consistent machine parsing during tests and tooling.

It contains the serialized `TrainConfig` payload (paths normalized to strings,
tags normalized to lists).

If you are consuming it programmatically, read it with `json.loads(...)`.

## Checkpoints

Under `checkpoints/` the runtime writes:

- `step_<n>.pt` — periodic checkpoints.
- `best.pt` — alias to the best checkpoint when `benchmark.best_metric` is
  enabled (copied from the best step checkpoint).

## TensorBoard

Metrics are written under `tensorboard/` as event files and can be inspected
with:

```bash
tensorboard --logdir <run_dir>/tensorboard
```

## Study artifacts (`axiomrl tune`)

Each tuning run creates one study directory under the configured study
`output_dir`:

```text
<study_root>/<study_name>/
  study.json
  trials.jsonl
  best_trial.json
  best_config.yaml
  trials/
    <standard training run dirs...>
```

Files:

- `study.json` — summary metadata, status counts, best-trial pointers, and the
  serialized study config snapshot.
- `trials.jsonl` — one record per trial with `trial_index`, `status`, `params`,
  objective value, run path, checkpoint path, timestamps, and error text.
- `best_trial.json` — the winning trial record.
- `best_config.yaml` — the resolved `TrainConfig` payload for the winning trial.
- `trials/` — ordinary train run directories, each with their own
  `metadata.json`, `config.yaml`, checkpoints, and TensorBoard logs.

`axiomrl tune --resume-study <study_dir>` reloads `study.json` plus the
existing `trials.jsonl` history, skips already-recorded trial indices, and
appends only the missing trials before rewriting the study summary and best
artifacts.

`axiomrl tune-report --study-dir <study_dir>` reads the same artifact set
without resuming trials and can emit:

- `text` — summary lines plus per-trial blocks
- `json` — the full `study.json` payload plus a `trials` array loaded from
  `trials.jsonl`
- `csv` — one row per trial with study metadata, flattened `param_*` columns,
  and a `params_json` fallback column

When filters are active, report payloads also carry:

- `selected_trial_count` — number of visible trials after filtering / truncation
- `report_filters` — the active `status`, `param_filters`, `sort_by`,
  `descending`, `top_k`, optional `objective_at_least` /
  `objective_at_most`, optional `duration_at_least` / `duration_at_most`,
  optional `frontier_only`, and optional `error` / `error_contains` /
  `error_type` selection settings
- `selected_status_counts` — status totals for the visible trial slice
- `selected_objective_summary` — min / max / mean / median over completed
  visible trials plus completed / failed counts
- `selected_duration_summary` — visible-trial timing rollup with timed /
  untimed counts plus min / max / mean / median duration in seconds
- `selected_incumbent_trace` — a visible-slice best-so-far timeline ordered by
  `trial_index`, recording when each visible trial updated or inherited the
  current incumbent objective
- `selected_incumbent_update_summary` — a visible-slice rollup of incumbent
  update count, first/latest update location, latest incumbent value, mean/max
  improvement over the previous incumbent, and mean/max visible trial spacing
  between updates
- `selected_incumbent_staleness_summary` — a visible-slice rollup of the
  latest incumbent age and maximum incumbent age over both visible trial count
  and wall-clock seconds
- `selected_objective_duration_frontier` — the visible completed timed trials
  that are nondominated on objective value and wall-clock duration, ordered by
  duration and objective strength for quick Pareto-frontier inspection
- `selected_error_summaries` — failed visible trials grouped by exact error
  text, with per-error counts, visible/failed shares, and trial indices
- `selected_error_type_summaries` — failed visible trials grouped by derived
  exception type, with per-type counts, visible/failed shares, trial indices,
  and the distinct full error texts seen for that type
- `selected_parameter_summaries` — per-parameter unique completed/failed values,
  selected best value, numeric min / max / mean when applicable, plus
  `observed_unique_count`; when the serialized study search space is discrete,
  summaries also expose `search_space_kind`, `candidate_count`, and
  `coverage_ratio`
- `selected_parameter_incumbent_summaries` — per-parameter rollups over
  visible incumbent updates, including which values contributed, which value
  contributed most, and which value most recently updated the incumbent
- `selected_parameter_incumbent_leaderboard` — an ordered list form of the
  parameter incumbent summaries, sorted so the parameters contributing the most
  visible incumbent updates appear first
- `selected_parameter_effect_leaderboard` — an ordered list ranking parameters
  by the spread in visible bucket-level best/mean objective values, including
  the strongest and weakest observed values for each parameter under the
  current visible slice
- `selected_parameter_value_summaries` — per-parameter buckets keyed by each
  observed value, including trial counts, completed/failed counts, and
  completion/failure rates, leaderboard-style ranks by best/mean objective,
  best/mean/median objective values for each visible bucket, plus
  `incumbent_updates`, `latest_incumbent_trial_index`, `timed_trials`,
  `untimed_trials`, `min_duration_seconds`,
  `max_duration_seconds`, `mean_duration_seconds`, and
  `median_duration_seconds` for timing-aware bucket comparisons
- `focused_parameter_name` / `focused_parameter_value_summary` — present when
  `--focus-param` is used; surfaces one parameter's bucket leaderboard as a
  dedicated ordered block. Text reports render this as a dedicated section and
  CSV rows also expose flattened `focused_parameter_*` columns for the matching
  bucket attached to each visible trial row. `--focus-sort-by` accepts
  `best-objective-value`, `mean-objective-value`, `completion-rate`,
  `incumbent-updates`, `mean-duration-seconds`, or `value`. Focused entries
  inherit the same bucket-level duration and incumbent-update fields so the
  focused text / CSV views can expose them.
  `report_filters.focus_sort_by` records the ordering criterion when
  `--focus-sort-by` is used and
  `report_filters.focus_top_k` records focused bucket truncation when
  `--focus-top-k` is used
- `selected_best_trial_index` / `selected_best_objective_value` — the best
  completed trial inside the visible slice after filtering and truncation
- `selected_best_objective_delta` — per-trial gap from the visible best
  objective, normalized so the visible best trial is always `0.0`
- `duration_seconds` — per-trial wall-clock duration derived from
  `started_at` / `ended_at` when both timestamps are present and ordered
- `selected_incumbent_trial_index` — the incumbent trial index visible at that
  trial's position in the visible slice timeline
- `selected_incumbent_objective_value` — the incumbent objective value visible
  at that trial's position in the visible slice timeline
- `selected_is_incumbent_update` — per-trial boolean flag indicating whether
  that visible trial improved the current incumbent
- `selected_incumbent_update_improvement` — per-trial incumbent improvement
  magnitude versus the previous visible incumbent, normalized positive for both
  maximize and minimize objectives; `null` for non-updates and the first visible
  incumbent update
- `selected_incumbent_trials_since_previous_update` — per-trial count of
  visible trials elapsed since the previous incumbent update; `null` when no
  previous visible incumbent exists or the trial did not update the incumbent
- `selected_incumbent_age_trials` — per-trial age of the currently visible
  incumbent measured in visible trials since that incumbent was last updated;
  `null` when no visible incumbent exists yet
- `selected_incumbent_age_seconds` — per-trial age of the currently visible
  incumbent measured from incumbent `ended_at` to the current trial `ended_at`;
  `null` when either timestamp is unavailable or no visible incumbent exists
- `is_objective_duration_frontier` — per-trial boolean flag indicating whether
  that visible trial belongs to `selected_objective_duration_frontier`
- `search_efficiency_summary` — a higher-level rollup with selected-trial count,
  failure rate, best-vs-mean / median deltas, highest / lowest coverage
  parameter entries, plus visible-slice convergence speed fields including
  `selected_trials_until_best`, `selected_trial_share_until_best`,
  `completed_trials_until_best`, `completed_trial_share_until_best`, and
  `time_to_best_seconds`. Text and CSV outputs also flatten these convergence
  metrics into `search_efficiency_*` fields for easier spreadsheet / grep usage
- `config_export_summary` — present when `--export-configs-dir` is used and
  reports the export directory, manifest path, exported config count, and
  skipped trial indices

`axiomrl tune-report --export-configs-dir <dir>` also writes:

- `<dir>/rank-<rank>_trial-<index>.yaml` — one ranked YAML config per exported
  visible completed trial
- `<dir>/manifest.json` — machine-readable metadata linking ranks, trial
  indices, source run dirs, and exported config paths
