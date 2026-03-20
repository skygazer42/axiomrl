# Run Artifacts & Metadata

This document describes what AxiomRL writes to disk during training, how to
interpret those artifacts, and which other tools consume them.

## Run directory layout

Each training run creates one run directory under `TrainConfig.output_dir` with
an id like:

```
<algo>__<env_id>__seed<seed>__<timestamp>
```

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

