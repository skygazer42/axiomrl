# AxiomRL Zoo

The zoo layer collects benchmark-oriented presets and launch manifests without
introducing a second training runtime.

Current focus:

- Atari DQN presets
- Atari representation-learning presets (SPR-lite / JOWA-lite)
- Atari planning/world-model presets (DreamerV3-lite / DIAMOND-lite / Horizon Imagination-lite / PO-Dreamer-lite / TWISTED-lite / EADream-lite / MoW-lite / MuZero / Gumbel MuZero-lite / EfficientZero-lite / ScaleZero-lite)
- Atari recurrent exploration presets (R2D2 / Agent57-lite)
- Atari on-policy presets (A2C / IMPALA / PPG / PPO)
- Atari recurrent PPO presets (partial observability)

Use the CLI directly with the config files, or enumerate them with
`axiomrl zoo` or `axiomrl-zoo`.

Each zoo preset points at a full training config, and the CLI can resolve that
link directly. When a sibling `benchmark.yaml` manifest exists, `axiomrl train --config zoo/...`
also inherits suite benchmark defaults and protocol-specific train/eval env overrides.

Examples:

```bash
axiomrl train --config zoo/atari/dqn_breakout.yaml
axiomrl train --config zoo/atari/apex_dqn_breakout.yaml
axiomrl train --config zoo/atari/spr_breakout.yaml
axiomrl train --config zoo/atari/jowa_breakout.yaml
axiomrl train --config zoo/atari/dreamerv3_breakout.yaml
axiomrl train --config zoo/atari/diamond_breakout.yaml
axiomrl train --config zoo/atari/horizon_imagination_breakout.yaml
axiomrl train --config zoo/atari/po_dreamer_breakout.yaml
axiomrl train --config zoo/atari/twisted_breakout.yaml
axiomrl train --config zoo/atari/eadream_breakout.yaml
axiomrl train --config zoo/atari/mow_breakout.yaml
axiomrl train --config zoo/atari/muzero_breakout.yaml
axiomrl train --config zoo/atari/gumbel_muzero_breakout.yaml
axiomrl train --config zoo/atari/efficientzero_breakout.yaml
axiomrl train --config zoo/atari/scalezero_breakout.yaml
axiomrl train --config zoo/atari/a2c_breakout.yaml
axiomrl train --config zoo/atari/impala_breakout.yaml
axiomrl train --config zoo/atari/ppg_breakout.yaml
axiomrl train --config zoo/atari/r2d2_breakout.yaml
axiomrl train --config zoo/atari/agent57_breakout.yaml
axiomrl train --config zoo/atari/ppo_breakout.yaml
axiomrl train --config zoo/atari/recurrent_ppo_breakout.yaml
axiomrl zoo --format commands
axiomrl zoo --format report --runs-dir runs
axiomrl zoo --format report --runs-dir runs --report-output json --algo dqn
axiomrl zoo --format report --runs-dir runs --report-output csv --env-id ALE/Breakout-v5 --sort-by best_eval_return_mean --descending
axiomrl zoo --format report --runs-dir runs --report-output json --output reports/benchmark_report.json
axiomrl zoo --format report --runs-dir runs --report-output csv --output reports/benchmark_report.csv
axiomrl zoo --format report --runs-dir runs --group-by preset --sort-by best_eval_return_mean --descending --top-k 5
axiomrl zoo --format leaderboard --runs-dir runs --group-by preset --top-k 10
axiomrl zoo --format leaderboard --runs-dir runs --group-by preset --min-seeds 3
axiomrl zoo --format leaderboard --runs-dir runs --group-by preset --leaderboard-metric latest-normalized
axiomrl zoo --format leaderboard --runs-dir runs --group-by preset --leaderboard-metric gap-return
axiomrl zoo --format leaderboard --runs-dir runs --group-by preset --compare-to latest
axiomrl zoo --format leaderboard --runs-dir runs --group-by preset --compare-to latest --score-view return
axiomrl zoo --format leaderboard --runs-dir runs --group-by preset --leaderboard-metric stability-normalized
axiomrl zoo --format leaderboard --runs-dir runs --group-by preset --leaderboard-metric confidence-normalized
axiomrl zoo --format leaderboard --runs-dir runs --group-by preset --leaderboard-metric median-normalized
axiomrl zoo --format leaderboard --runs-dir runs --group-by preset --leaderboard-metric iqr-normalized
axiomrl zoo --format leaderboard --runs-dir runs --group-by preset --baseline-preset dqn_breakout --leaderboard-metric delta-vs-baseline-normalized
axiomrl zoo --format leaderboard --runs-dir runs --group-by preset --report-output json --fail-on-manifest-drift
axiomrl zoo --format leaderboard --runs-dir runs --group-by preset --report-output json --fail-on-manifest-drift-severity error
axiomrl zoo --format leaderboard --runs-dir runs --group-by preset --report-output json --fail-on-manifest-drift-type unknown-preset
axiomrl-zoo --format commands
```

`axiomrl zoo --format report --runs-dir runs` prints:

- one line per run from `runs/*/metadata.json`
- aggregate summaries grouped by `(algo, env_id)`
- mean latest return / normalized score across seeds
- max best-checkpoint return across grouped runs
- explicit `seed_count`, `best_over_latest_*` ratios, `rank_*` columns, latest `min/max/std` stability fields, latest `stderr/ci95` confidence fields, latest `median/iqr` robustness fields, and optional baseline delta/ratio fields on aggregate rows

Use `--report-output json` or `--report-output csv` for machine-readable exports, add `--output <path>` when you want a saved artifact, use `--group-by preset` when ranking named benchmark presets, add `--min-seeds <n>` when you want to exclude under-seeded aggregate groups, use `--compare-to latest|best` for the higher-level final-vs-peak switch, use `--score-view return|normalized` for the return-vs-normalized axis switch, use `--baseline-preset <preset>` when you want delta/ratio comparisons against a named preset while using `--group-by preset`, use `--leaderboard-metric` when you want latest/best/gap/stability/baseline ranking aliases without memorizing raw field names, and combine `--algo`, `--env-id`, `--sort-by`, `--descending`, and `--top-k` when inspecting a narrower benchmark slice. JSON exports now include top-level `manifest_source`, top-level `manifest_metadata`, top-level `manifest_alignment_summary`, top-level `manifest_alignment_fail_reasons`, top-level `protocol_metadata`, resolved `score_normalization_metadata`, and per-row `preset_metadata` so benchmark protocol context stays attached to each run or preset aggregate.

Report rows also expose `best_minus_latest_*` delta fields, while `axiomrl zoo --format leaderboard` renders only ranked aggregate entries with `seed_count` and per-metric rank columns for quick benchmark inspection. When `--baseline-preset` is active, JSON and CSV exports also include a `baseline_summary` / summary-row view of top movers and regressions by return and normalized delta across the full filtered preset set. Machine-readable outputs also include manifest drift markers that show whether each run still maps to a known preset and whether its `protocol_name` matches the current manifest, plus per-row / per-aggregate `manifest_alignment_severity` and top-level summary counts, severity, and named drifted presets computed from the full filtered run set even when `--top-k` truncates visible entries. When the benchmark manifest enables score normalization, leaderboard default ordering prefers `best_eval_human_normalized_score`; otherwise it falls back to `best_eval_return_mean`. `--compare-to latest|best` selects the latest or best metric family for that default view, `--score-view return|normalized` selects the score axis for that view, and `--leaderboard-metric` supports `best-return`, `latest-return`, `gap-return`, `stability-return`, `confidence-return`, `median-return`, `iqr-return`, `delta-vs-baseline-return`, `ratio-vs-baseline-return`, `best-normalized`, `latest-normalized`, `gap-normalized`, `stability-normalized`, `confidence-normalized`, `median-normalized`, `iqr-normalized`, `delta-vs-baseline-normalized`, and `ratio-vs-baseline-normalized`. Stability modes rank lower cross-seed standard deviation higher, confidence modes rank lower 95% CI half-width higher, median modes rank higher robust central tendency higher, IQR modes rank lower cross-seed spread higher, and baseline modes rank larger uplift over the named baseline higher. `--baseline-preset` requires `--group-by preset`. `--score-view normalized` requires score normalization in the manifest. CSV exports flatten the same manifest context into `manifest_requested_path`, `manifest_resolved_path`, `manifest_source_kind`, `manifest_fingerprint`, `manifest_preset_count`, `manifest_preset_names`, `manifest_alignment_*`, `protocol_description`, `protocol_training`, `protocol_evaluation`, `score_normalization_*`, `preset_config`, and `preset_description` columns. Add `--fail-on-manifest-drift` when you want the command to keep emitting report or leaderboard output but return exit code `1` for any `warning` or `error` drift, `--fail-on-manifest-drift-severity error` when you only want CI failure for `error` drift, repeat `--fail-on-manifest-drift-type unknown-preset|protocol-mismatch` to gate specific drift categories, and read `manifest_alignment_fail_reasons` when you want the current fail gate summarized as an explicit machine-readable reason list.
