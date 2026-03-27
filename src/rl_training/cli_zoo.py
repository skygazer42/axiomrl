from __future__ import annotations

import argparse


def build_zoo_forward_argv(args: argparse.Namespace, *, format_override: str | None = None) -> list[str]:
    manifest = str(getattr(args, "manifest", "zoo/atari/benchmark.yaml"))
    runs_dir = str(getattr(args, "runs_dir", "runs"))
    report_output = str(getattr(args, "report_output", "text"))
    format_value = format_override or str(getattr(args, "format", "table"))

    zoo_argv = [
        "--manifest",
        manifest,
        "--format",
        format_value,
        "--runs-dir",
        runs_dir,
        "--report-output",
        report_output,
    ]

    output = getattr(args, "output", None)
    if output is not None:
        zoo_argv.extend(["--output", str(output)])

    algo = getattr(args, "algo", None)
    if algo is not None:
        zoo_argv.extend(["--algo", str(algo)])

    env_id = getattr(args, "env_id", None)
    if env_id is not None:
        zoo_argv.extend(["--env-id", str(env_id)])

    group_by = getattr(args, "group_by", None)
    if group_by is not None:
        zoo_argv.extend(["--group-by", str(group_by)])

    min_seeds = getattr(args, "min_seeds", None)
    if min_seeds is not None:
        zoo_argv.extend(["--min-seeds", str(min_seeds)])

    top_k = getattr(args, "top_k", None)
    if top_k is not None:
        zoo_argv.extend(["--top-k", str(top_k)])

    baseline_preset = getattr(args, "baseline_preset", None)
    if baseline_preset is not None:
        zoo_argv.extend(["--baseline-preset", str(baseline_preset)])

    leaderboard_metric = getattr(args, "leaderboard_metric", None)
    if leaderboard_metric is not None:
        zoo_argv.extend(["--leaderboard-metric", str(leaderboard_metric)])

    compare_to = getattr(args, "compare_to", None)
    if compare_to is not None:
        zoo_argv.extend(["--compare-to", str(compare_to)])

    score_view = getattr(args, "score_view", None)
    if score_view is not None:
        zoo_argv.extend(["--score-view", str(score_view)])

    sort_by = getattr(args, "sort_by", None)
    if sort_by is not None:
        zoo_argv.extend(["--sort-by", str(sort_by)])

    if getattr(args, "descending", False):
        zoo_argv.append("--descending")

    if getattr(args, "fail_on_manifest_drift", False):
        zoo_argv.append("--fail-on-manifest-drift")

    fail_on_manifest_drift_severity = getattr(args, "fail_on_manifest_drift_severity", None)
    if fail_on_manifest_drift_severity is not None:
        zoo_argv.extend(["--fail-on-manifest-drift-severity", str(fail_on_manifest_drift_severity)])

    fail_on_manifest_drift_type = getattr(args, "fail_on_manifest_drift_type", None)
    if fail_on_manifest_drift_type is not None:
        for drift_type in fail_on_manifest_drift_type:
            zoo_argv.extend(["--fail-on-manifest-drift-type", str(drift_type)])

    return zoo_argv
