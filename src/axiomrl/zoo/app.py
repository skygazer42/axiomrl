from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from axiomrl.zoo.leaderboard import (
    COMPARE_TO_CHOICES,
    LEADERBOARD_METRIC_CHOICES,
    SCORE_VIEW_CHOICES,
    _render_csv_leaderboard,
    _render_json_leaderboard,
    _render_text_leaderboard,
    build_leaderboard_payload,
)
from axiomrl.zoo.manifests import (
    MANIFEST_DRIFT_TYPE_CHOICES,
    MANIFEST_DRIFT_TYPE_TO_SUMMARY_FIELD,
    _default_manifest_path,
    load_manifest_with_source,
)
from axiomrl.zoo.reporting import (
    _emit_output,
    _render_commands_output,
    _render_csv_report,
    _render_json_report,
    _render_table_output,
    _render_text_report,
    build_report_payload,
)


def _resolve_manifest_drift_exit_code(
    payload: Mapping[str, Any],
    *,
    fail_on_manifest_drift: bool,
    fail_on_manifest_drift_severity: str | None = None,
    fail_on_manifest_drift_types: list[str] | None = None,
) -> int:
    fail_reasons = _build_manifest_alignment_fail_reasons(
        payload,
        fail_on_manifest_drift=fail_on_manifest_drift,
        fail_on_manifest_drift_severity=fail_on_manifest_drift_severity,
        fail_on_manifest_drift_types=fail_on_manifest_drift_types,
    )
    if fail_reasons:
        return 1
    return 0


def _build_manifest_alignment_fail_reasons(
    payload: Mapping[str, Any],
    *,
    fail_on_manifest_drift: bool,
    fail_on_manifest_drift_severity: str | None = None,
    fail_on_manifest_drift_types: list[str] | None = None,
) -> list[str] | None:
    threshold = fail_on_manifest_drift_severity
    if threshold is None and fail_on_manifest_drift:
        threshold = "warning"
    selected_types = list(dict.fromkeys(fail_on_manifest_drift_types or []))
    if threshold is None and not selected_types:
        return None

    manifest_alignment_summary = payload.get("manifest_alignment_summary")
    if not isinstance(manifest_alignment_summary, Mapping):
        return []

    present_drift_types = [
        drift_type
        for drift_type, summary_field in MANIFEST_DRIFT_TYPE_TO_SUMMARY_FIELD.items()
        if isinstance(manifest_alignment_summary.get(summary_field), int)
        and manifest_alignment_summary.get(summary_field, 0) > 0
    ]
    fail_reasons: list[str] = []

    for drift_type in selected_types:
        if drift_type in present_drift_types:
            fail_reasons.append(drift_type)
    if threshold is not None:
        if threshold == "warning":
            fail_reasons.extend(present_drift_types)
        elif threshold == "error" and "unknown-preset" in present_drift_types:
            fail_reasons.append("unknown-preset")
    return list(dict.fromkeys(fail_reasons))


def _print_report(
    manifest: dict[str, Any],
    *,
    runs_dir: Path,
    manifest_source: Mapping[str, object] | None = None,
    report_output: str = "text",
    output_path: str | Path | None = None,
    algo: str | None = None,
    env_id: str | None = None,
    group_by: str = "algo-env",
    min_seeds: int | None = None,
    top_k: int | None = None,
    baseline_preset: str | None = None,
    sort_by: str | None = None,
    descending: bool = False,
    fail_on_manifest_drift: bool = False,
    fail_on_manifest_drift_severity: str | None = None,
    fail_on_manifest_drift_types: list[str] | None = None,
) -> int:
    payload = build_report_payload(
        manifest,
        runs_dir=runs_dir,
        manifest_source=manifest_source,
        algo=algo,
        env_id=env_id,
        group_by=group_by,
        min_seeds=min_seeds,
        top_k=top_k,
        baseline_preset=baseline_preset,
        sort_by=sort_by,
        descending=descending,
    )
    payload = dict(payload)
    manifest_alignment_fail_reasons = _build_manifest_alignment_fail_reasons(
        payload,
        fail_on_manifest_drift=fail_on_manifest_drift,
        fail_on_manifest_drift_severity=fail_on_manifest_drift_severity,
        fail_on_manifest_drift_types=fail_on_manifest_drift_types,
    )
    if manifest_alignment_fail_reasons is not None:
        payload["manifest_alignment_fail_reasons"] = manifest_alignment_fail_reasons
    if report_output == "json":
        _emit_output(_render_json_report(payload), output_path=output_path)
    elif report_output == "csv":
        _emit_output(_render_csv_report(payload), output_path=output_path)
    else:
        _emit_output(_render_text_report(payload), output_path=output_path)
    return _resolve_manifest_drift_exit_code(
        payload,
        fail_on_manifest_drift=fail_on_manifest_drift,
        fail_on_manifest_drift_severity=fail_on_manifest_drift_severity,
        fail_on_manifest_drift_types=fail_on_manifest_drift_types,
    )


def _print_leaderboard(
    manifest: dict[str, Any],
    *,
    runs_dir: Path,
    manifest_source: Mapping[str, object] | None = None,
    report_output: str = "text",
    output_path: str | Path | None = None,
    algo: str | None = None,
    env_id: str | None = None,
    group_by: str = "algo-env",
    min_seeds: int | None = None,
    top_k: int | None = None,
    leaderboard_metric: str | None = None,
    baseline_preset: str | None = None,
    compare_to: str | None = None,
    score_view: str | None = None,
    sort_by: str | None = None,
    descending: bool = False,
    fail_on_manifest_drift: bool = False,
    fail_on_manifest_drift_severity: str | None = None,
    fail_on_manifest_drift_types: list[str] | None = None,
) -> int:
    payload = build_leaderboard_payload(
        manifest,
        runs_dir=runs_dir,
        manifest_source=manifest_source,
        algo=algo,
        env_id=env_id,
        group_by=group_by,
        min_seeds=min_seeds,
        top_k=top_k,
        leaderboard_metric=leaderboard_metric,
        baseline_preset=baseline_preset,
        compare_to=compare_to,
        score_view=score_view,
        sort_by=sort_by,
        descending=descending,
    )
    payload = dict(payload)
    manifest_alignment_fail_reasons = _build_manifest_alignment_fail_reasons(
        payload,
        fail_on_manifest_drift=fail_on_manifest_drift,
        fail_on_manifest_drift_severity=fail_on_manifest_drift_severity,
        fail_on_manifest_drift_types=fail_on_manifest_drift_types,
    )
    if manifest_alignment_fail_reasons is not None:
        payload["manifest_alignment_fail_reasons"] = manifest_alignment_fail_reasons
    if report_output == "json":
        _emit_output(_render_json_leaderboard(payload), output_path=output_path)
    elif report_output == "csv":
        _emit_output(_render_csv_leaderboard(payload), output_path=output_path)
    else:
        _emit_output(_render_text_leaderboard(payload), output_path=output_path)
    return _resolve_manifest_drift_exit_code(
        payload,
        fail_on_manifest_drift=fail_on_manifest_drift,
        fail_on_manifest_drift_severity=fail_on_manifest_drift_severity,
        fail_on_manifest_drift_types=fail_on_manifest_drift_types,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List or print benchmark commands from the AxiomRL zoo.")
    parser.add_argument(
        "--manifest",
        default=str(_default_manifest_path()),
        help="Path to a zoo benchmark manifest.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "commands", "report", "leaderboard"),
        default="table",
        help="Choose plain table output, shell commands, or run report output.",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory containing benchmark run folders with metadata.json files.",
    )
    parser.add_argument(
        "--report-output",
        choices=("text", "json", "csv"),
        default="text",
        help="Choose text, JSON, or CSV rendering for report mode.",
    )
    parser.add_argument("--algo", help="Filter report mode to a single algorithm.")
    parser.add_argument("--env-id", help="Filter report mode to a single environment id.")
    parser.add_argument(
        "--group-by",
        choices=("algo-env", "preset"),
        default="algo-env",
        help="Choose aggregate grouping for report mode.",
    )
    parser.add_argument("--min-seeds", type=int, help="Keep only aggregate groups with at least this many seeds.")
    parser.add_argument("--top-k", type=int, help="Keep only the top-k sorted run and aggregate rows.")
    parser.add_argument(
        "--baseline-preset",
        help="Attach baseline delta and ratio fields relative to a preset when --group-by preset is active.",
    )
    parser.add_argument(
        "--leaderboard-metric",
        choices=LEADERBOARD_METRIC_CHOICES,
        help="Leaderboard-only metric alias for latest/best/gap return or normalized-score ranking.",
    )
    parser.add_argument(
        "--compare-to",
        choices=COMPARE_TO_CHOICES,
        help="Leaderboard-only default selector between best and latest metric families.",
    )
    parser.add_argument(
        "--score-view",
        choices=SCORE_VIEW_CHOICES,
        help="Leaderboard-only selector between return and normalized-score ranking axes.",
    )
    parser.add_argument("--sort-by", help="Sort report records by a metric or field.")
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort report records in descending order when --sort-by is provided.",
    )
    parser.add_argument(
        "--fail-on-manifest-drift",
        action="store_true",
        help="Return exit code 1 for report or leaderboard output when the filtered slice contains manifest drift.",
    )
    parser.add_argument(
        "--fail-on-manifest-drift-severity",
        choices=("warning", "error"),
        help="Return a non-zero exit only when manifest drift reaches at least this severity.",
    )
    parser.add_argument(
        "--fail-on-manifest-drift-type",
        action="append",
        choices=MANIFEST_DRIFT_TYPE_CHOICES,
        help="Return a non-zero exit only when the selected manifest drift type occurs; may be repeated.",
    )
    parser.add_argument("--output", help="Write rendered output to a file path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, manifest_source = load_manifest_with_source(args.manifest)
    presets = manifest.get("presets", [])
    if not isinstance(presets, list):
        raise TypeError("manifest 'presets' must be a list")

    if args.format == "commands":
        _emit_output(_render_commands_output(manifest), output_path=args.output)
    elif args.format == "report":
        return _print_report(
            manifest,
            runs_dir=Path(args.runs_dir),
            manifest_source=manifest_source,
            report_output=args.report_output,
            output_path=args.output,
            algo=args.algo,
            env_id=args.env_id,
            group_by=args.group_by,
            min_seeds=args.min_seeds,
            top_k=args.top_k,
            baseline_preset=args.baseline_preset,
            sort_by=args.sort_by,
            descending=args.descending,
            fail_on_manifest_drift=args.fail_on_manifest_drift,
            fail_on_manifest_drift_severity=args.fail_on_manifest_drift_severity,
            fail_on_manifest_drift_types=args.fail_on_manifest_drift_type,
        )
    elif args.format == "leaderboard":
        return _print_leaderboard(
            manifest,
            runs_dir=Path(args.runs_dir),
            manifest_source=manifest_source,
            report_output=args.report_output,
            output_path=args.output,
            algo=args.algo,
            env_id=args.env_id,
            group_by=args.group_by,
            min_seeds=args.min_seeds,
            top_k=args.top_k,
            leaderboard_metric=args.leaderboard_metric,
            baseline_preset=args.baseline_preset,
            compare_to=args.compare_to,
            score_view=args.score_view,
            sort_by=args.sort_by,
            descending=args.descending,
            fail_on_manifest_drift=args.fail_on_manifest_drift,
            fail_on_manifest_drift_severity=args.fail_on_manifest_drift_severity,
            fail_on_manifest_drift_types=args.fail_on_manifest_drift_type,
        )
    else:
        _emit_output(_render_table_output(manifest), output_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
