from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rl_training.zoo.manifests import (
    _attach_manifest_metadata,
    _build_manifest_alignment_summary,
    _build_manifest_metadata,
    _build_manifest_preset_lookup,
    _copy_metadata_value,
    _resolve_manifest_protocol_metadata,
    _resolve_manifest_protocol_name,
    _resolve_manifest_score_normalization_metadata,
)
from rl_training.zoo.reporting_render import (
    emit_output,
    render_commands_output,
    render_csv_report,
    render_json_report,
    render_table_output,
    render_text_report,
)
from rl_training.zoo.reporting_runs import apply_top_k, filter_run_reports, iter_run_reports, sort_records
from rl_training.zoo.reporting_stats import (
    aggregate_run_reports,
    attach_baseline_comparison_fields,
    build_baseline_summary,
)


def build_report_payload(
    manifest: dict[str, Any],
    *,
    runs_dir: Path,
    manifest_source: Mapping[str, object] | None = None,
    algo: str | None = None,
    env_id: str | None = None,
    group_by: str = "algo-env",
    min_seeds: int | None = None,
    top_k: int | None = None,
    baseline_preset: str | None = None,
    sort_by: str | None = None,
    descending: bool = False,
) -> dict[str, Any]:
    suite = manifest.get("suite", "unknown")
    protocol = manifest.get("protocol", {})
    score_normalization = manifest.get("score_normalization", {})
    manifest_metadata = _build_manifest_metadata(manifest)
    manifest_protocol_name = _resolve_manifest_protocol_name(manifest)
    protocol_metadata = _resolve_manifest_protocol_metadata(manifest)
    score_normalization_metadata = _resolve_manifest_score_normalization_metadata(manifest)
    preset_lookup = _build_manifest_preset_lookup(manifest)
    sorted_reports = sort_records(
        filter_run_reports(iter_run_reports(runs_dir), algo=algo, env_id=env_id),
        sort_by=sort_by,
        descending=descending,
    )
    enriched_sorted_reports = [
        _attach_manifest_metadata(
            report,
            protocol_metadata=protocol_metadata,
            manifest_protocol_name=manifest_protocol_name,
            score_normalization_metadata=score_normalization_metadata,
            preset_lookup=preset_lookup,
        )
        for report in sorted_reports
    ]
    reports = apply_top_k(enriched_sorted_reports, top_k=top_k)
    aggregate_records = [
        _attach_manifest_metadata(
            aggregate,
            protocol_metadata=protocol_metadata,
            manifest_protocol_name=manifest_protocol_name,
            score_normalization_metadata=score_normalization_metadata,
            preset_lookup=preset_lookup,
        )
        for aggregate in aggregate_run_reports(enriched_sorted_reports, group_by=group_by, min_seeds=min_seeds)
    ]
    baseline_summary: dict[str, Any] | None = None
    manifest_alignment_summary = _build_manifest_alignment_summary(enriched_sorted_reports)
    if baseline_preset is not None:
        if group_by != "preset":
            raise ValueError("--baseline-preset requires --group-by preset")
        aggregate_records = attach_baseline_comparison_fields(
            aggregate_records,
            baseline_preset=baseline_preset,
        )
        baseline_summary = build_baseline_summary(
            aggregate_records,
            baseline_preset=baseline_preset,
        )
    aggregates = apply_top_k(
        sort_records(
            aggregate_records,
            sort_by=sort_by,
            descending=descending,
        ),
        top_k=top_k,
    )

    payload: dict[str, Any] = {
        "suite": suite,
        "protocol": protocol.get("name", "unknown") if isinstance(protocol, dict) else "unknown",
        "score_normalization": score_normalization.get("type", "none")
        if isinstance(score_normalization, dict)
        else "none",
        "runs_dir": str(runs_dir),
        "group_by": group_by,
        "manifest_metadata": manifest_metadata,
        "manifest_alignment_summary": manifest_alignment_summary,
        "runs": reports,
        "aggregates": aggregates,
    }
    if manifest_source is not None:
        payload["manifest_source"] = _copy_metadata_value(manifest_source)
    if protocol_metadata is not None:
        payload["protocol_metadata"] = _copy_metadata_value(protocol_metadata)
    if score_normalization_metadata is not None:
        payload["score_normalization_metadata"] = _copy_metadata_value(score_normalization_metadata)
    if baseline_preset is not None:
        payload["baseline_preset"] = baseline_preset
    if baseline_summary is not None:
        payload["baseline_summary"] = baseline_summary
    if algo is not None or env_id is not None:
        payload["filters"] = {"algo": algo, "env_id": env_id}
    if min_seeds is not None:
        payload["min_seeds"] = min_seeds
    if sort_by is not None:
        payload["sort_by"] = sort_by
        payload["descending"] = descending
    if top_k is not None:
        payload["top_k"] = top_k
    return payload


_emit_output = emit_output
_render_commands_output = render_commands_output
_render_csv_report = render_csv_report
_render_json_report = render_json_report
_render_table_output = render_table_output
_render_text_report = render_text_report


__all__ = ["build_report_payload"]
