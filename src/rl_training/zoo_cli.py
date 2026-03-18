from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import yaml

from rl_training.experiment.benchmarking import resolve_score_normalization_settings
from rl_training.resources import find_packaged_asset

MANIFEST_DRIFT_TYPE_CHOICES = ("unknown-preset", "protocol-mismatch")
MANIFEST_DRIFT_TYPE_TO_SUMMARY_FIELD = {
    "unknown-preset": "unknown_preset_runs",
    "protocol-mismatch": "protocol_mismatch_runs",
}


def _default_manifest_path() -> Path:
    packaged = find_packaged_asset("zoo/atari/benchmark.yaml")
    if packaged is not None:
        return packaged
    return Path("zoo/atari/benchmark.yaml")


def resolve_manifest_source(path: str | Path) -> dict[str, str]:
    manifest_path = Path(path)
    if manifest_path.exists():
        return {
            "requested_path": str(path),
            "resolved_path": str(manifest_path.resolve()),
            "source_kind": "filesystem",
        }

    packaged = find_packaged_asset(manifest_path)
    if packaged is not None:
        return {
            "requested_path": str(path),
            "resolved_path": str(packaged.resolve()),
            "source_kind": "packaged_asset",
        }

    return {
        "requested_path": str(path),
        "resolved_path": str(manifest_path.resolve()),
        "source_kind": "filesystem",
    }


def load_manifest_with_source(path: str | Path) -> tuple[dict[str, Any], dict[str, str]]:
    source = resolve_manifest_source(path)
    payload = yaml.safe_load(Path(source["resolved_path"]).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"expected YAML object in {source['resolved_path']}, got {type(payload)!r}")
    return payload, source


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload, _ = load_manifest_with_source(path)
    return payload


def _merge_mappings(base: Mapping[str, object], override: Mapping[str, object]) -> dict[str, object]:
    merged: dict[str, object] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mappings(existing, value)
        else:
            merged[key] = value
    return merged


def _manifest_matches_preset(
    manifest: Mapping[str, object],
    *,
    preset_path: Path,
    preset_payload: Mapping[str, object],
) -> bool:
    presets = manifest.get("presets", [])
    if not isinstance(presets, list):
        return False

    preset_name = str(preset_payload.get("name", "")).strip()
    for entry in presets:
        if not isinstance(entry, Mapping):
            continue
        if preset_name and str(entry.get("name", "")).strip() == preset_name:
            return True
        config_path = entry.get("config")
        if isinstance(config_path, str) and Path(config_path).name == preset_path.name:
            return True
    return False


def find_manifest_for_preset(
    preset_path: str | Path,
    *,
    preset_payload: Mapping[str, object],
) -> dict[str, Any] | None:
    resolved_preset_path = Path(preset_path).resolve()
    for parent in resolved_preset_path.parents:
        candidate = parent / "benchmark.yaml"
        if not candidate.exists():
            continue
        manifest = load_manifest(candidate)
        if _manifest_matches_preset(manifest, preset_path=resolved_preset_path, preset_payload=preset_payload):
            return manifest
    return None


def _apply_manifest_protocol_defaults(
    env_kwargs: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        return dict(env_kwargs)

    resolved_env_kwargs = dict(env_kwargs)
    for mode in ("training", "evaluation"):
        defaults = protocol.get(mode)
        if defaults is None:
            continue
        if not isinstance(defaults, Mapping):
            raise TypeError(f"expected manifest protocol '{mode}' defaults to be a mapping, got {type(defaults)!r}")
        existing = resolved_env_kwargs.get(mode, {})
        if existing is None:
            existing = {}
        if not isinstance(existing, Mapping):
            raise TypeError(f"expected env_kwargs['{mode}'] to be a mapping, got {type(existing)!r}")
        resolved_env_kwargs[mode] = _merge_mappings(defaults, existing)
    return resolved_env_kwargs


def _build_manifest_benchmark_defaults(
    manifest: Mapping[str, object],
    *,
    preset_payload: Mapping[str, object],
) -> dict[str, object]:
    benchmark: dict[str, object] = {}

    suite = manifest.get("suite")
    if suite is not None:
        benchmark["suite"] = suite

    preset_name = str(preset_payload.get("name", "")).strip()
    if preset_name:
        benchmark["preset_name"] = preset_name

    protocol = manifest.get("protocol")
    if isinstance(protocol, Mapping):
        protocol_name = str(protocol.get("name", "")).strip()
        if protocol_name:
            benchmark["protocol_name"] = protocol_name

    for key in ("best_metric", "best_metric_mode"):
        if key in manifest:
            benchmark[key] = manifest[key]

    score_normalization = manifest.get("score_normalization")
    if isinstance(score_normalization, Mapping):
        benchmark["score_normalization"] = resolve_score_normalization_settings(score_normalization)

    return benchmark


def _copy_metadata_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _copy_metadata_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_metadata_value(item) for item in value]
    return value


def _resolve_manifest_protocol_metadata(manifest: Mapping[str, object]) -> dict[str, object] | None:
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        return None
    copied = _copy_metadata_value(protocol)
    if not isinstance(copied, dict):
        return None
    return copied


def _resolve_manifest_score_normalization_metadata(manifest: Mapping[str, object]) -> dict[str, object] | None:
    score_normalization = manifest.get("score_normalization")
    if not isinstance(score_normalization, Mapping):
        return None
    strategy = str(score_normalization.get("type", "")).strip().lower()
    if strategy in {"", "none"}:
        copied = _copy_metadata_value(score_normalization)
    else:
        copied = _copy_metadata_value(resolve_score_normalization_settings(score_normalization))
    if not isinstance(copied, dict):
        return None
    return copied


def _build_manifest_preset_lookup(manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    presets = manifest.get("presets", [])
    if not isinstance(presets, list):
        return {}

    lookup: dict[str, dict[str, object]] = {}
    for entry in presets:
        if not isinstance(entry, Mapping):
            continue
        copied = _copy_metadata_value(entry)
        if not isinstance(copied, dict):
            continue
        preset_name = str(copied.get("name", "")).strip()
        if preset_name:
            lookup[preset_name] = copied
        config_path = str(copied.get("config", "")).strip()
        if config_path:
            lookup.setdefault(config_path, copied)
            lookup.setdefault(Path(config_path).name, copied)
    return lookup


def _resolve_manifest_preset_metadata(
    preset_lookup: Mapping[str, dict[str, object]],
    *,
    preset_name: object,
) -> dict[str, object] | None:
    preset_key = str(preset_name).strip()
    if not preset_key or preset_key == "multiple":
        return None
    preset_metadata = preset_lookup.get(preset_key)
    if preset_metadata is None:
        return None
    copied = _copy_metadata_value(preset_metadata)
    if not isinstance(copied, dict):
        return None
    return copied


def _attach_manifest_metadata(
    record: Mapping[str, Any],
    *,
    protocol_metadata: Mapping[str, object] | None,
    manifest_protocol_name: str | None,
    score_normalization_metadata: Mapping[str, object] | None,
    preset_lookup: Mapping[str, dict[str, object]],
) -> dict[str, Any]:
    enriched = dict(record)

    resolved_protocol_metadata = _copy_metadata_value(protocol_metadata) if protocol_metadata is not None else None
    enriched["protocol_metadata"] = resolved_protocol_metadata
    if isinstance(resolved_protocol_metadata, Mapping):
        enriched["protocol_description"] = resolved_protocol_metadata.get("description")
        enriched["protocol_training"] = resolved_protocol_metadata.get("training")
        enriched["protocol_evaluation"] = resolved_protocol_metadata.get("evaluation")
    else:
        enriched["protocol_description"] = None
        enriched["protocol_training"] = None
        enriched["protocol_evaluation"] = None

    resolved_score_normalization_metadata = (
        _copy_metadata_value(score_normalization_metadata) if score_normalization_metadata is not None else None
    )
    enriched["score_normalization_metadata"] = resolved_score_normalization_metadata
    if isinstance(resolved_score_normalization_metadata, Mapping):
        enriched["score_normalization_game"] = resolved_score_normalization_metadata.get("game")
        enriched["score_normalization_source"] = resolved_score_normalization_metadata.get("source")
        enriched["score_normalization_random_score"] = resolved_score_normalization_metadata.get("random_score")
        enriched["score_normalization_human_score"] = resolved_score_normalization_metadata.get("human_score")
        enriched["score_normalization_scale"] = resolved_score_normalization_metadata.get("scale")
    else:
        enriched["score_normalization_game"] = None
        enriched["score_normalization_source"] = None
        enriched["score_normalization_random_score"] = None
        enriched["score_normalization_human_score"] = None
        enriched["score_normalization_scale"] = None

    preset_metadata = _resolve_manifest_preset_metadata(preset_lookup, preset_name=record.get("preset_name"))
    enriched["preset_metadata"] = preset_metadata
    enriched["preset_config"] = preset_metadata.get("config") if isinstance(preset_metadata, Mapping) else None
    enriched["preset_description"] = preset_metadata.get("description") if isinstance(preset_metadata, Mapping) else None
    if "manifest_alignment_status" not in enriched:
        alignment_fields = _resolve_manifest_alignment_fields(
            record,
            manifest_protocol_name=manifest_protocol_name,
            preset_lookup=preset_lookup,
        )
        if alignment_fields is not None:
            enriched.update(alignment_fields)
    return enriched


def _serialize_metadata_field(value: object) -> object:
    if isinstance(value, Mapping) or isinstance(value, list):
        return json.dumps(_copy_metadata_value(value), sort_keys=True)
    return value


def _build_payload_metadata_fields(payload: Mapping[str, Any]) -> dict[str, object]:
    fields: dict[str, object] = {
        "manifest_requested_path": None,
        "manifest_resolved_path": None,
        "manifest_source_kind": None,
        "manifest_fingerprint": None,
        "manifest_preset_count": None,
        "manifest_preset_names": None,
        "manifest_alignment_total_runs": None,
        "manifest_alignment_aligned_runs": None,
        "manifest_alignment_drifted_runs": None,
        "manifest_alignment_unknown_preset_runs": None,
        "manifest_alignment_protocol_mismatch_runs": None,
        "manifest_alignment_all_runs": None,
        "manifest_alignment_severity": None,
        "manifest_alignment_drifted_presets": None,
        "manifest_alignment_fail_reasons": None,
        "protocol_description": None,
        "protocol_training": None,
        "protocol_evaluation": None,
        "score_normalization_game": None,
        "score_normalization_source": None,
        "score_normalization_random_score": None,
        "score_normalization_human_score": None,
        "score_normalization_scale": None,
    }

    manifest_source = payload.get("manifest_source")
    if isinstance(manifest_source, Mapping):
        fields["manifest_requested_path"] = manifest_source.get("requested_path")
        fields["manifest_resolved_path"] = manifest_source.get("resolved_path")
        fields["manifest_source_kind"] = manifest_source.get("source_kind")

    manifest_metadata = payload.get("manifest_metadata")
    if isinstance(manifest_metadata, Mapping):
        fields["manifest_fingerprint"] = manifest_metadata.get("fingerprint")
        fields["manifest_preset_count"] = manifest_metadata.get("preset_count")
        fields["manifest_preset_names"] = _serialize_metadata_field(manifest_metadata.get("preset_names"))

    manifest_alignment_summary = payload.get("manifest_alignment_summary")
    if isinstance(manifest_alignment_summary, Mapping):
        fields["manifest_alignment_total_runs"] = manifest_alignment_summary.get("total_runs")
        fields["manifest_alignment_aligned_runs"] = manifest_alignment_summary.get("aligned_runs")
        fields["manifest_alignment_drifted_runs"] = manifest_alignment_summary.get("drifted_runs")
        fields["manifest_alignment_unknown_preset_runs"] = manifest_alignment_summary.get("unknown_preset_runs")
        fields["manifest_alignment_protocol_mismatch_runs"] = manifest_alignment_summary.get("protocol_mismatch_runs")
        fields["manifest_alignment_all_runs"] = manifest_alignment_summary.get("all_runs_aligned")
        fields["manifest_alignment_severity"] = manifest_alignment_summary.get("severity")
        fields["manifest_alignment_drifted_presets"] = _serialize_metadata_field(
            manifest_alignment_summary.get("drifted_presets")
        )
    fields["manifest_alignment_fail_reasons"] = _serialize_metadata_field(
        payload.get("manifest_alignment_fail_reasons")
    )

    protocol_metadata = payload.get("protocol_metadata")
    if isinstance(protocol_metadata, Mapping):
        fields["protocol_description"] = protocol_metadata.get("description")
        fields["protocol_training"] = _serialize_metadata_field(protocol_metadata.get("training"))
        fields["protocol_evaluation"] = _serialize_metadata_field(protocol_metadata.get("evaluation"))

    score_normalization_metadata = payload.get("score_normalization_metadata")
    if isinstance(score_normalization_metadata, Mapping):
        fields["score_normalization_game"] = score_normalization_metadata.get("game")
        fields["score_normalization_source"] = score_normalization_metadata.get("source")
        fields["score_normalization_random_score"] = score_normalization_metadata.get("random_score")
        fields["score_normalization_human_score"] = score_normalization_metadata.get("human_score")
        fields["score_normalization_scale"] = score_normalization_metadata.get("scale")

    return fields


def _build_manifest_metadata(manifest: Mapping[str, object]) -> dict[str, object]:
    copied_manifest = _copy_metadata_value(manifest)
    canonical = json.dumps(copied_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    presets = manifest.get("presets", [])
    preset_names: list[str] = []
    if isinstance(presets, list):
        for preset in presets:
            if not isinstance(preset, Mapping):
                continue
            preset_name = str(preset.get("name", "")).strip()
            if preset_name:
                preset_names.append(preset_name)

    return {
        "suite": manifest.get("suite", "unknown"),
        "preset_count": len(preset_names),
        "preset_names": preset_names,
        "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _resolve_manifest_protocol_name(manifest: Mapping[str, object]) -> str | None:
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        return None
    protocol_name = str(protocol.get("name", "")).strip()
    return protocol_name or None


def _build_manifest_alignment_status(
    *,
    preset_known: bool | None,
    protocol_matches_manifest: bool | None,
) -> str:
    issues: list[str] = []
    if preset_known is False:
        issues.append("preset_unknown")
    if protocol_matches_manifest is False:
        issues.append("protocol_mismatch")
    if not issues:
        return "aligned"
    return "_and_".join(issues)


def _build_manifest_alignment_severity(
    *,
    drifted_runs: int,
    unknown_preset_runs: int,
    protocol_mismatch_runs: int,
) -> str:
    if unknown_preset_runs > 0:
        return "error"
    if protocol_mismatch_runs > 0 or drifted_runs > 0:
        return "warning"
    return "clean"


def _resolve_manifest_alignment_label(record: Mapping[str, Any]) -> str | None:
    preset_name = str(record.get("preset_name", "")).strip()
    if preset_name and preset_name != "multiple":
        return preset_name

    algo = str(record.get("algo", "")).strip()
    env_id = str(record.get("env_id", "")).strip()
    if algo and env_id:
        return f"{algo}::{env_id}"
    if algo:
        return algo
    if env_id:
        return env_id

    run_id = str(record.get("run_id", "")).strip()
    return run_id or None


def _resolve_manifest_alignment_fields(
    record: Mapping[str, Any],
    *,
    manifest_protocol_name: str | None,
    preset_lookup: Mapping[str, dict[str, object]],
) -> dict[str, Any] | None:
    preset_name = str(record.get("preset_name", "")).strip()
    protocol_name = str(record.get("protocol_name", "")).strip()

    if preset_name == "multiple" or protocol_name == "multiple":
        return None

    preset_known = bool(preset_name) and preset_name in preset_lookup
    protocol_matches_manifest: bool | None
    if manifest_protocol_name is None:
        protocol_matches_manifest = None
    else:
        protocol_matches_manifest = protocol_name == manifest_protocol_name

    unknown_preset_runs = int(preset_known is False)
    protocol_mismatch_runs = int(protocol_matches_manifest is False)
    status = _build_manifest_alignment_status(
        preset_known=preset_known,
        protocol_matches_manifest=protocol_matches_manifest,
    )
    return {
        "manifest_preset_known": preset_known,
        "manifest_protocol_matches_manifest": protocol_matches_manifest,
        "manifest_alignment_status": status,
        "manifest_alignment_severity": _build_manifest_alignment_severity(
            drifted_runs=int(status != "aligned"),
            unknown_preset_runs=unknown_preset_runs,
            protocol_mismatch_runs=protocol_mismatch_runs,
        ),
    }


def _build_manifest_alignment_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    total_runs = len(reports)
    aligned_runs = sum(1 for report in reports if report.get("manifest_alignment_status") == "aligned")
    unknown_preset_runs = sum(1 for report in reports if report.get("manifest_preset_known") is False)
    protocol_mismatch_runs = sum(
        1 for report in reports if report.get("manifest_protocol_matches_manifest") is False
    )
    drifted_runs = total_runs - aligned_runs
    drifted_presets = sorted(
        {
            label
            for report in reports
            if report.get("manifest_alignment_status") != "aligned"
            for label in [_resolve_manifest_alignment_label(report)]
            if label is not None
        }
    )
    return {
        "total_runs": total_runs,
        "aligned_runs": aligned_runs,
        "drifted_runs": drifted_runs,
        "unknown_preset_runs": unknown_preset_runs,
        "protocol_mismatch_runs": protocol_mismatch_runs,
        "all_runs_aligned": drifted_runs == 0,
        "severity": _build_manifest_alignment_severity(
            drifted_runs=drifted_runs,
            unknown_preset_runs=unknown_preset_runs,
            protocol_mismatch_runs=protocol_mismatch_runs,
        ),
        "drifted_presets": drifted_presets,
    }


def apply_manifest_defaults_to_config_payload(
    payload: Mapping[str, object],
    *,
    preset_path: str | Path,
    preset_payload: Mapping[str, object],
) -> dict[str, object]:
    manifest = find_manifest_for_preset(preset_path, preset_payload=preset_payload)
    if manifest is None:
        return dict(payload)

    resolved_payload = dict(payload)

    env_kwargs = resolved_payload.get("env_kwargs", {})
    if env_kwargs is None:
        env_kwargs = {}
    if not isinstance(env_kwargs, Mapping):
        raise TypeError(f"expected config env_kwargs to be a mapping, got {type(env_kwargs)!r}")
    resolved_payload["env_kwargs"] = _apply_manifest_protocol_defaults(env_kwargs, manifest=manifest)

    benchmark = resolved_payload.get("benchmark", {})
    if benchmark is None:
        benchmark = {}
    if not isinstance(benchmark, Mapping):
        raise TypeError(f"expected config benchmark to be a mapping, got {type(benchmark)!r}")
    resolved_payload["benchmark"] = _merge_mappings(
        _build_manifest_benchmark_defaults(manifest, preset_payload=preset_payload),
        benchmark,
    )

    return resolved_payload


def _load_run_metadata(metadata_path: Path) -> dict[str, Any]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {metadata_path}, got {type(payload)!r}")
    return payload


def _iter_run_reports(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []

    reports: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        payload = _load_run_metadata(metadata_path)
        latest_metrics = payload.get("latest_metrics", {})
        best_checkpoint = payload.get("best_checkpoint", {})
        benchmark = payload.get("benchmark", {})
        if not isinstance(latest_metrics, dict):
            latest_metrics = {}
        if not isinstance(best_checkpoint, dict):
            best_checkpoint = {}
        if not isinstance(benchmark, dict):
            benchmark = {}
        reports.append(
            {
                "run_id": run_dir.name,
                "algo": payload.get("algo", "unknown"),
                "env_id": payload.get("env_id", "unknown"),
                "seed": payload.get("seed", "unknown"),
                "suite": benchmark.get("suite"),
                "preset_name": benchmark.get("preset_name"),
                "protocol_name": benchmark.get("protocol_name"),
                "latest_eval_return_mean": latest_metrics.get("eval_return_mean"),
                "latest_eval_human_normalized_score": latest_metrics.get("eval_human_normalized_score"),
                "best_eval_return_mean": latest_metrics.get("best_eval_return_mean", best_checkpoint.get("metric_value")),
                "best_eval_human_normalized_score": latest_metrics.get(
                    "best_eval_human_normalized_score",
                    best_checkpoint.get("eval_human_normalized_score"),
                ),
                "best_minus_latest_eval_return_mean": _difference(
                    latest_metrics.get("best_eval_return_mean", best_checkpoint.get("metric_value")),
                    latest_metrics.get("eval_return_mean"),
                ),
                "best_minus_latest_eval_human_normalized_score": _difference(
                    latest_metrics.get(
                        "best_eval_human_normalized_score",
                        best_checkpoint.get("eval_human_normalized_score"),
                    ),
                    latest_metrics.get("eval_human_normalized_score"),
                ),
                "best_checkpoint_path": best_checkpoint.get("path"),
            }
        )
    return reports


def _seed_sort_key(seed: object) -> tuple[int, int | str]:
    if isinstance(seed, int):
        return (0, seed)
    seed_text = str(seed)
    try:
        return (0, int(seed_text))
    except ValueError:
        return (1, seed_text)


def _mean(values: list[object]) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _sorted_numeric_values(values: list[object]) -> list[float]:
    return sorted(float(value) for value in values if value is not None)


def _max_value(values: list[object]) -> float | None:
    numeric_values = _sorted_numeric_values(values)
    if not numeric_values:
        return None
    return max(numeric_values)


def _min_value(values: list[object]) -> float | None:
    numeric_values = _sorted_numeric_values(values)
    if not numeric_values:
        return None
    return min(numeric_values)


def _median(values: list[object]) -> float | None:
    numeric_values = _sorted_numeric_values(values)
    if not numeric_values:
        return None
    midpoint = len(numeric_values) // 2
    if len(numeric_values) % 2 == 1:
        return numeric_values[midpoint]
    return (numeric_values[midpoint - 1] + numeric_values[midpoint]) / 2.0


def _quantile_inclusive(values: list[object], quantile: float) -> float | None:
    numeric_values = _sorted_numeric_values(values)
    if not numeric_values:
        return None
    if len(numeric_values) == 1:
        return numeric_values[0]
    position = (len(numeric_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return numeric_values[lower_index]
    weight = position - lower_index
    lower_value = numeric_values[lower_index]
    upper_value = numeric_values[upper_index]
    return lower_value + (upper_value - lower_value) * weight


def _iqr(values: list[object]) -> float | None:
    numeric_values = _sorted_numeric_values(values)
    if len(numeric_values) < 2:
        return None
    q1 = _quantile_inclusive(numeric_values, 0.25)
    q3 = _quantile_inclusive(numeric_values, 0.75)
    if q1 is None or q3 is None:
        return None
    return q3 - q1


def _stddev(values: list[object]) -> float | None:
    numeric_values = _sorted_numeric_values(values)
    if len(numeric_values) < 2:
        return None
    mean_value = sum(numeric_values) / len(numeric_values)
    variance = sum((value - mean_value) ** 2 for value in numeric_values) / (len(numeric_values) - 1)
    return math.sqrt(variance)


def _stderr(values: list[object]) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if len(numeric_values) < 2:
        return None
    stddev = _stddev(numeric_values)
    if stddev is None:
        return None
    return stddev / math.sqrt(len(numeric_values))


def _ci95(values: list[object]) -> float | None:
    stderr = _stderr(values)
    if stderr is None:
        return None
    return 1.96 * stderr


def _difference(lhs: object, rhs: object) -> float | None:
    if lhs is None or rhs is None:
        return None
    return float(lhs) - float(rhs)


def _ratio(numerator: object, denominator: object) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator_value = float(denominator)
    if denominator_value == 0.0:
        return None
    return float(numerator) / denominator_value


def _single_or_multiple(values: list[object]) -> object | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    first = present[0]
    if all(value == first for value in present[1:]):
        return first
    return "multiple"


def _attach_aggregate_rank_fields(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank_specs = (
        ("best_eval_return_mean_max", "rank_best_eval_return_mean"),
        ("latest_eval_return_mean_mean", "rank_latest_eval_return_mean"),
        ("best_eval_human_normalized_score_max", "rank_best_eval_human_normalized_score"),
        ("latest_eval_human_normalized_score_mean", "rank_latest_eval_human_normalized_score"),
    )

    for _, rank_field in rank_specs:
        for aggregate in aggregates:
            aggregate[rank_field] = None

    for metric_field, rank_field in rank_specs:
        ranked_records = [
            (index, aggregate, float(aggregate[metric_field]))
            for index, aggregate in enumerate(aggregates)
            if aggregate.get(metric_field) is not None
        ]
        for rank, (_, aggregate, _) in enumerate(
            sorted(ranked_records, key=lambda item: item[2], reverse=True),
            start=1,
        ):
            aggregate[rank_field] = rank
    return aggregates


def _attach_baseline_comparison_fields(
    aggregates: list[dict[str, Any]],
    *,
    baseline_preset: str,
) -> list[dict[str, Any]]:
    baseline_aggregate = next(
        (aggregate for aggregate in aggregates if str(aggregate.get("preset_name")) == baseline_preset),
        None,
    )
    if baseline_aggregate is None:
        raise ValueError(f"--baseline-preset {baseline_preset!r} was not found in the aggregate preset groups")

    baseline_return_mean = baseline_aggregate.get("latest_eval_return_mean_mean")
    baseline_normalized_mean = baseline_aggregate.get("latest_eval_human_normalized_score_mean")

    for aggregate in aggregates:
        aggregate["baseline_preset"] = baseline_preset
        aggregate["baseline_latest_eval_return_mean_mean"] = baseline_return_mean
        aggregate["baseline_latest_eval_human_normalized_score_mean"] = baseline_normalized_mean
        aggregate["delta_vs_baseline_latest_eval_return_mean_mean"] = _difference(
            aggregate.get("latest_eval_return_mean_mean"),
            baseline_return_mean,
        )
        aggregate["delta_vs_baseline_latest_eval_human_normalized_score_mean"] = _difference(
            aggregate.get("latest_eval_human_normalized_score_mean"),
            baseline_normalized_mean,
        )
        aggregate["ratio_vs_baseline_latest_eval_return_mean_mean"] = _ratio(
            aggregate.get("latest_eval_return_mean_mean"),
            baseline_return_mean,
        )
        aggregate["ratio_vs_baseline_latest_eval_human_normalized_score_mean"] = _ratio(
            aggregate.get("latest_eval_human_normalized_score_mean"),
            baseline_normalized_mean,
        )
    return aggregates


def _baseline_summary_entry(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "group": aggregate.get("group"),
        "preset_name": aggregate.get("preset_name"),
        "preset_metadata": aggregate.get("preset_metadata"),
        "preset_config": aggregate.get("preset_config"),
        "preset_description": aggregate.get("preset_description"),
        "algo": aggregate.get("algo"),
        "env_id": aggregate.get("env_id"),
        "baseline_preset": aggregate.get("baseline_preset"),
        "latest_eval_return_mean_mean": aggregate.get("latest_eval_return_mean_mean"),
        "latest_eval_human_normalized_score_mean": aggregate.get("latest_eval_human_normalized_score_mean"),
        "baseline_latest_eval_return_mean_mean": aggregate.get("baseline_latest_eval_return_mean_mean"),
        "baseline_latest_eval_human_normalized_score_mean": aggregate.get("baseline_latest_eval_human_normalized_score_mean"),
        "delta_vs_baseline_latest_eval_return_mean_mean": aggregate.get("delta_vs_baseline_latest_eval_return_mean_mean"),
        "delta_vs_baseline_latest_eval_human_normalized_score_mean": aggregate.get(
            "delta_vs_baseline_latest_eval_human_normalized_score_mean"
        ),
        "ratio_vs_baseline_latest_eval_return_mean_mean": aggregate.get("ratio_vs_baseline_latest_eval_return_mean_mean"),
        "ratio_vs_baseline_latest_eval_human_normalized_score_mean": aggregate.get(
            "ratio_vs_baseline_latest_eval_human_normalized_score_mean"
        ),
        "manifest_alignment_status": aggregate.get("manifest_alignment_status"),
        "manifest_alignment_all_runs": aggregate.get("manifest_alignment_all_runs"),
        "manifest_alignment_total_runs": aggregate.get("manifest_alignment_total_runs"),
        "manifest_alignment_aligned_runs": aggregate.get("manifest_alignment_aligned_runs"),
        "manifest_alignment_drifted_runs": aggregate.get("manifest_alignment_drifted_runs"),
        "manifest_alignment_unknown_preset_runs": aggregate.get("manifest_alignment_unknown_preset_runs"),
        "manifest_alignment_protocol_mismatch_runs": aggregate.get("manifest_alignment_protocol_mismatch_runs"),
    }


def _build_baseline_summary(
    aggregates: list[dict[str, Any]],
    *,
    baseline_preset: str,
    limit: int = 3,
) -> dict[str, Any]:
    comparison_candidates = [
        aggregate for aggregate in aggregates if str(aggregate.get("preset_name")) != baseline_preset
    ]

    def top_entries(metric_field: str, *, descending: bool) -> list[dict[str, Any]]:
        ordered = _sort_records(comparison_candidates, sort_by=metric_field, descending=descending)
        return [_baseline_summary_entry(entry) for entry in ordered[:limit]]

    return {
        "baseline_preset": baseline_preset,
        "top_movers_by_return_delta": top_entries(
            "delta_vs_baseline_latest_eval_return_mean_mean",
            descending=True,
        ),
        "top_regressions_by_return_delta": top_entries(
            "delta_vs_baseline_latest_eval_return_mean_mean",
            descending=False,
        ),
        "top_movers_by_normalized_delta": top_entries(
            "delta_vs_baseline_latest_eval_human_normalized_score_mean",
            descending=True,
        ),
        "top_regressions_by_normalized_delta": top_entries(
            "delta_vs_baseline_latest_eval_human_normalized_score_mean",
            descending=False,
        ),
    }


def _aggregate_run_reports(
    reports: list[dict[str, Any]],
    *,
    group_by: str = "algo-env",
    min_seeds: int | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for report in reports:
        if group_by == "preset":
            preset_name = report.get("preset_name")
            key = (report.get("suite"), preset_name, report.get("protocol_name"))
        else:
            key = (str(report["algo"]), str(report["env_id"]))
        grouped.setdefault(key, []).append(report)

    aggregates: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple("" if value is None else str(value) for value in item[0])):
        ordered_group = sorted(group, key=lambda report: (_seed_sort_key(report["seed"]), str(report["run_id"])))
        if min_seeds is not None and len(ordered_group) < min_seeds:
            continue
        seeds = ",".join(str(report["seed"]) for report in ordered_group)
        latest_return_values = [report["latest_eval_return_mean"] for report in ordered_group]
        latest_normalized_values = [report["latest_eval_human_normalized_score"] for report in ordered_group]
        aggregate = {
            "group_by": group_by,
            "suite": _single_or_multiple([report.get("suite") for report in ordered_group]),
            "preset_name": _single_or_multiple([report.get("preset_name") for report in ordered_group]),
            "protocol_name": _single_or_multiple([report.get("protocol_name") for report in ordered_group]),
            "algo": _single_or_multiple([report["algo"] for report in ordered_group]),
            "env_id": _single_or_multiple([report["env_id"] for report in ordered_group]),
            "runs": len(ordered_group),
            "seed_count": len(ordered_group),
            "seeds": seeds,
            "latest_eval_return_mean_mean": _mean(latest_return_values),
            "latest_eval_return_mean_median": _median(latest_return_values),
            "latest_eval_return_mean_min": _min_value(latest_return_values),
            "latest_eval_return_mean_max": _max_value(latest_return_values),
            "latest_eval_return_mean_iqr": _iqr(latest_return_values),
            "latest_eval_return_mean_std": _stddev(latest_return_values),
            "latest_eval_return_mean_stderr": _stderr(latest_return_values),
            "latest_eval_return_mean_ci95": _ci95(latest_return_values),
            "latest_eval_human_normalized_score_mean": _mean(latest_normalized_values),
            "latest_eval_human_normalized_score_median": _median(latest_normalized_values),
            "latest_eval_human_normalized_score_min": _min_value(latest_normalized_values),
            "latest_eval_human_normalized_score_max": _max_value(latest_normalized_values),
            "latest_eval_human_normalized_score_iqr": _iqr(latest_normalized_values),
            "latest_eval_human_normalized_score_std": _stddev(latest_normalized_values),
            "latest_eval_human_normalized_score_stderr": _stderr(latest_normalized_values),
            "latest_eval_human_normalized_score_ci95": _ci95(latest_normalized_values),
            "best_eval_return_mean_max": _max_value([report["best_eval_return_mean"] for report in ordered_group]),
            "best_eval_human_normalized_score_max": _max_value(
                [report["best_eval_human_normalized_score"] for report in ordered_group]
            ),
        }
        manifest_alignment_total_runs = len(ordered_group)
        manifest_alignment_aligned_runs = sum(
            1 for report in ordered_group if report.get("manifest_alignment_status") == "aligned"
        )
        aggregate["manifest_alignment_total_runs"] = manifest_alignment_total_runs
        aggregate["manifest_alignment_aligned_runs"] = manifest_alignment_aligned_runs
        aggregate["manifest_alignment_drifted_runs"] = (
            manifest_alignment_total_runs - manifest_alignment_aligned_runs
        )
        aggregate["manifest_alignment_unknown_preset_runs"] = sum(
            1 for report in ordered_group if report.get("manifest_preset_known") is False
        )
        aggregate["manifest_alignment_protocol_mismatch_runs"] = sum(
            1 for report in ordered_group if report.get("manifest_protocol_matches_manifest") is False
        )
        aggregate["manifest_alignment_all_runs"] = aggregate["manifest_alignment_drifted_runs"] == 0
        aggregate["manifest_alignment_status"] = (
            "aligned" if aggregate["manifest_alignment_all_runs"] else "mixed"
        )
        aggregate["manifest_alignment_severity"] = _build_manifest_alignment_severity(
            drifted_runs=aggregate["manifest_alignment_drifted_runs"],
            unknown_preset_runs=aggregate["manifest_alignment_unknown_preset_runs"],
            protocol_mismatch_runs=aggregate["manifest_alignment_protocol_mismatch_runs"],
        )
        aggregate["best_minus_latest_eval_return_mean_gap"] = _difference(
            aggregate["best_eval_return_mean_max"],
            aggregate["latest_eval_return_mean_mean"],
        )
        aggregate["best_minus_latest_eval_human_normalized_score_gap"] = _difference(
            aggregate["best_eval_human_normalized_score_max"],
            aggregate["latest_eval_human_normalized_score_mean"],
        )
        aggregate["best_over_latest_eval_return_ratio"] = _ratio(
            aggregate["best_eval_return_mean_max"],
            aggregate["latest_eval_return_mean_mean"],
        )
        aggregate["best_over_latest_eval_human_normalized_score_ratio"] = _ratio(
            aggregate["best_eval_human_normalized_score_max"],
            aggregate["latest_eval_human_normalized_score_mean"],
        )
        if group_by == "preset":
            aggregate["group"] = aggregate["preset_name"]
        else:
            aggregate["group"] = f"{aggregate['algo']}::{aggregate['env_id']}"
        aggregates.append(aggregate)
    return _attach_aggregate_rank_fields(aggregates)


def _filter_run_reports(
    reports: list[dict[str, Any]],
    *,
    algo: str | None = None,
    env_id: str | None = None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for report in reports:
        if algo is not None and str(report["algo"]) != algo:
            continue
        if env_id is not None and str(report["env_id"]) != env_id:
            continue
        filtered.append(report)
    return filtered


def _resolve_sort_value(record: Mapping[str, object], sort_by: str) -> object | None:
    for candidate in (sort_by, f"{sort_by}_mean", f"{sort_by}_max"):
        if candidate in record and record[candidate] is not None:
            return record[candidate]
    return None


def _sortable_value(value: object) -> tuple[int, float | str]:
    if isinstance(value, (int, float)):
        return (0, float(value))

    value_text = str(value)
    try:
        return (0, float(value_text))
    except ValueError:
        return (1, value_text.lower())


def _sort_records(
    records: list[dict[str, Any]],
    *,
    sort_by: str | None = None,
    descending: bool = False,
) -> list[dict[str, Any]]:
    if not sort_by:
        return list(records)

    with_values: list[tuple[int, dict[str, Any], tuple[int, float | str]]] = []
    without_values: list[tuple[int, dict[str, Any]]] = []
    for index, record in enumerate(records):
        resolved = _resolve_sort_value(record, sort_by)
        if resolved is None:
            without_values.append((index, record))
            continue
        with_values.append((index, record, _sortable_value(resolved)))

    ordered = sorted(with_values, key=lambda item: (item[2], item[0]), reverse=descending)
    return [record for _, record, _ in ordered] + [record for _, record in without_values]


def _apply_top_k(records: list[dict[str, Any]], *, top_k: int | None = None) -> list[dict[str, Any]]:
    if top_k is None:
        return list(records)
    return list(records[:top_k])


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
    sorted_reports = _sort_records(
        _filter_run_reports(_iter_run_reports(runs_dir), algo=algo, env_id=env_id),
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
    reports = _apply_top_k(enriched_sorted_reports, top_k=top_k)
    aggregate_records = [
        _attach_manifest_metadata(
            aggregate,
            protocol_metadata=protocol_metadata,
            manifest_protocol_name=manifest_protocol_name,
            score_normalization_metadata=score_normalization_metadata,
            preset_lookup=preset_lookup,
        )
        for aggregate in _aggregate_run_reports(enriched_sorted_reports, group_by=group_by, min_seeds=min_seeds)
    ]
    baseline_summary: dict[str, Any] | None = None
    manifest_alignment_summary = _build_manifest_alignment_summary(enriched_sorted_reports)
    if baseline_preset is not None:
        if group_by != "preset":
            raise ValueError("--baseline-preset requires --group-by preset")
        aggregate_records = _attach_baseline_comparison_fields(
            aggregate_records,
            baseline_preset=baseline_preset,
        )
        baseline_summary = _build_baseline_summary(
            aggregate_records,
            baseline_preset=baseline_preset,
        )
    aggregates = _apply_top_k(
        _sort_records(
            aggregate_records,
            sort_by=sort_by,
            descending=descending,
        ),
        top_k=top_k,
    )

    payload: dict[str, Any] = {
        "suite": suite,
        "protocol": protocol.get("name", "unknown") if isinstance(protocol, dict) else "unknown",
        "score_normalization": score_normalization.get("type", "none") if isinstance(score_normalization, dict) else "none",
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


def _render_text_report(payload: Mapping[str, Any]) -> str:
    reports = payload["runs"]
    aggregates = payload["aggregates"]
    lines = [
        f"suite={payload['suite']}",
        f"protocol={payload['protocol']}",
        f"score_normalization={payload['score_normalization']}",
        f"runs_dir={payload['runs_dir']}",
    ]
    metadata_fields = _build_payload_metadata_fields(payload)
    if metadata_fields["manifest_requested_path"] is not None:
        lines.append(f"manifest_requested_path={metadata_fields['manifest_requested_path']}")
    if metadata_fields["manifest_resolved_path"] is not None:
        lines.append(f"manifest_resolved_path={metadata_fields['manifest_resolved_path']}")
    if metadata_fields["manifest_source_kind"] is not None:
        lines.append(f"manifest_source_kind={metadata_fields['manifest_source_kind']}")
    if metadata_fields["manifest_fingerprint"] is not None:
        lines.append(f"manifest_fingerprint={metadata_fields['manifest_fingerprint']}")
    if metadata_fields["manifest_preset_count"] is not None:
        lines.append(f"manifest_preset_count={metadata_fields['manifest_preset_count']}")
    if metadata_fields["manifest_alignment_total_runs"] is not None:
        lines.append(f"manifest_alignment_total_runs={metadata_fields['manifest_alignment_total_runs']}")
    if metadata_fields["manifest_alignment_aligned_runs"] is not None:
        lines.append(f"manifest_alignment_aligned_runs={metadata_fields['manifest_alignment_aligned_runs']}")
    if metadata_fields["manifest_alignment_drifted_runs"] is not None:
        lines.append(f"manifest_alignment_drifted_runs={metadata_fields['manifest_alignment_drifted_runs']}")
    if metadata_fields["manifest_alignment_unknown_preset_runs"] is not None:
        lines.append(
            f"manifest_alignment_unknown_preset_runs={metadata_fields['manifest_alignment_unknown_preset_runs']}"
        )
    if metadata_fields["manifest_alignment_protocol_mismatch_runs"] is not None:
        lines.append(
            "manifest_alignment_protocol_mismatch_runs="
            f"{metadata_fields['manifest_alignment_protocol_mismatch_runs']}"
        )
    if metadata_fields["manifest_alignment_all_runs"] is not None:
        lines.append(f"manifest_alignment_all_runs={metadata_fields['manifest_alignment_all_runs']}")
    if metadata_fields["manifest_alignment_severity"] is not None:
        lines.append(f"manifest_alignment_severity={metadata_fields['manifest_alignment_severity']}")
    if metadata_fields["manifest_alignment_fail_reasons"] is not None:
        lines.append(f"manifest_alignment_fail_reasons={metadata_fields['manifest_alignment_fail_reasons']}")
    if metadata_fields["protocol_description"] is not None:
        lines.append(f"protocol_description={metadata_fields['protocol_description']}")
    if metadata_fields["protocol_training"] is not None:
        lines.append(f"protocol_training={metadata_fields['protocol_training']}")
    if metadata_fields["protocol_evaluation"] is not None:
        lines.append(f"protocol_evaluation={metadata_fields['protocol_evaluation']}")
    if metadata_fields["score_normalization_game"] is not None:
        lines.append(f"score_normalization_game={metadata_fields['score_normalization_game']}")
    if metadata_fields["score_normalization_source"] is not None:
        lines.append(f"score_normalization_source={metadata_fields['score_normalization_source']}")
    if metadata_fields["score_normalization_random_score"] is not None:
        lines.append(f"score_normalization_random_score={metadata_fields['score_normalization_random_score']}")
    if metadata_fields["score_normalization_human_score"] is not None:
        lines.append(f"score_normalization_human_score={metadata_fields['score_normalization_human_score']}")
    if metadata_fields["score_normalization_scale"] is not None:
        lines.append(f"score_normalization_scale={metadata_fields['score_normalization_scale']}")
    if payload.get("baseline_preset") is not None:
        lines.append(f"baseline_preset={payload['baseline_preset']}")

    for report in reports:
        lines.append(
            " ".join(
                [
                    f"run_id={report['run_id']}",
                    f"algo={report['algo']}",
                    f"env_id={report['env_id']}",
                    f"seed={report['seed']}",
                    f"preset_name={report['preset_name']}",
                    f"preset_config={report.get('preset_config')}",
                    f"preset_description={report.get('preset_description')}",
                    f"protocol_name={report['protocol_name']}",
                    f"manifest_preset_known={report.get('manifest_preset_known')}",
                    f"manifest_protocol_matches_manifest={report.get('manifest_protocol_matches_manifest')}",
                    f"manifest_alignment_status={report.get('manifest_alignment_status')}",
                    f"manifest_alignment_severity={report.get('manifest_alignment_severity')}",
                    f"latest_eval_return_mean={report['latest_eval_return_mean']}",
                    f"latest_eval_human_normalized_score={report['latest_eval_human_normalized_score']}",
                    f"best_eval_return_mean={report['best_eval_return_mean']}",
                    f"best_eval_human_normalized_score={report['best_eval_human_normalized_score']}",
                    f"best_minus_latest_eval_return_mean={report['best_minus_latest_eval_return_mean']}",
                    f"best_minus_latest_eval_human_normalized_score={report['best_minus_latest_eval_human_normalized_score']}",
                    f"best_checkpoint_path={report['best_checkpoint_path']}",
                ]
            )
        )

    for aggregate in aggregates:
        lines.append(
            " ".join(
                [
                    "aggregate",
                    f"algo={aggregate['algo']}",
                    f"env_id={aggregate['env_id']}",
                    f"runs={aggregate['runs']}",
                    f"seeds={aggregate['seeds']}",
                    f"seed_count={aggregate['seed_count']}",
                    f"group_by={aggregate['group_by']}",
                    f"group={aggregate['group']}",
                    f"preset_name={aggregate['preset_name']}",
                    f"preset_config={aggregate.get('preset_config')}",
                    f"preset_description={aggregate.get('preset_description')}",
                    f"baseline_preset={aggregate.get('baseline_preset')}",
                    f"protocol_name={aggregate['protocol_name']}",
                    f"manifest_alignment_status={aggregate.get('manifest_alignment_status')}",
                    f"manifest_alignment_severity={aggregate.get('manifest_alignment_severity')}",
                    f"manifest_alignment_all_runs={aggregate.get('manifest_alignment_all_runs')}",
                    f"manifest_alignment_total_runs={aggregate.get('manifest_alignment_total_runs')}",
                    f"manifest_alignment_aligned_runs={aggregate.get('manifest_alignment_aligned_runs')}",
                    f"manifest_alignment_drifted_runs={aggregate.get('manifest_alignment_drifted_runs')}",
                    f"manifest_alignment_unknown_preset_runs={aggregate.get('manifest_alignment_unknown_preset_runs')}",
                    "manifest_alignment_protocol_mismatch_runs="
                    f"{aggregate.get('manifest_alignment_protocol_mismatch_runs')}",
                    f"latest_eval_return_mean_mean={aggregate['latest_eval_return_mean_mean']}",
                    f"latest_eval_return_mean_median={aggregate['latest_eval_return_mean_median']}",
                    f"latest_eval_return_mean_min={aggregate['latest_eval_return_mean_min']}",
                    f"latest_eval_return_mean_max={aggregate['latest_eval_return_mean_max']}",
                    f"latest_eval_return_mean_iqr={aggregate['latest_eval_return_mean_iqr']}",
                    f"latest_eval_return_mean_std={aggregate['latest_eval_return_mean_std']}",
                    f"latest_eval_return_mean_stderr={aggregate['latest_eval_return_mean_stderr']}",
                    f"latest_eval_return_mean_ci95={aggregate['latest_eval_return_mean_ci95']}",
                    f"latest_eval_human_normalized_score_mean={aggregate['latest_eval_human_normalized_score_mean']}",
                    f"latest_eval_human_normalized_score_median={aggregate['latest_eval_human_normalized_score_median']}",
                    f"latest_eval_human_normalized_score_min={aggregate['latest_eval_human_normalized_score_min']}",
                    f"latest_eval_human_normalized_score_max={aggregate['latest_eval_human_normalized_score_max']}",
                    f"latest_eval_human_normalized_score_iqr={aggregate['latest_eval_human_normalized_score_iqr']}",
                    f"latest_eval_human_normalized_score_std={aggregate['latest_eval_human_normalized_score_std']}",
                    f"latest_eval_human_normalized_score_stderr={aggregate['latest_eval_human_normalized_score_stderr']}",
                    f"latest_eval_human_normalized_score_ci95={aggregate['latest_eval_human_normalized_score_ci95']}",
                    f"best_eval_return_mean_max={aggregate['best_eval_return_mean_max']}",
                    f"best_eval_human_normalized_score_max={aggregate['best_eval_human_normalized_score_max']}",
                    f"best_minus_latest_eval_return_mean_gap={aggregate['best_minus_latest_eval_return_mean_gap']}",
                    f"best_minus_latest_eval_human_normalized_score_gap={aggregate['best_minus_latest_eval_human_normalized_score_gap']}",
                    f"best_over_latest_eval_return_ratio={aggregate['best_over_latest_eval_return_ratio']}",
                    f"best_over_latest_eval_human_normalized_score_ratio={aggregate['best_over_latest_eval_human_normalized_score_ratio']}",
                    f"baseline_latest_eval_return_mean_mean={aggregate.get('baseline_latest_eval_return_mean_mean')}",
                    f"baseline_latest_eval_human_normalized_score_mean={aggregate.get('baseline_latest_eval_human_normalized_score_mean')}",
                    f"delta_vs_baseline_latest_eval_return_mean_mean={aggregate.get('delta_vs_baseline_latest_eval_return_mean_mean')}",
                    f"delta_vs_baseline_latest_eval_human_normalized_score_mean={aggregate.get('delta_vs_baseline_latest_eval_human_normalized_score_mean')}",
                    f"ratio_vs_baseline_latest_eval_return_mean_mean={aggregate.get('ratio_vs_baseline_latest_eval_return_mean_mean')}",
                    f"ratio_vs_baseline_latest_eval_human_normalized_score_mean={aggregate.get('ratio_vs_baseline_latest_eval_human_normalized_score_mean')}",
                    f"rank_best_eval_return_mean={aggregate['rank_best_eval_return_mean']}",
                    f"rank_latest_eval_return_mean={aggregate['rank_latest_eval_return_mean']}",
                    f"rank_best_eval_human_normalized_score={aggregate['rank_best_eval_human_normalized_score']}",
                    f"rank_latest_eval_human_normalized_score={aggregate['rank_latest_eval_human_normalized_score']}",
                ]
            )
        )

    summary = payload.get("baseline_summary")
    if isinstance(summary, Mapping):
        for summary_kind in (
            "top_movers_by_return_delta",
            "top_regressions_by_return_delta",
            "top_movers_by_normalized_delta",
            "top_regressions_by_normalized_delta",
        ):
            entries = summary.get(summary_kind, [])
            if not isinstance(entries, list):
                continue
            for rank, entry in enumerate(entries, start=1):
                lines.append(
                    " ".join(
                        [
                            "summary",
                            f"baseline_preset={summary.get('baseline_preset')}",
                            f"summary_kind={summary_kind}",
                            f"summary_rank={rank}",
                            f"preset_name={entry.get('preset_name')}",
                            f"preset_config={entry.get('preset_config')}",
                            f"preset_description={entry.get('preset_description')}",
                            f"algo={entry.get('algo')}",
                            f"env_id={entry.get('env_id')}",
                            f"manifest_alignment_status={entry.get('manifest_alignment_status')}",
                            f"manifest_alignment_severity={entry.get('manifest_alignment_severity')}",
                            f"manifest_alignment_all_runs={entry.get('manifest_alignment_all_runs')}",
                            f"manifest_alignment_total_runs={entry.get('manifest_alignment_total_runs')}",
                            f"manifest_alignment_aligned_runs={entry.get('manifest_alignment_aligned_runs')}",
                            f"manifest_alignment_drifted_runs={entry.get('manifest_alignment_drifted_runs')}",
                            f"manifest_alignment_unknown_preset_runs={entry.get('manifest_alignment_unknown_preset_runs')}",
                            "manifest_alignment_protocol_mismatch_runs="
                            f"{entry.get('manifest_alignment_protocol_mismatch_runs')}",
                            f"delta_vs_baseline_latest_eval_return_mean_mean={entry.get('delta_vs_baseline_latest_eval_return_mean_mean')}",
                            f"delta_vs_baseline_latest_eval_human_normalized_score_mean={entry.get('delta_vs_baseline_latest_eval_human_normalized_score_mean')}",
                            f"ratio_vs_baseline_latest_eval_return_mean_mean={entry.get('ratio_vs_baseline_latest_eval_return_mean_mean')}",
                            f"ratio_vs_baseline_latest_eval_human_normalized_score_mean={entry.get('ratio_vs_baseline_latest_eval_human_normalized_score_mean')}",
                        ]
                    )
                )
    return "\n".join(lines) + "\n"


def _render_json_report(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2) + "\n"


def _csv_report_rows(payload: Mapping[str, Any]) -> list[dict[str, object]]:
    common = {
        "suite": payload["suite"],
        "protocol": payload["protocol"],
        "score_normalization": payload["score_normalization"],
        "runs_dir": payload["runs_dir"],
        "baseline_preset": payload.get("baseline_preset"),
    }
    common.update(_build_payload_metadata_fields(payload))
    rows: list[dict[str, object]] = []
    for report in payload["runs"]:
        row = dict(common)
        row["kind"] = "run"
        row.update(report)
        rows.append(row)
    for aggregate in payload["aggregates"]:
        row = dict(common)
        row["kind"] = "aggregate"
        row.update(aggregate)
        rows.append(row)
    summary = payload.get("baseline_summary")
    if isinstance(summary, Mapping):
        for summary_kind in (
            "top_movers_by_return_delta",
            "top_regressions_by_return_delta",
            "top_movers_by_normalized_delta",
            "top_regressions_by_normalized_delta",
        ):
            entries = summary.get(summary_kind, [])
            if not isinstance(entries, list):
                continue
            for rank, entry in enumerate(entries, start=1):
                row = dict(common)
                row["kind"] = "summary"
                row["summary_kind"] = summary_kind
                row["summary_rank"] = rank
                row.update(entry)
                rows.append(row)
    return rows


def _render_csv_report(payload: Mapping[str, Any]) -> str:
    rows = _csv_report_rows(payload)
    fieldnames = [
        "kind",
        "summary_kind",
        "summary_rank",
        "suite",
        "manifest_requested_path",
        "manifest_resolved_path",
        "manifest_source_kind",
        "manifest_fingerprint",
        "manifest_preset_count",
        "manifest_preset_names",
        "manifest_alignment_total_runs",
        "manifest_alignment_aligned_runs",
        "manifest_alignment_drifted_runs",
        "manifest_alignment_unknown_preset_runs",
        "manifest_alignment_protocol_mismatch_runs",
        "manifest_alignment_all_runs",
        "manifest_alignment_severity",
        "manifest_alignment_drifted_presets",
        "manifest_alignment_fail_reasons",
        "protocol",
        "protocol_description",
        "protocol_training",
        "protocol_evaluation",
        "score_normalization",
        "score_normalization_game",
        "score_normalization_source",
        "score_normalization_random_score",
        "score_normalization_human_score",
        "score_normalization_scale",
        "runs_dir",
        "baseline_preset",
        "run_id",
        "algo",
        "env_id",
        "seed",
        "preset_name",
        "preset_config",
        "preset_description",
        "protocol_name",
        "manifest_preset_known",
        "manifest_protocol_matches_manifest",
        "manifest_alignment_status",
        "manifest_alignment_severity",
        "latest_eval_return_mean",
        "latest_eval_human_normalized_score",
        "best_eval_return_mean",
        "best_eval_human_normalized_score",
        "best_minus_latest_eval_return_mean",
        "best_minus_latest_eval_human_normalized_score",
        "best_checkpoint_path",
        "group_by",
        "group",
        "runs",
        "seed_count",
        "seeds",
        "latest_eval_return_mean_mean",
        "latest_eval_return_mean_median",
        "latest_eval_return_mean_min",
        "latest_eval_return_mean_max",
        "latest_eval_return_mean_iqr",
        "latest_eval_return_mean_std",
        "latest_eval_return_mean_stderr",
        "latest_eval_return_mean_ci95",
        "latest_eval_human_normalized_score_mean",
        "latest_eval_human_normalized_score_median",
        "latest_eval_human_normalized_score_min",
        "latest_eval_human_normalized_score_max",
        "latest_eval_human_normalized_score_iqr",
        "latest_eval_human_normalized_score_std",
        "latest_eval_human_normalized_score_stderr",
        "latest_eval_human_normalized_score_ci95",
        "best_eval_return_mean_max",
        "best_eval_human_normalized_score_max",
        "best_minus_latest_eval_return_mean_gap",
        "best_minus_latest_eval_human_normalized_score_gap",
        "best_over_latest_eval_return_ratio",
        "best_over_latest_eval_human_normalized_score_ratio",
        "baseline_latest_eval_return_mean_mean",
        "baseline_latest_eval_human_normalized_score_mean",
        "delta_vs_baseline_latest_eval_return_mean_mean",
        "delta_vs_baseline_latest_eval_human_normalized_score_mean",
        "ratio_vs_baseline_latest_eval_return_mean_mean",
        "ratio_vs_baseline_latest_eval_human_normalized_score_mean",
        "rank_best_eval_return_mean",
        "rank_latest_eval_return_mean",
        "rank_best_eval_human_normalized_score",
        "rank_latest_eval_human_normalized_score",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _render_commands_output(manifest: Mapping[str, Any]) -> str:
    presets = manifest.get("presets", [])
    if not isinstance(presets, list):
        raise TypeError("manifest 'presets' must be a list")
    lines = [f"axiomrl train --config {preset['config']}" for preset in presets]
    return "\n".join(lines) + ("\n" if lines else "")


def _render_table_output(manifest: Mapping[str, Any]) -> str:
    presets = manifest.get("presets", [])
    if not isinstance(presets, list):
        raise TypeError("manifest 'presets' must be a list")
    lines = [f"suite={manifest.get('suite', 'unknown')}"]
    for preset in presets:
        lines.append(f"{preset['name']}: {preset['config']}")
    return "\n".join(lines) + "\n"


def _emit_output(content: str, *, output_path: str | Path | None = None) -> None:
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    print(content, end="")


LEADERBOARD_METRIC_CHOICES = (
    "best-return",
    "latest-return",
    "gap-return",
    "stability-return",
    "confidence-return",
    "median-return",
    "iqr-return",
    "delta-vs-baseline-return",
    "ratio-vs-baseline-return",
    "best-normalized",
    "latest-normalized",
    "gap-normalized",
    "stability-normalized",
    "confidence-normalized",
    "median-normalized",
    "iqr-normalized",
    "delta-vs-baseline-normalized",
    "ratio-vs-baseline-normalized",
)
COMPARE_TO_CHOICES = ("best", "latest")
SCORE_VIEW_CHOICES = ("return", "normalized")


def _resolve_leaderboard_metric_alias(
    manifest: Mapping[str, Any],
    *,
    leaderboard_metric: str | None = None,
    baseline_preset: str | None = None,
    compare_to: str | None = None,
    score_view: str | None = None,
    sort_by: str | None = None,
    descending: bool = False,
) -> tuple[str | None, str | None, str | None, str, bool]:
    metric_to_sort_by = {
        "best-return": ("best_eval_return_mean", True),
        "latest-return": ("latest_eval_return_mean", True),
        "gap-return": ("best_minus_latest_eval_return_mean_gap", True),
        "stability-return": ("latest_eval_return_mean_std", False),
        "confidence-return": ("latest_eval_return_mean_ci95", False),
        "median-return": ("latest_eval_return_mean_median", True),
        "iqr-return": ("latest_eval_return_mean_iqr", False),
        "delta-vs-baseline-return": ("delta_vs_baseline_latest_eval_return_mean_mean", True),
        "ratio-vs-baseline-return": ("ratio_vs_baseline_latest_eval_return_mean_mean", True),
        "best-normalized": ("best_eval_human_normalized_score", True),
        "latest-normalized": ("latest_eval_human_normalized_score", True),
        "gap-normalized": ("best_minus_latest_eval_human_normalized_score_gap", True),
        "stability-normalized": ("latest_eval_human_normalized_score_std", False),
        "confidence-normalized": ("latest_eval_human_normalized_score_ci95", False),
        "median-normalized": ("latest_eval_human_normalized_score_median", True),
        "iqr-normalized": ("latest_eval_human_normalized_score_iqr", False),
        "delta-vs-baseline-normalized": ("delta_vs_baseline_latest_eval_human_normalized_score_mean", True),
        "ratio-vs-baseline-normalized": ("ratio_vs_baseline_latest_eval_human_normalized_score_mean", True),
    }
    baseline_metric_choices = {
        "delta-vs-baseline-return",
        "ratio-vs-baseline-return",
        "delta-vs-baseline-normalized",
        "ratio-vs-baseline-normalized",
    }
    if leaderboard_metric is not None:
        if leaderboard_metric in baseline_metric_choices and baseline_preset is None:
            raise ValueError(f"--leaderboard-metric {leaderboard_metric} requires --baseline-preset")
        resolved_sort_by, resolved_descending = metric_to_sort_by[leaderboard_metric]
        return None, None, leaderboard_metric, resolved_sort_by, resolved_descending

    if sort_by is not None:
        return None, None, None, sort_by, descending

    score_normalization = manifest.get("score_normalization", {})
    prefers_normalized_scores = (
        isinstance(score_normalization, dict)
        and str(score_normalization.get("type", "none")).lower() != "none"
    )
    resolved_compare_to = compare_to if compare_to is not None else "best"
    if score_view == "normalized" and not prefers_normalized_scores:
        raise ValueError("--score-view normalized requires score normalization in the benchmark manifest")
    resolved_score_view = score_view if score_view is not None else ("normalized" if prefers_normalized_scores else "return")
    resolved_metric = f"{resolved_compare_to}-{resolved_score_view}"
    resolved_sort_by, resolved_descending = metric_to_sort_by[resolved_metric]
    return resolved_compare_to, resolved_score_view, resolved_metric, resolved_sort_by, resolved_descending


def build_leaderboard_payload(
    manifest: dict[str, Any],
    *,
    runs_dir: Path,
    manifest_source: Mapping[str, object] | None = None,
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
) -> dict[str, Any]:
    resolved_compare_to, resolved_score_view, resolved_metric, resolved_sort_by, resolved_descending = (
        _resolve_leaderboard_metric_alias(
        manifest,
        leaderboard_metric=leaderboard_metric,
        baseline_preset=baseline_preset,
        compare_to=compare_to,
        score_view=score_view,
        sort_by=sort_by,
        descending=descending,
    ))
    report_payload = build_report_payload(
        manifest,
        runs_dir=runs_dir,
        manifest_source=manifest_source,
        algo=algo,
        env_id=env_id,
        group_by=group_by,
        min_seeds=min_seeds,
        top_k=top_k,
        baseline_preset=baseline_preset,
        sort_by=resolved_sort_by,
        descending=resolved_descending,
    )
    payload: dict[str, Any] = {
        "format": "leaderboard",
        "suite": report_payload["suite"],
        "protocol": report_payload["protocol"],
        "score_normalization": report_payload["score_normalization"],
        "runs_dir": report_payload["runs_dir"],
        "group_by": report_payload["group_by"],
        "sort_by": resolved_sort_by,
        "descending": resolved_descending,
        "entries": report_payload["aggregates"],
    }
    if "manifest_source" in report_payload:
        payload["manifest_source"] = report_payload["manifest_source"]
    if "manifest_metadata" in report_payload:
        payload["manifest_metadata"] = report_payload["manifest_metadata"]
    if "manifest_alignment_summary" in report_payload:
        payload["manifest_alignment_summary"] = report_payload["manifest_alignment_summary"]
    if "protocol_metadata" in report_payload:
        payload["protocol_metadata"] = report_payload["protocol_metadata"]
    if "score_normalization_metadata" in report_payload:
        payload["score_normalization_metadata"] = report_payload["score_normalization_metadata"]
    if "baseline_preset" in report_payload:
        payload["baseline_preset"] = report_payload["baseline_preset"]
    if "baseline_summary" in report_payload:
        payload["baseline_summary"] = report_payload["baseline_summary"]
    if resolved_compare_to is not None:
        payload["compare_to"] = resolved_compare_to
    if resolved_score_view is not None:
        payload["score_view"] = resolved_score_view
    if resolved_metric is not None:
        payload["leaderboard_metric"] = resolved_metric
    if "filters" in report_payload:
        payload["filters"] = report_payload["filters"]
    if "min_seeds" in report_payload:
        payload["min_seeds"] = report_payload["min_seeds"]
    if top_k is not None:
        payload["top_k"] = top_k
    return payload


def _render_text_leaderboard(payload: Mapping[str, Any]) -> str:
    lines = [
        "leaderboard",
        f"suite={payload['suite']}",
        f"protocol={payload['protocol']}",
        f"score_normalization={payload['score_normalization']}",
        f"runs_dir={payload['runs_dir']}",
        f"group_by={payload['group_by']}",
    ]
    metadata_fields = _build_payload_metadata_fields(payload)
    if metadata_fields["manifest_requested_path"] is not None:
        lines.append(f"manifest_requested_path={metadata_fields['manifest_requested_path']}")
    if metadata_fields["manifest_resolved_path"] is not None:
        lines.append(f"manifest_resolved_path={metadata_fields['manifest_resolved_path']}")
    if metadata_fields["manifest_source_kind"] is not None:
        lines.append(f"manifest_source_kind={metadata_fields['manifest_source_kind']}")
    if metadata_fields["manifest_fingerprint"] is not None:
        lines.append(f"manifest_fingerprint={metadata_fields['manifest_fingerprint']}")
    if metadata_fields["manifest_preset_count"] is not None:
        lines.append(f"manifest_preset_count={metadata_fields['manifest_preset_count']}")
    if metadata_fields["manifest_alignment_total_runs"] is not None:
        lines.append(f"manifest_alignment_total_runs={metadata_fields['manifest_alignment_total_runs']}")
    if metadata_fields["manifest_alignment_aligned_runs"] is not None:
        lines.append(f"manifest_alignment_aligned_runs={metadata_fields['manifest_alignment_aligned_runs']}")
    if metadata_fields["manifest_alignment_drifted_runs"] is not None:
        lines.append(f"manifest_alignment_drifted_runs={metadata_fields['manifest_alignment_drifted_runs']}")
    if metadata_fields["manifest_alignment_unknown_preset_runs"] is not None:
        lines.append(
            f"manifest_alignment_unknown_preset_runs={metadata_fields['manifest_alignment_unknown_preset_runs']}"
        )
    if metadata_fields["manifest_alignment_protocol_mismatch_runs"] is not None:
        lines.append(
            "manifest_alignment_protocol_mismatch_runs="
            f"{metadata_fields['manifest_alignment_protocol_mismatch_runs']}"
        )
    if metadata_fields["manifest_alignment_all_runs"] is not None:
        lines.append(f"manifest_alignment_all_runs={metadata_fields['manifest_alignment_all_runs']}")
    if metadata_fields["manifest_alignment_severity"] is not None:
        lines.append(f"manifest_alignment_severity={metadata_fields['manifest_alignment_severity']}")
    if metadata_fields["manifest_alignment_fail_reasons"] is not None:
        lines.append(f"manifest_alignment_fail_reasons={metadata_fields['manifest_alignment_fail_reasons']}")
    if metadata_fields["protocol_description"] is not None:
        lines.append(f"protocol_description={metadata_fields['protocol_description']}")
    if metadata_fields["protocol_training"] is not None:
        lines.append(f"protocol_training={metadata_fields['protocol_training']}")
    if metadata_fields["protocol_evaluation"] is not None:
        lines.append(f"protocol_evaluation={metadata_fields['protocol_evaluation']}")
    if metadata_fields["score_normalization_game"] is not None:
        lines.append(f"score_normalization_game={metadata_fields['score_normalization_game']}")
    if metadata_fields["score_normalization_source"] is not None:
        lines.append(f"score_normalization_source={metadata_fields['score_normalization_source']}")
    if metadata_fields["score_normalization_random_score"] is not None:
        lines.append(f"score_normalization_random_score={metadata_fields['score_normalization_random_score']}")
    if metadata_fields["score_normalization_human_score"] is not None:
        lines.append(f"score_normalization_human_score={metadata_fields['score_normalization_human_score']}")
    if metadata_fields["score_normalization_scale"] is not None:
        lines.append(f"score_normalization_scale={metadata_fields['score_normalization_scale']}")
    if payload.get("baseline_preset") is not None:
        lines.append(f"baseline_preset={payload['baseline_preset']}")
    if payload.get("compare_to") is not None:
        lines.append(f"compare_to={payload['compare_to']}")
    if payload.get("score_view") is not None:
        lines.append(f"score_view={payload['score_view']}")
    if payload.get("leaderboard_metric") is not None:
        lines.append(f"leaderboard_metric={payload['leaderboard_metric']}")
    lines.extend(
        [
            f"sort_by={payload['sort_by']}",
            f"descending={payload['descending']}",
        ]
    )
    if "top_k" in payload:
        lines.append(f"top_k={payload['top_k']}")

    for index, entry in enumerate(payload["entries"], start=1):
        lines.append(
            " ".join(
                [
                    f"rank={index}",
                    f"group={entry['group']}",
                    f"preset_name={entry['preset_name']}",
                    f"preset_config={entry.get('preset_config')}",
                    f"preset_description={entry.get('preset_description')}",
                    f"protocol_name={entry['protocol_name']}",
                    f"manifest_alignment_status={entry.get('manifest_alignment_status')}",
                    f"manifest_alignment_severity={entry.get('manifest_alignment_severity')}",
                    f"manifest_alignment_all_runs={entry.get('manifest_alignment_all_runs')}",
                    f"manifest_alignment_total_runs={entry.get('manifest_alignment_total_runs')}",
                    f"manifest_alignment_aligned_runs={entry.get('manifest_alignment_aligned_runs')}",
                    f"manifest_alignment_drifted_runs={entry.get('manifest_alignment_drifted_runs')}",
                    f"manifest_alignment_unknown_preset_runs={entry.get('manifest_alignment_unknown_preset_runs')}",
                    "manifest_alignment_protocol_mismatch_runs="
                    f"{entry.get('manifest_alignment_protocol_mismatch_runs')}",
                    f"algo={entry['algo']}",
                    f"env_id={entry['env_id']}",
                    f"runs={entry['runs']}",
                    f"seeds={entry['seeds']}",
                    f"seed_count={entry['seed_count']}",
                    f"baseline_preset={entry.get('baseline_preset')}",
                    f"best_eval_return_mean_max={entry['best_eval_return_mean_max']}",
                    f"latest_eval_return_mean_mean={entry['latest_eval_return_mean_mean']}",
                    f"latest_eval_return_mean_median={entry['latest_eval_return_mean_median']}",
                    f"latest_eval_return_mean_min={entry['latest_eval_return_mean_min']}",
                    f"latest_eval_return_mean_max={entry['latest_eval_return_mean_max']}",
                    f"latest_eval_return_mean_iqr={entry['latest_eval_return_mean_iqr']}",
                    f"latest_eval_return_mean_std={entry['latest_eval_return_mean_std']}",
                    f"latest_eval_return_mean_stderr={entry['latest_eval_return_mean_stderr']}",
                    f"latest_eval_return_mean_ci95={entry['latest_eval_return_mean_ci95']}",
                    f"best_minus_latest_eval_return_mean_gap={entry['best_minus_latest_eval_return_mean_gap']}",
                    f"best_over_latest_eval_return_ratio={entry['best_over_latest_eval_return_ratio']}",
                    f"baseline_latest_eval_return_mean_mean={entry.get('baseline_latest_eval_return_mean_mean')}",
                    f"delta_vs_baseline_latest_eval_return_mean_mean={entry.get('delta_vs_baseline_latest_eval_return_mean_mean')}",
                    f"ratio_vs_baseline_latest_eval_return_mean_mean={entry.get('ratio_vs_baseline_latest_eval_return_mean_mean')}",
                    f"rank_best_eval_return_mean={entry['rank_best_eval_return_mean']}",
                    f"rank_latest_eval_return_mean={entry['rank_latest_eval_return_mean']}",
                    f"best_eval_human_normalized_score_max={entry['best_eval_human_normalized_score_max']}",
                    f"latest_eval_human_normalized_score_mean={entry['latest_eval_human_normalized_score_mean']}",
                    f"latest_eval_human_normalized_score_median={entry['latest_eval_human_normalized_score_median']}",
                    f"latest_eval_human_normalized_score_min={entry['latest_eval_human_normalized_score_min']}",
                    f"latest_eval_human_normalized_score_max={entry['latest_eval_human_normalized_score_max']}",
                    f"latest_eval_human_normalized_score_iqr={entry['latest_eval_human_normalized_score_iqr']}",
                    f"latest_eval_human_normalized_score_std={entry['latest_eval_human_normalized_score_std']}",
                    f"latest_eval_human_normalized_score_stderr={entry['latest_eval_human_normalized_score_stderr']}",
                    f"latest_eval_human_normalized_score_ci95={entry['latest_eval_human_normalized_score_ci95']}",
                    f"best_minus_latest_eval_human_normalized_score_gap={entry['best_minus_latest_eval_human_normalized_score_gap']}",
                    f"best_over_latest_eval_human_normalized_score_ratio={entry['best_over_latest_eval_human_normalized_score_ratio']}",
                    f"baseline_latest_eval_human_normalized_score_mean={entry.get('baseline_latest_eval_human_normalized_score_mean')}",
                    f"delta_vs_baseline_latest_eval_human_normalized_score_mean={entry.get('delta_vs_baseline_latest_eval_human_normalized_score_mean')}",
                    f"ratio_vs_baseline_latest_eval_human_normalized_score_mean={entry.get('ratio_vs_baseline_latest_eval_human_normalized_score_mean')}",
                    f"rank_best_eval_human_normalized_score={entry['rank_best_eval_human_normalized_score']}",
                    f"rank_latest_eval_human_normalized_score={entry['rank_latest_eval_human_normalized_score']}",
                ]
            )
        )

    summary = payload.get("baseline_summary")
    if isinstance(summary, Mapping):
        for summary_kind in (
            "top_movers_by_return_delta",
            "top_regressions_by_return_delta",
            "top_movers_by_normalized_delta",
            "top_regressions_by_normalized_delta",
        ):
            entries = summary.get(summary_kind, [])
            if not isinstance(entries, list):
                continue
            for rank, entry in enumerate(entries, start=1):
                lines.append(
                    " ".join(
                        [
                            "summary",
                            f"baseline_preset={summary.get('baseline_preset')}",
                            f"summary_kind={summary_kind}",
                            f"summary_rank={rank}",
                            f"preset_name={entry.get('preset_name')}",
                            f"preset_config={entry.get('preset_config')}",
                            f"preset_description={entry.get('preset_description')}",
                            f"algo={entry.get('algo')}",
                            f"env_id={entry.get('env_id')}",
                            f"manifest_alignment_status={entry.get('manifest_alignment_status')}",
                            f"manifest_alignment_severity={entry.get('manifest_alignment_severity')}",
                            f"manifest_alignment_all_runs={entry.get('manifest_alignment_all_runs')}",
                            f"manifest_alignment_total_runs={entry.get('manifest_alignment_total_runs')}",
                            f"manifest_alignment_aligned_runs={entry.get('manifest_alignment_aligned_runs')}",
                            f"manifest_alignment_drifted_runs={entry.get('manifest_alignment_drifted_runs')}",
                            f"manifest_alignment_unknown_preset_runs={entry.get('manifest_alignment_unknown_preset_runs')}",
                            "manifest_alignment_protocol_mismatch_runs="
                            f"{entry.get('manifest_alignment_protocol_mismatch_runs')}",
                            f"delta_vs_baseline_latest_eval_return_mean_mean={entry.get('delta_vs_baseline_latest_eval_return_mean_mean')}",
                            f"delta_vs_baseline_latest_eval_human_normalized_score_mean={entry.get('delta_vs_baseline_latest_eval_human_normalized_score_mean')}",
                            f"ratio_vs_baseline_latest_eval_return_mean_mean={entry.get('ratio_vs_baseline_latest_eval_return_mean_mean')}",
                            f"ratio_vs_baseline_latest_eval_human_normalized_score_mean={entry.get('ratio_vs_baseline_latest_eval_human_normalized_score_mean')}",
                        ]
                    )
                )
    return "\n".join(lines) + "\n"


def _render_json_leaderboard(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2) + "\n"


def _render_csv_leaderboard(payload: Mapping[str, Any]) -> str:
    rows: list[dict[str, object]] = []
    common = {
        "suite": payload["suite"],
        "protocol": payload["protocol"],
        "score_normalization": payload["score_normalization"],
        "runs_dir": payload["runs_dir"],
        "baseline_preset": payload.get("baseline_preset"),
        "group_by": payload["group_by"],
        "compare_to": payload.get("compare_to"),
        "score_view": payload.get("score_view"),
        "leaderboard_metric": payload.get("leaderboard_metric"),
        "sort_by": payload["sort_by"],
        "descending": payload["descending"],
    }
    common.update(_build_payload_metadata_fields(payload))
    for index, entry in enumerate(payload["entries"], start=1):
        row = dict(common)
        row["kind"] = "leaderboard"
        row["rank"] = index
        row.update(entry)
        rows.append(row)
    summary = payload.get("baseline_summary")
    if isinstance(summary, Mapping):
        for summary_kind in (
            "top_movers_by_return_delta",
            "top_regressions_by_return_delta",
            "top_movers_by_normalized_delta",
            "top_regressions_by_normalized_delta",
        ):
            entries = summary.get(summary_kind, [])
            if not isinstance(entries, list):
                continue
            for rank, entry in enumerate(entries, start=1):
                row = dict(common)
                row["kind"] = "summary"
                row["rank"] = rank
                row["summary_kind"] = summary_kind
                row.update(entry)
                rows.append(row)

    fieldnames = [
        "kind",
        "rank",
        "summary_kind",
        "suite",
        "manifest_requested_path",
        "manifest_resolved_path",
        "manifest_source_kind",
        "manifest_fingerprint",
        "manifest_preset_count",
        "manifest_preset_names",
        "manifest_alignment_total_runs",
        "manifest_alignment_aligned_runs",
        "manifest_alignment_drifted_runs",
        "manifest_alignment_unknown_preset_runs",
        "manifest_alignment_protocol_mismatch_runs",
        "manifest_alignment_all_runs",
        "manifest_alignment_severity",
        "manifest_alignment_drifted_presets",
        "manifest_alignment_fail_reasons",
        "protocol",
        "protocol_description",
        "protocol_training",
        "protocol_evaluation",
        "score_normalization",
        "score_normalization_game",
        "score_normalization_source",
        "score_normalization_random_score",
        "score_normalization_human_score",
        "score_normalization_scale",
        "runs_dir",
        "baseline_preset",
        "group_by",
        "compare_to",
        "score_view",
        "leaderboard_metric",
        "sort_by",
        "descending",
        "group",
        "preset_name",
        "preset_config",
        "preset_description",
        "protocol_name",
        "manifest_preset_known",
        "manifest_protocol_matches_manifest",
        "manifest_alignment_status",
        "manifest_alignment_severity",
        "algo",
        "env_id",
        "runs",
        "seed_count",
        "seeds",
        "latest_eval_return_mean_mean",
        "latest_eval_return_mean_median",
        "latest_eval_return_mean_min",
        "latest_eval_return_mean_max",
        "latest_eval_return_mean_iqr",
        "latest_eval_return_mean_std",
        "latest_eval_return_mean_stderr",
        "latest_eval_return_mean_ci95",
        "latest_eval_human_normalized_score_mean",
        "latest_eval_human_normalized_score_median",
        "latest_eval_human_normalized_score_min",
        "latest_eval_human_normalized_score_max",
        "latest_eval_human_normalized_score_iqr",
        "latest_eval_human_normalized_score_std",
        "latest_eval_human_normalized_score_stderr",
        "latest_eval_human_normalized_score_ci95",
        "best_eval_return_mean_max",
        "best_eval_human_normalized_score_max",
        "best_minus_latest_eval_return_mean_gap",
        "best_minus_latest_eval_human_normalized_score_gap",
        "best_over_latest_eval_return_ratio",
        "best_over_latest_eval_human_normalized_score_ratio",
        "baseline_latest_eval_return_mean_mean",
        "baseline_latest_eval_human_normalized_score_mean",
        "delta_vs_baseline_latest_eval_return_mean_mean",
        "delta_vs_baseline_latest_eval_human_normalized_score_mean",
        "ratio_vs_baseline_latest_eval_return_mean_mean",
        "ratio_vs_baseline_latest_eval_human_normalized_score_mean",
        "rank_best_eval_return_mean",
        "rank_latest_eval_return_mean",
        "rank_best_eval_human_normalized_score",
        "rank_latest_eval_human_normalized_score",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


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
    parser = argparse.ArgumentParser(description="List or print benchmark commands from the RL Training zoo.")
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
