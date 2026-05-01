import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from axiomrl.experiment.benchmarking import resolve_score_normalization_settings
from axiomrl.resources import find_packaged_asset

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
        benchmark_candidate = parent / "benchmark.yaml"
        candidates: list[Path] = []
        if benchmark_candidate.exists():
            candidates.append(benchmark_candidate)
        candidates.extend(sorted(candidate for candidate in parent.glob("*.yaml") if candidate != benchmark_candidate))
        for candidate in candidates:
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
    enriched["preset_description"] = (
        preset_metadata.get("description") if isinstance(preset_metadata, Mapping) else None
    )
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
    protocol_mismatch_runs = sum(1 for report in reports if report.get("manifest_protocol_matches_manifest") is False)
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


__all__ = [
    "MANIFEST_DRIFT_TYPE_CHOICES",
    "MANIFEST_DRIFT_TYPE_TO_SUMMARY_FIELD",
    "apply_manifest_defaults_to_config_payload",
    "load_manifest",
    "load_manifest_with_source",
    "resolve_manifest_source",
]
