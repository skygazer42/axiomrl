import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast


def difference(lhs: object, rhs: object) -> float | None:
    if lhs is None or rhs is None:
        return None
    return float(cast(Any, lhs)) - float(cast(Any, rhs))


def load_run_metadata(metadata_path: Path) -> dict[str, Any]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {metadata_path}, got {type(payload)!r}")
    return payload


def iter_run_reports(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []

    metadata_paths: list[Path] = []
    for child_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        direct_metadata = child_dir / "metadata.json"
        if direct_metadata.exists():
            metadata_paths.append(direct_metadata)
            continue
        metadata_paths.extend(sorted(child_dir.glob("*/metadata.json")))

    reports: list[dict[str, Any]] = []
    for metadata_path in metadata_paths:
        run_dir = metadata_path.parent
        try:
            relative_run_dir = run_dir.relative_to(runs_dir)
            run_id = "/".join(relative_run_dir.parts)
        except ValueError:
            run_id = run_dir.name

        payload = load_run_metadata(metadata_path)
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
                "run_id": run_id,
                "algo": payload.get("algo", "unknown"),
                "env_id": payload.get("env_id", "unknown"),
                "seed": payload.get("seed", "unknown"),
                "suite": benchmark.get("suite"),
                "preset_name": benchmark.get("preset_name"),
                "protocol_name": benchmark.get("protocol_name"),
                "latest_eval_return_mean": latest_metrics.get("eval_return_mean"),
                "latest_eval_human_normalized_score": latest_metrics.get("eval_human_normalized_score"),
                "best_eval_return_mean": latest_metrics.get(
                    "best_eval_return_mean", best_checkpoint.get("metric_value")
                ),
                "best_eval_human_normalized_score": latest_metrics.get(
                    "best_eval_human_normalized_score",
                    best_checkpoint.get("eval_human_normalized_score"),
                ),
                "best_minus_latest_eval_return_mean": difference(
                    latest_metrics.get("best_eval_return_mean", best_checkpoint.get("metric_value")),
                    latest_metrics.get("eval_return_mean"),
                ),
                "best_minus_latest_eval_human_normalized_score": difference(
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


def filter_run_reports(
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


def resolve_sort_value(record: Mapping[str, object], sort_by: str) -> object | None:
    for candidate in (sort_by, f"{sort_by}_mean", f"{sort_by}_max"):
        if candidate in record and record[candidate] is not None:
            return record[candidate]
    return None


def sortable_value(value: object) -> tuple[int, float | str]:
    if isinstance(value, int | float):
        return (0, float(value))

    value_text = str(value)
    try:
        return (0, float(value_text))
    except ValueError:
        return (1, value_text.lower())


def sort_records(
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
        resolved = resolve_sort_value(record, sort_by)
        if resolved is None:
            without_values.append((index, record))
            continue
        with_values.append((index, record, sortable_value(resolved)))

    ordered = sorted(with_values, key=lambda item: (item[2], item[0]), reverse=descending)
    return [record for _, record, _ in ordered] + [record for _, record in without_values]


def apply_top_k(records: list[dict[str, Any]], *, top_k: int | None = None) -> list[dict[str, Any]]:
    if top_k is None:
        return list(records)
    return list(records[:top_k])
