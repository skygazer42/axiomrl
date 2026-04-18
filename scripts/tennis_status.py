from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _resolve_run_dir(path: Path) -> Path:
    if (path / "metadata.json").exists():
        return path

    candidates = [entry for entry in path.iterdir() if entry.is_dir()]
    if not candidates:
        raise ValueError(f"no run directories found under {path}")
    return max(candidates, key=lambda entry: entry.stat().st_mtime)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}, got {type(payload)!r}")
    return payload


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    metrics: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            metrics.append(payload)
    return metrics


def _eval_value(record: dict[str, Any]) -> float:
    metrics = record.get("metrics", {})
    if not isinstance(metrics, dict):
        return float("nan")
    return float(metrics.get("eval_return_mean", 0.0))


def _avg(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _format_float(value: float) -> str:
    if value != value:
        return "nan"
    if value.is_integer():
        return str(int(value))
    return f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="run dir or parent output dir")
    args = parser.parse_args()

    base_path = Path(args.path).resolve()
    run_dir = _resolve_run_dir(base_path)
    metadata_path = run_dir / "metadata.json"
    metrics_path = run_dir / "metrics.jsonl"
    checkpoints_dir = run_dir / "checkpoints"

    metadata = _load_json(metadata_path)
    metrics = _load_metrics(metrics_path)
    eval_values = [_eval_value(record) for record in metrics]

    print(f"run_dir={run_dir}")
    print(f"created_at_utc={metadata.get('created_at_utc')}")
    print(f"latest_checkpoint_path={metadata.get('latest_checkpoint_path')}")

    if checkpoints_dir.exists():
        checkpoints = sorted(entry.name for entry in checkpoints_dir.iterdir() if entry.is_file())
        print(f"checkpoints={len(checkpoints)}")
        if checkpoints:
            print(f"latest_checkpoint_file={checkpoints[-1]}")
    else:
        print("checkpoints=0")

    print(f"metrics_file_exists={metrics_path.exists()}")
    if metrics_path.exists():
        metrics_mtime = datetime.fromtimestamp(metrics_path.stat().st_mtime).isoformat(timespec="seconds")
        print(f"metrics_mtime={metrics_mtime}")
    print(f"eval_count={len(metrics)}")

    if metrics:
        best_record = max(metrics, key=_eval_value)
        latest_record = metrics[-1]
        recent5 = eval_values[-5:]
        recent10 = eval_values[-10:]
        print(
            "best_eval="
            f"{_format_float(_eval_value(best_record))}@{int(best_record.get('step', 0))}"
        )
        print(
            "latest_eval="
            f"{_format_float(_eval_value(latest_record))}@{int(latest_record.get('step', 0))}"
        )
        print(f"recent5_avg={_format_float(_avg(recent5))}")
        print(f"recent10_avg={_format_float(_avg(recent10))}")
        print(f"positive_evals={sum(1 for value in eval_values if value > 0.0)}")
        print(f"zero_evals={sum(1 for value in eval_values if value == 0.0)}")
        print(f"negative_evals={sum(1 for value in eval_values if value < 0.0)}")
    else:
        print("best_eval=pending")
        print("latest_eval=pending")
        print("recent5_avg=pending")
        print("recent10_avg=pending")
        print("positive_evals=0")
        print("zero_evals=0")
        print("negative_evals=0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
