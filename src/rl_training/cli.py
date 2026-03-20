from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
import platform
import sys
from typing import Any, cast

import yaml

from rl_training.experiment.config import TrainConfig
from rl_training.experiment.default_manager import DefaultExperimentManager
from rl_training.experiment.sweeps import resolve_benchmark_seeds
from rl_training.resources import find_packaged_asset
from rl_training.runtime.trainer import TrainResult
from rl_training.runtime.workflows import evaluate_checkpoint, resume_training
from rl_training.zoo_cli import apply_manifest_defaults_to_config_payload, main as zoo_main


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"unable to read YAML config at {path}: {exc}") from exc

    try:
        payload = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:  # pragma: no cover
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"expected YAML mapping in {path}, got {type(payload)!r}")
    return payload


def _resolve_linked_config_path(config_path: Path, linked_path: str) -> Path:
    candidate = Path(linked_path)
    if candidate.is_absolute():
        return candidate

    for base_dir in config_path.parents:
        parent_candidate = (base_dir / candidate).resolve()
        if parent_candidate.exists():
            return parent_candidate

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (config_path.parent / candidate).resolve()


def _load_config_payload(path: Path, *, visited: set[Path] | None = None) -> dict[str, Any]:
    resolved_path = path.resolve()
    seen = set() if visited is None else set(visited)
    if resolved_path in seen:
        raise ValueError(f"detected config include cycle at {resolved_path}")

    seen.add(resolved_path)
    payload = _load_yaml_mapping(resolved_path)
    if "algo" in payload:
        return payload

    linked_config = payload.get("config")
    if isinstance(linked_config, str):
        linked_path = _resolve_linked_config_path(resolved_path, linked_config)
        linked_payload = _load_config_payload(linked_path, visited=seen)
        return apply_manifest_defaults_to_config_payload(linked_payload, preset_path=resolved_path, preset_payload=payload)

    raise ValueError(f"config file {resolved_path} must define 'algo' or reference another config via 'config'")


def _resolve_input_config_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    packaged = find_packaged_asset(candidate)
    if packaged is not None:
        return packaged
    return candidate


def _require_payload_field(payload: Mapping[str, Any], field: str, *, config_path: Path) -> object:
    if field not in payload:
        raise ValueError(f"config file {config_path} missing required key '{field}'")
    return payload[field]


def _coerce_required_str(value: object, field: str, *, config_path: Path) -> str:
    if value is None:
        raise ValueError(f"config file {config_path} missing required key '{field}'")
    if not isinstance(value, str):
        raise TypeError(f"config file {config_path} field '{field}' must be a string, got {type(value)!r}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"config file {config_path} field '{field}' must not be empty")
    return normalized


def _coerce_optional_str(payload: Mapping[str, Any], field: str, default: str, *, config_path: Path) -> str:
    value = payload.get(field)
    if value is None:
        return default
    if not isinstance(value, str):
        raise TypeError(f"config file {config_path} field '{field}' must be a string, got {type(value)!r}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"config file {config_path} field '{field}' must not be empty")
    return normalized


def _coerce_int(value: object, field: str, *, config_path: Path) -> int:
    if isinstance(value, bool):
        raise TypeError(f"config file {config_path} field '{field}' must be an integer, got {type(value)!r}")
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"config file {config_path} field '{field}' must be an integer, got {type(value)!r}") from exc


def _coerce_required_int(payload: Mapping[str, Any], field: str, *, config_path: Path, min_value: int | None = None) -> int:
    value = _require_payload_field(payload, field, config_path=config_path)
    int_value = _coerce_int(value, field, config_path=config_path)
    if min_value is not None and int_value < min_value:
        raise ValueError(f"config file {config_path} field '{field}' must be >= {min_value}, got {int_value}")
    return int_value


def _coerce_optional_int(
    payload: Mapping[str, Any],
    field: str,
    default: int,
    *,
    config_path: Path,
    min_value: int | None = None,
) -> int:
    value = payload.get(field)
    if value is None:
        return default
    int_value = _coerce_int(value, field, config_path=config_path)
    if min_value is not None and int_value < min_value:
        raise ValueError(f"config file {config_path} field '{field}' must be >= {min_value}, got {int_value}")
    return int_value


def _coerce_required_path(value: object, field: str, *, config_path: Path) -> Path:
    if value is None:
        raise ValueError(f"config file {config_path} missing required key '{field}'")
    if isinstance(value, Path):
        return value
    if not isinstance(value, str):
        raise TypeError(f"config file {config_path} field '{field}' must be a string path, got {type(value)!r}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"config file {config_path} field '{field}' must not be empty")
    return Path(normalized)


def _coerce_optional_mapping(
    payload: Mapping[str, Any],
    field: str,
    *,
    config_path: Path,
) -> dict[str, Any]:
    value = payload.get(field)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"config file {config_path} field '{field}' must be a mapping, got {type(value)!r}")
    return dict(value)


def _coerce_optional_tags(payload: Mapping[str, Any], *, config_path: Path) -> tuple[str, ...]:
    value = payload.get("tags")
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError(f"config file {config_path} field 'tags' must be a sequence of strings, got {type(value)!r}")
    if not isinstance(value, Sequence):
        raise TypeError(f"config file {config_path} field 'tags' must be a sequence of strings, got {type(value)!r}")
    normalized: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            raise TypeError(
                f"config file {config_path} field 'tags' entries must be strings, got {type(entry)!r}"
            )
        tag = entry.strip()
        if tag:
            normalized.append(tag)
    return tuple(normalized)


def load_config(path: str | Path) -> TrainConfig:
    config_path = _resolve_input_config_path(path)
    if not config_path.exists():
        raise ValueError(f"config file {config_path} does not exist")
    payload = _load_config_payload(config_path)

    return TrainConfig(
        algo=_coerce_required_str(_require_payload_field(payload, "algo", config_path=config_path), "algo", config_path=config_path),
        env_id=_coerce_required_str(
            _require_payload_field(payload, "env_id", config_path=config_path),
            "env_id",
            config_path=config_path,
        ),
        seed=_coerce_required_int(payload, "seed", config_path=config_path, min_value=0),
        total_timesteps=_coerce_required_int(payload, "total_timesteps", config_path=config_path, min_value=1),
        output_dir=_coerce_required_path(
            _require_payload_field(payload, "output_dir", config_path=config_path),
            "output_dir",
            config_path=config_path,
        ),
        execution_backend=_coerce_optional_str(payload, "execution_backend", "local_sync", config_path=config_path),
        device=_coerce_optional_str(payload, "device", "auto", config_path=config_path),
        num_envs=_coerce_optional_int(payload, "num_envs", 1, config_path=config_path, min_value=1),
        eval_episodes=_coerce_optional_int(payload, "eval_episodes", 5, config_path=config_path, min_value=1),
        log_interval=_coerce_optional_int(payload, "log_interval", 1, config_path=config_path, min_value=1),
        checkpoint_interval=_coerce_optional_int(payload, "checkpoint_interval", 1, config_path=config_path, min_value=1),
        tags=_coerce_optional_tags(payload, config_path=config_path),
        benchmark=_coerce_optional_mapping(payload, "benchmark", config_path=config_path),
        algo_kwargs=_coerce_optional_mapping(payload, "algo_kwargs", config_path=config_path),
        env_kwargs=_coerce_optional_mapping(payload, "env_kwargs", config_path=config_path),
    )


def _apply_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    overrides: dict[str, Any] = {}

    if getattr(args, "output_dir", None) is not None:
        overrides["output_dir"] = Path(args.output_dir)
    if getattr(args, "execution_backend", None) is not None:
        overrides["execution_backend"] = str(args.execution_backend)
    if getattr(args, "total_timesteps", None) is not None:
        overrides["total_timesteps"] = int(args.total_timesteps)
    if getattr(args, "num_envs", None) is not None:
        overrides["num_envs"] = int(args.num_envs)
    if getattr(args, "eval_episodes", None) is not None:
        overrides["eval_episodes"] = int(args.eval_episodes)
    if getattr(args, "seeds", None) is not None:
        raw_value = str(args.seeds)
        tokens = [token.strip() for token in raw_value.split(",")]
        if any(token == "" for token in tokens):
            raise ValueError("--seeds expects a comma-separated list of integers, for example: 1,2,3")
        try:
            seed_values = [int(token) for token in tokens]
        except ValueError as exc:
            raise ValueError("--seeds expects a comma-separated list of integers, for example: 1,2,3") from exc
        if any(seed < 0 for seed in seed_values):
            raise ValueError("--seeds expects a comma-separated list of non-negative integers, for example: 1,2,3")
        benchmark = dict(config.benchmark)
        benchmark["seeds"] = seed_values
        overrides["benchmark"] = benchmark

    return cast(TrainConfig, replace(config, **overrides))


def _print_result(result: TrainResult) -> None:
    print(f"run_dir={result.run_dir}")
    print(f"checkpoint_path={result.checkpoint_path}")
    if result.benchmark_summary_path is not None:
        print(f"benchmark_summary_path={result.benchmark_summary_path}")
    print(f"metrics={result.metrics}")


def _print_doctor() -> None:
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover
        metadata = None  # type: ignore[assignment]

    def resolve_version(distribution: str) -> str:
        if metadata is None:
            return "unknown"
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return "missing"

    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None  # type: ignore[assignment]

    print(f"python_version={platform.python_version()}")
    print(f"platform={platform.platform()}")
    print(f"torch_version={resolve_version('torch')}")
    print(f"gymnasium_version={resolve_version('gymnasium')}")
    print(f"numpy_version={resolve_version('numpy')}")
    if torch is None:
        print("cuda_available=unknown")
    else:
        print(f"cuda_available={torch.cuda.is_available()}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="axiomrl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--output-dir")
    train_parser.add_argument("--execution-backend")
    train_parser.add_argument("--total-timesteps", type=int)
    train_parser.add_argument("--num-envs", type=int)
    train_parser.add_argument("--eval-episodes", type=int)
    train_parser.add_argument("--seeds")

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--num-episodes", type=int)

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--checkpoint", required=True)
    resume_parser.add_argument("--total-timesteps", type=int)
    resume_parser.add_argument("--output-dir")
    resume_parser.add_argument("--execution-backend")
    resume_parser.add_argument("--eval-episodes", type=int)

    zoo_parser = subparsers.add_parser("zoo")
    zoo_parser.add_argument("--manifest", default="zoo/atari/benchmark.yaml")
    zoo_parser.add_argument(
        "--format",
        choices=("table", "commands", "report", "leaderboard"),
        default="table",
    )
    zoo_parser.add_argument("--runs-dir", default="runs")
    zoo_parser.add_argument(
        "--report-output",
        choices=("text", "json", "csv"),
        default="text",
    )
    zoo_parser.add_argument("--algo")
    zoo_parser.add_argument("--env-id")
    zoo_parser.add_argument(
        "--group-by",
        choices=("algo-env", "preset"),
        default="algo-env",
    )
    zoo_parser.add_argument("--min-seeds", type=int)
    zoo_parser.add_argument("--top-k", type=int)
    zoo_parser.add_argument("--baseline-preset")
    zoo_parser.add_argument(
        "--leaderboard-metric",
        choices=(
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
        ),
    )
    zoo_parser.add_argument(
        "--compare-to",
        choices=("best", "latest"),
    )
    zoo_parser.add_argument(
        "--score-view",
        choices=("return", "normalized"),
    )
    zoo_parser.add_argument("--sort-by")
    zoo_parser.add_argument("--descending", action="store_true")
    zoo_parser.add_argument("--fail-on-manifest-drift", action="store_true")
    zoo_parser.add_argument("--fail-on-manifest-drift-severity", choices=("warning", "error"))
    zoo_parser.add_argument(
        "--fail-on-manifest-drift-type",
        action="append",
        choices=("unknown-preset", "protocol-mismatch"),
    )
    zoo_parser.add_argument("--output")

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--manifest", default="zoo/atari/benchmark.yaml")
    report_parser.add_argument("--runs-dir", default="runs")
    report_parser.add_argument(
        "--report-output",
        choices=("text", "json", "csv"),
        default="text",
    )
    report_parser.add_argument("--algo")
    report_parser.add_argument("--env-id")
    report_parser.add_argument(
        "--group-by",
        choices=("algo-env", "preset"),
        default="algo-env",
    )
    report_parser.add_argument("--min-seeds", type=int)
    report_parser.add_argument("--top-k", type=int)
    report_parser.add_argument("--baseline-preset")
    report_parser.add_argument("--sort-by")
    report_parser.add_argument("--descending", action="store_true")
    report_parser.add_argument("--fail-on-manifest-drift", action="store_true")
    report_parser.add_argument("--fail-on-manifest-drift-severity", choices=("warning", "error"))
    report_parser.add_argument(
        "--fail-on-manifest-drift-type",
        action="append",
        choices=("unknown-preset", "protocol-mismatch"),
    )
    report_parser.add_argument("--output")

    leaderboard_parser = subparsers.add_parser("leaderboard")
    leaderboard_parser.add_argument("--manifest", default="zoo/atari/benchmark.yaml")
    leaderboard_parser.add_argument("--runs-dir", default="runs")
    leaderboard_parser.add_argument(
        "--report-output",
        choices=("text", "json", "csv"),
        default="text",
    )
    leaderboard_parser.add_argument("--algo")
    leaderboard_parser.add_argument("--env-id")
    leaderboard_parser.add_argument(
        "--group-by",
        choices=("algo-env", "preset"),
        default="algo-env",
    )
    leaderboard_parser.add_argument("--min-seeds", type=int)
    leaderboard_parser.add_argument("--top-k", type=int)
    leaderboard_parser.add_argument("--baseline-preset")
    leaderboard_parser.add_argument(
        "--leaderboard-metric",
        choices=(
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
        ),
    )
    leaderboard_parser.add_argument(
        "--compare-to",
        choices=("best", "latest"),
    )
    leaderboard_parser.add_argument(
        "--score-view",
        choices=("return", "normalized"),
    )
    leaderboard_parser.add_argument("--sort-by")
    leaderboard_parser.add_argument("--descending", action="store_true")
    leaderboard_parser.add_argument("--fail-on-manifest-drift", action="store_true")
    leaderboard_parser.add_argument("--fail-on-manifest-drift-severity", choices=("warning", "error"))
    leaderboard_parser.add_argument(
        "--fail-on-manifest-drift-type",
        action="append",
        choices=("unknown-preset", "protocol-mismatch"),
    )
    leaderboard_parser.add_argument("--output")

    subparsers.add_parser("doctor")
    return parser


def _normalize_seed_argument_tokens(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--seeds" and index + 1 < len(argv):
            value_token = argv[index + 1]
            if value_token.startswith("-") and not value_token.startswith("--"):
                normalized.append(f"--seeds={value_token}")
                index += 2
                continue
        normalized.append(token)
        index += 1
    return normalized


def _build_zoo_forward_argv(args: argparse.Namespace, *, format_override: str | None = None) -> list[str]:
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(_normalize_seed_argument_tokens(args_argv))

    if args.command == "doctor":
        _print_doctor()
        return 0

    if args.command == "train":
        try:
            config = _apply_overrides(load_config(args.config), args)
            resolve_benchmark_seeds(config)
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        manager = DefaultExperimentManager()
        try:
            result = manager.setup(config).train()
        except FileExistsError as exc:
            parser.error(str(exc))
        _print_result(result)
        return 0

    if args.command == "eval":
        metrics = evaluate_checkpoint(
            args.checkpoint,
            num_episodes=args.num_episodes,
        )
        print(metrics)
        return 0

    if args.command == "resume":
        result = resume_training(
            args.checkpoint,
            total_timesteps=args.total_timesteps,
            output_dir=args.output_dir,
            execution_backend=args.execution_backend,
            eval_episodes=args.eval_episodes,
        )
        _print_result(result)
        return 0

    if args.command == "report":
        return zoo_main(_build_zoo_forward_argv(args, format_override="report"))

    if args.command == "leaderboard":
        return zoo_main(_build_zoo_forward_argv(args, format_override="leaderboard"))

    if args.command == "zoo":
        return zoo_main(_build_zoo_forward_argv(args))

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
