from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from rl_training.experiment.config import TrainConfig
from rl_training.experiment.registry import get_algorithm_spec
from rl_training.resources import find_packaged_asset
from rl_training.runtime.trainer import TrainResult
from rl_training.runtime.workflows import evaluate_checkpoint, resume_training
from rl_training.zoo_cli import main as zoo_main


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
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
        return _load_config_payload(linked_path, visited=seen)

    raise KeyError(f"config file {resolved_path} must define 'algo' or reference another config via 'config'")


def _resolve_input_config_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    packaged = find_packaged_asset(candidate)
    if packaged is not None:
        return packaged
    return candidate


def load_config(path: str | Path) -> TrainConfig:
    config_path = _resolve_input_config_path(path)
    payload = _load_config_payload(config_path)

    return TrainConfig(
        algo=payload["algo"],
        env_id=payload["env_id"],
        seed=int(payload["seed"]),
        total_timesteps=int(payload["total_timesteps"]),
        output_dir=Path(payload["output_dir"]),
        device=payload.get("device", "auto"),
        num_envs=int(payload.get("num_envs", 1)),
        eval_episodes=int(payload.get("eval_episodes", 5)),
        log_interval=int(payload.get("log_interval", 1)),
        checkpoint_interval=int(payload.get("checkpoint_interval", 1)),
        tags=tuple(payload.get("tags", ())),
        algo_kwargs=dict(payload.get("algo_kwargs", {})),
        env_kwargs=dict(payload.get("env_kwargs", {})),
    )


def _apply_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    overrides: dict[str, Any] = {}

    if getattr(args, "output_dir", None) is not None:
        overrides["output_dir"] = Path(args.output_dir)
    if getattr(args, "total_timesteps", None) is not None:
        overrides["total_timesteps"] = int(args.total_timesteps)
    if getattr(args, "num_envs", None) is not None:
        overrides["num_envs"] = int(args.num_envs)
    if getattr(args, "eval_episodes", None) is not None:
        overrides["eval_episodes"] = int(args.eval_episodes)

    return replace(config, **overrides)


def _print_result(result: TrainResult) -> None:
    print(f"run_dir={result.run_dir}")
    print(f"checkpoint_path={result.checkpoint_path}")
    print(f"metrics={result.metrics}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rl-training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--output-dir")
    train_parser.add_argument("--total-timesteps", type=int)
    train_parser.add_argument("--num-envs", type=int)
    train_parser.add_argument("--eval-episodes", type=int)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--num-episodes", type=int)

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--checkpoint", required=True)
    resume_parser.add_argument("--total-timesteps", type=int)
    resume_parser.add_argument("--output-dir")
    resume_parser.add_argument("--eval-episodes", type=int)

    zoo_parser = subparsers.add_parser("zoo")
    zoo_parser.add_argument("--manifest", default="zoo/atari/benchmark.yaml")
    zoo_parser.add_argument(
        "--format",
        choices=("table", "commands"),
        default="table",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = _apply_overrides(load_config(args.config), args)
        spec = get_algorithm_spec(config.algo)
        result = spec.train_fn(config)
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
            eval_episodes=args.eval_episodes,
        )
        _print_result(result)
        return 0

    if args.command == "zoo":
        zoo_argv = ["--manifest", args.manifest, "--format", args.format]
        return zoo_main(zoo_argv)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
