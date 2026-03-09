from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from rl_training.experiment.config import TrainConfig
from rl_training.experiment.registry import get_algorithm_spec
from rl_training.runtime.workflows import evaluate_checkpoint, resume_training


def load_config(path: str | Path) -> TrainConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rl_training")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = _apply_overrides(load_config(args.config), args)
        spec = get_algorithm_spec(config.algo)
        spec.train_fn(config)
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
        print(result.run_dir)
        print(result.checkpoint_path)
        print(result.metrics)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
