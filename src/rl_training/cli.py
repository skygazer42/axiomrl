from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from rl_training.cli_config import apply_overrides, load_config, serialize_train_config
from rl_training.cli_doctor import print_doctor
from rl_training.cli_zoo import build_zoo_forward_argv
from rl_training.version import __version__


def _print_result(result: Any) -> None:
    print(f"run_dir={result.run_dir}")
    print(f"checkpoint_path={result.checkpoint_path}")
    if result.benchmark_summary_path is not None:
        print(f"benchmark_summary_path={result.benchmark_summary_path}")
    print(f"metrics={result.metrics}")


def _emit_output(content: str, *, output_path: str | Path | None = None) -> None:
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    print(content, end="")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="axiomrl")
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")
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

    config_parser = subparsers.add_parser("config")
    config_parser.add_argument("--config", required=True)
    config_parser.add_argument("--output-dir")
    config_parser.add_argument("--execution-backend")
    config_parser.add_argument("--total-timesteps", type=int)
    config_parser.add_argument("--num-envs", type=int)
    config_parser.add_argument("--eval-episodes", type=int)
    config_parser.add_argument("--seeds")
    config_parser.add_argument("--format", choices=("json", "yaml"), default="json")
    config_parser.add_argument("--output")

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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(_normalize_seed_argument_tokens(args_argv))

    if args.command == "doctor":
        print_doctor()
        return 0

    if args.command == "config":
        try:
            config = apply_overrides(load_config(args.config), args)
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        payload = serialize_train_config(config)
        if args.format == "yaml":
            rendered = yaml.safe_dump(payload, sort_keys=False)
        else:
            rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        _emit_output(rendered, output_path=args.output)
        return 0

    if args.command == "train":
        try:
            config = apply_overrides(load_config(args.config), args)
            from rl_training.experiment.sweeps import resolve_benchmark_seeds

            resolve_benchmark_seeds(config)
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))

        from rl_training.experiment.default_manager import DefaultExperimentManager

        manager = DefaultExperimentManager()
        try:
            result = manager.setup(config).train()
        except FileExistsError as exc:
            parser.error(str(exc))
        _print_result(result)
        return 0

    if args.command == "eval":
        from rl_training.runtime.workflows import evaluate_checkpoint

        metrics = evaluate_checkpoint(
            args.checkpoint,
            num_episodes=args.num_episodes,
        )
        print(metrics)
        return 0

    if args.command == "resume":
        from rl_training.runtime.workflows import resume_training

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
        from rl_training.zoo_cli import main as zoo_main

        return zoo_main(build_zoo_forward_argv(args, format_override="report"))

    if args.command == "leaderboard":
        from rl_training.zoo_cli import main as zoo_main

        return zoo_main(build_zoo_forward_argv(args, format_override="leaderboard"))

    if args.command == "zoo":
        from rl_training.zoo_cli import main as zoo_main

        return zoo_main(build_zoo_forward_argv(args))

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
