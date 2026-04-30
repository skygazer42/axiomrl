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


def _print_study_result(result: Any) -> None:
    print(f"study_dir={result.study_dir}")
    print(f"best_trial_index={result.best_trial_index}")
    print(f"best_objective_value={result.best_objective_value}")
    print(f"best_run_dir={result.best_run_dir}")
    print(f"best_checkpoint_path={result.best_checkpoint_path}")


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
    resume_parser.add_argument("--config")
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

    tune_parser = subparsers.add_parser("tune")
    tune_group = tune_parser.add_mutually_exclusive_group(required=True)
    tune_group.add_argument("--config")
    tune_group.add_argument("--resume-study")
    tune_parser.add_argument("--output-dir")
    tune_parser.add_argument("--backend", choices=("native", "optuna"))

    tune_report_parser = subparsers.add_parser("tune-report")
    tune_report_parser.add_argument("--study-dir", required=True)
    tune_report_parser.add_argument(
        "--report-output",
        choices=("text", "json", "csv"),
        default="text",
    )
    tune_report_parser.add_argument(
        "--status",
        choices=("all", "completed", "failed"),
        default="all",
    )
    tune_report_parser.add_argument(
        "--sort-by",
        choices=("trial-index", "objective-value", "duration-seconds"),
        default="trial-index",
    )
    tune_report_parser.add_argument("--descending", action="store_true")
    tune_report_parser.add_argument("--top-k", type=int)
    tune_report_parser.add_argument("--objective-at-least", type=float)
    tune_report_parser.add_argument("--objective-at-most", type=float)
    tune_report_parser.add_argument("--duration-at-least", type=float)
    tune_report_parser.add_argument("--duration-at-most", type=float)
    tune_report_parser.add_argument("--frontier-only", action="store_true")
    tune_report_parser.add_argument("--param", dest="param_filters", action="append")
    tune_report_parser.add_argument("--error")
    tune_report_parser.add_argument("--error-contains")
    tune_report_parser.add_argument("--error-type")
    tune_report_parser.add_argument("--focus-param")
    tune_report_parser.add_argument(
        "--focus-sort-by",
        choices=(
            "best-objective-value",
            "mean-objective-value",
            "completion-rate",
            "incumbent-updates",
            "mean-duration-seconds",
            "value",
        ),
    )
    tune_report_parser.add_argument("--focus-top-k", type=int)
    tune_report_parser.add_argument("--export-configs-dir")
    tune_report_parser.add_argument("--output")

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


def _parse_param_filters(tokens: list[str] | None) -> dict[str, object]:
    if not tokens:
        return {}
    parsed: dict[str, object] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"expected --param in key=value form, got {token!r}")
        key, raw_value = token.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"expected --param in key=value form, got {token!r}")
        if key in parsed:
            raise ValueError(f"duplicate --param filter for {key!r}")
        parsed[key] = yaml.safe_load(raw_value)
    return parsed


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

    if args.command == "tune":
        from dataclasses import replace

        from rl_training.tuning.config import load_study_config
        from rl_training.tuning.study import resume_study, run_study

        if getattr(args, "resume_study", None) is not None:
            if getattr(args, "output_dir", None) is not None:
                parser.error("--output-dir is not supported with --resume-study")
            if getattr(args, "backend", None) is not None:
                parser.error("--backend is not supported with --resume-study")
            try:
                result = resume_study(args.resume_study)
            except (TypeError, ValueError, FileExistsError, ModuleNotFoundError) as exc:
                parser.error(str(exc))
            _print_study_result(result)
            return 0

        try:
            study_config = load_study_config(args.config)
            if getattr(args, "output_dir", None) is not None:
                study_config = replace(study_config, output_dir=Path(args.output_dir).resolve())
            if getattr(args, "backend", None) is not None:
                study_config = replace(
                    study_config,
                    study=replace(study_config.study, backend=str(args.backend)),
                )
            result = run_study(study_config)
        except (TypeError, ValueError, FileExistsError, ModuleNotFoundError) as exc:
            parser.error(str(exc))
        _print_study_result(result)
        return 0

    if args.command == "tune-report":
        from rl_training.tuning.study import (
            export_selected_study_configs,
            load_study_report,
            render_csv_study_report,
            render_json_study_report,
            render_text_study_report,
            select_study_report,
        )

        try:
            payload = load_study_report(args.study_dir)
            payload = select_study_report(
                payload,
                status=str(args.status),
                sort_by=str(args.sort_by),
                descending=bool(args.descending),
                top_k=args.top_k,
                frontier_only=bool(getattr(args, "frontier_only", False)),
                objective_at_least=getattr(args, "objective_at_least", None),
                objective_at_most=getattr(args, "objective_at_most", None),
                duration_at_least=getattr(args, "duration_at_least", None),
                duration_at_most=getattr(args, "duration_at_most", None),
                param_filters=_parse_param_filters(getattr(args, "param_filters", None)),
                error=getattr(args, "error", None),
                error_contains=getattr(args, "error_contains", None),
                error_type=getattr(args, "error_type", None),
                focus_param=getattr(args, "focus_param", None),
                focus_sort_by=getattr(args, "focus_sort_by", None),
                focus_top_k=getattr(args, "focus_top_k", None),
            )
            if args.export_configs_dir is not None:
                payload = dict(payload)
                payload["config_export_summary"] = export_selected_study_configs(payload, args.export_configs_dir)
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        if args.report_output == "json":
            rendered = render_json_study_report(payload)
        elif args.report_output == "csv":
            rendered = render_csv_study_report(payload)
        else:
            rendered = render_text_study_report(payload)
        _emit_output(rendered, output_path=args.output)
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

        try:
            result = resume_training(
                args.checkpoint,
                config_path=args.config,
                total_timesteps=args.total_timesteps,
                output_dir=args.output_dir,
                execution_backend=args.execution_backend,
                eval_episodes=args.eval_episodes,
            )
        except (TypeError, ValueError, FileExistsError) as exc:
            parser.error(str(exc))
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
