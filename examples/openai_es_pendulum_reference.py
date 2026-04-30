from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.openai_es_trainer import train_openai_es


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small OpenAI ES training job on Pendulum-v1.")
    parser.add_argument("--total-timesteps", type=int, default=200)
    parser.add_argument("--output-dir", default="runs/reference-openai-es")
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--eval-episodes", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        algo="openai_es",
        env_id="Pendulum-v1",
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        output_dir=Path(args.output_dir),
        eval_episodes=args.eval_episodes,
        algo_kwargs={
            "hidden_sizes": (32, 32),
            "step_size": 0.02,
            "noise_std": 0.03,
            "num_directions": 2,
            "weight_decay": 0.0,
        },
        env_kwargs={
            "max_episode_steps": 25,
        },
    )

    result = train_openai_es(config)
    print(f"run_dir={result.run_dir}")
    print(f"checkpoint_path={result.checkpoint_path}")
    print(f"metrics={result.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
