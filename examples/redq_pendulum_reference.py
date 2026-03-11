from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.redq_trainer import train_redq


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small REDQ training job on Pendulum-v1.")
    parser.add_argument("--total-timesteps", type=int, default=256)
    parser.add_argument("--output-dir", default="runs/reference-redq")
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--num-critics", type=int, default=5)
    parser.add_argument("--subset-size", type=int, default=2)
    parser.add_argument("--gradient-updates-per-step", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        algo="redq",
        env_id="Pendulum-v1",
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        output_dir=Path(args.output_dir),
        eval_episodes=args.eval_episodes,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "gradient_updates_per_step": args.gradient_updates_per_step,
            "hidden_sizes": (32, 32),
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
            "num_critics": args.num_critics,
            "subset_size": args.subset_size,
        },
    )

    result = train_redq(config)
    print(f"run_dir={result.run_dir}")
    print(f"checkpoint_path={result.checkpoint_path}")
    print(f"metrics={result.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
