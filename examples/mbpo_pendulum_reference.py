from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.mbpo_trainer import train_mbpo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small MBPO training job on Pendulum-v1.")
    parser.add_argument("--total-timesteps", type=int, default=256)
    parser.add_argument("--output-dir", default="runs/reference-mbpo")
    parser.add_argument("--seed", type=int, default=223)
    parser.add_argument("--eval-episodes", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        algo="mbpo",
        env_id="Pendulum-v1",
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        output_dir=Path(args.output_dir),
        eval_episodes=args.eval_episodes,
        algo_kwargs={
            "buffer_capacity": 2048,
            "synthetic_buffer_capacity": 2048,
            "batch_size": 32,
            "model_batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "model_train_frequency": 1,
            "model_updates": 2,
            "hidden_sizes": (64, 64),
            "model_hidden_sizes": (64, 64),
            "num_ensembles": 3,
            "rollout_batch_size": 32,
            "rollout_horizon": 1,
            "rollout_refresh_interval": 16,
            "synthetic_batch_ratio": 0.5,
            "policy_learning_rate": 3e-4,
            "model_learning_rate": 1e-3,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
        },
    )

    result = train_mbpo(config)
    print(f"run_dir={result.run_dir}")
    print(f"checkpoint_path={result.checkpoint_path}")
    print(f"metrics={result.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
