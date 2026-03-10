from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.dqn_trainer import train_dqn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small Prioritized DQN training job on CartPole-v1.")
    parser.add_argument("--total-timesteps", type=int, default=256)
    parser.add_argument("--output-dir", default="runs/reference-prioritized-dqn")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--eval-episodes", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        algo="prioritized_dqn",
        env_id="CartPole-v1",
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        output_dir=Path(args.output_dir),
        eval_episodes=args.eval_episodes,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "target_update_interval": 16,
            "hidden_sizes": (32, 32),
            "prioritized_alpha": 0.6,
            "prioritized_beta_start": 0.4,
        },
    )

    result = train_dqn(config)
    print(f"run_dir={result.run_dir}")
    print(f"checkpoint_path={result.checkpoint_path}")
    print(f"metrics={result.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
