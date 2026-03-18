from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.dreamer_trainer import train_dreamer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small Dreamer-style job on CartPole-v1 pixels.")
    parser.add_argument("--total-timesteps", type=int, default=256)
    parser.add_argument("--output-dir", default="runs/reference-dreamer")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--eval-episodes", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        algo="dreamer",
        env_id="CartPole-v1",
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        output_dir=Path(args.output_dir),
        eval_episodes=args.eval_episodes,
        algo_kwargs={
            "buffer_capacity": 2048,
            "batch_size": 16,
            "learning_starts": 32,
            "train_frequency": 1,
            "world_model_updates": 1,
            "actor_critic_updates": 1,
            "imagination_batch_size": 8,
            "imagination_horizon": 3,
            "features_dim": 64,
            "action_embed_dim": 16,
            "world_model_learning_rate": 1e-3,
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "gamma": 0.99,
            "entropy_coef": 1e-3,
        },
        env_kwargs={
            "render_mode": "rgb_array",
            "wrappers": {
                "pixels": {
                    "resize_shape": [84, 84],
                    "frame_stack": 3,
                    "channel_first": True,
                }
            },
        },
    )

    result = train_dreamer(config)
    print(f"run_dir={result.run_dir}")
    print(f"checkpoint_path={result.checkpoint_path}")
    print(f"metrics={result.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

