from __future__ import annotations

import argparse
from pathlib import Path

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.dqn_trainer import train_dqn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a compact Atari DQN training job on Breakout.")
    parser.add_argument("--total-timesteps", type=int, default=1024)
    parser.add_argument("--output-dir", default="runs/reference-dqn-breakout")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--eval-episodes", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        algo="dqn",
        env_id="ALE/Breakout-v5",
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        output_dir=Path(args.output_dir),
        eval_episodes=args.eval_episodes,
        tags=("atari",),
        algo_kwargs={
            "buffer_capacity": 2048,
            "batch_size": 32,
            "learning_starts": 128,
            "train_frequency": 4,
            "target_update_interval": 64,
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "exploration_fraction": 0.2,
            "head_hidden_sizes": (512,),
            "features_dim": 512,
        },
        env_kwargs={
            "frameskip": 1,
            "repeat_action_probability": 0.0,
            "full_action_space": False,
            "wrappers": {
                "atari": {
                    "screen_size": 84,
                    "frame_skip": 4,
                    "noop_max": 30,
                    "grayscale_obs": True,
                    "frame_stack": 4,
                    "clip_reward": True,
                    "channel_first": True,
                }
            },
        },
    )

    result = train_dqn(config)
    print(f"run_dir={result.run_dir}")
    print(f"checkpoint_path={result.checkpoint_path}")
    print(f"metrics={result.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
