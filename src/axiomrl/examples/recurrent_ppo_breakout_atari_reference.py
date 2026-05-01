import argparse
from pathlib import Path

from axiomrl.contrib import RecurrentPPO
from axiomrl.experiment.config import TrainConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a compact recurrent PPO training job on Atari Breakout.")
    parser.add_argument("--total-timesteps", type=int, default=1024)
    parser.add_argument("--output-dir", default="runs/reference-recurrent-ppo-breakout")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        algo="recurrent_ppo",
        env_id="ALE/Breakout-v5",
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        output_dir=Path(args.output_dir),
        num_envs=args.num_envs,
        eval_episodes=args.eval_episodes,
        tags=("atari",),
        algo_kwargs={
            "num_steps": 64,
            "update_epochs": 1,
            "minibatch_size": 64,
            "sequence_length": 8,
            "sequences_per_batch": 8,
            "learning_rate": 2.5e-4,
            "clip_coef": 0.1,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "features_dim": 256,
            "recurrent_hidden_size": 256,
            "recurrent_num_layers": 1,
            "head_hidden_sizes": (128,),
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

    algo = RecurrentPPO(config)
    result = algo.learn()
    print(f"run_dir={result.run_dir}")
    print(f"checkpoint_path={result.checkpoint_path}")
    print(f"metrics={result.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
