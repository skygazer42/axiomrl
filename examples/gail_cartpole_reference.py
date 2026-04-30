from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.gail_trainer import train_gail


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small GAIL training job on CartPole-v1.")
    parser.add_argument("--total-timesteps", type=int, default=256)
    parser.add_argument("--output-dir", default="runs/reference-gail")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        algo="gail",
        env_id="CartPole-v1",
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        output_dir=Path(args.output_dir),
        num_envs=args.num_envs,
        eval_episodes=args.eval_episodes,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 64,
            "hidden_sizes": (64, 64),
            "learning_rate": 3e-4,
            "clip_coef": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "discriminator_learning_rate": 3e-4,
            "discriminator_updates": 4,
            "discriminator_batch_size": 64,
            "expert_dataset_kind": "random",
            "expert_dataset_size": 512,
            "expert_dataset_seed": 17,
        },
    )

    result = train_gail(config)
    print(f"run_dir={result.run_dir}")
    print(f"checkpoint_path={result.checkpoint_path}")
    print(f"metrics={result.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
