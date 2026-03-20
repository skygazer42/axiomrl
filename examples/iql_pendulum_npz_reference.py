from __future__ import annotations

import argparse
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_training import IQL, TrainConfig
from rl_training.data import export_random_transition_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect a real Pendulum-v1 rollout into NPZ, train IQL, then load weights for inference."
    )
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--dataset-seed", type=int, default=21)
    parser.add_argument("--total-timesteps", type=int, default=256)
    parser.add_argument("--output-dir", default="runs/reference-iql-npz")
    parser.add_argument("--seed", type=int, default=89)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=3.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    dataset_path = export_random_transition_dataset(
        "Pendulum-v1",
        output_dir / "datasets" / "pendulum_rollout.npz",
        num_steps=args.dataset_size,
        seed=args.dataset_seed,
    )

    algo = IQL(
        TrainConfig(
            algo="iql",
            env_id="Pendulum-v1",
            seed=args.seed,
            total_timesteps=args.total_timesteps,
            output_dir=output_dir,
            eval_episodes=args.eval_episodes,
            algo_kwargs={
                "dataset_kind": "npz",
                "dataset_path": str(dataset_path),
                "batch_size": 32,
                "hidden_sizes": (32, 32),
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "tau": 0.005,
                "expectile": args.expectile,
                "beta": args.beta,
                "max_advantage_weight": 100.0,
            },
        )
    )
    result = algo.learn()
    checkpoint_path = algo.save(output_dir / "exports" / "iql_reference.pt")
    loaded = IQL.load(checkpoint_path)

    env = gym.make("Pendulum-v1")
    try:
        obs, _ = env.reset(seed=args.seed)
        action = loaded.predict(obs)
        _, reward, terminated, truncated, _ = env.step(action)
    finally:
        env.close()

    metrics = loaded.evaluate(num_episodes=args.eval_episodes)

    print(f"dataset_path={dataset_path}")
    print(f"run_dir={result.run_dir}")
    print(f"checkpoint_path={checkpoint_path}")
    print(f"metrics={result.metrics}")
    print(f"eval_metrics={metrics}")
    print(f"inference_action={np.asarray(action).tolist()}")
    print(f"inference_step_reward={float(reward)}")
    print(f"inference_done={bool(terminated or truncated)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
