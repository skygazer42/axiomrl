from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.cql_trainer import train_cql


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small CQL offline training job on Pendulum-v1.")
    parser.add_argument("--total-timesteps", type=int, default=256)
    parser.add_argument("--output-dir", default="runs/reference-cql")
    parser.add_argument("--seed", type=int, default=107)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--dataset-seed", type=int, default=41)
    parser.add_argument("--cql-alpha", type=float, default=5.0)
    parser.add_argument("--num-cql-samples", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        algo="cql",
        env_id="Pendulum-v1",
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        output_dir=Path(args.output_dir),
        eval_episodes=args.eval_episodes,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": args.dataset_size,
            "dataset_seed": args.dataset_seed,
            "batch_size": 32,
            "hidden_sizes": (32, 32),
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
            "cql_alpha": args.cql_alpha,
            "num_cql_samples": args.num_cql_samples,
        },
    )

    result = train_cql(config)
    print(f"run_dir={result.run_dir}")
    print(f"checkpoint_path={result.checkpoint_path}")
    print(f"metrics={result.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
