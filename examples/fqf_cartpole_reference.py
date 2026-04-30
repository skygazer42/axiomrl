from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.dqn_trainer import train_dqn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small FQF training job on CartPole-v1.")
    parser.add_argument("--total-timesteps", type=int, default=256)
    parser.add_argument("--output-dir", default="runs/reference-fqf")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--num-quantiles", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        algo="fqf",
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
            "learning_rate": 0.001,
            "fraction_learning_rate": 0.0005,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "exploration_fraction": 0.3,
            "num_quantiles": args.num_quantiles,
            "embedding_dim": args.embedding_dim,
            "kappa": args.kappa,
            "entropy_coef": args.entropy_coef,
        },
    )

    result = train_dqn(config)
    print(f"run_dir={result.run_dir}")
    print(f"checkpoint_path={result.checkpoint_path}")
    print(f"metrics={result.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
