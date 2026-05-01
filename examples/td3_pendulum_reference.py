import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.td3_trainer import train_td3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small TD3 training job on Pendulum-v1.")
    parser.add_argument("--total-timesteps", type=int, default=256)
    parser.add_argument("--output-dir", default="runs/reference-td3")
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument("--eval-episodes", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        algo="td3",
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
            "hidden_sizes": (32, 32),
            "tau": 0.005,
            "exploration_noise": 0.1,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_delay": 2,
        },
    )

    result = train_td3(config)
    print(f"run_dir={result.run_dir}")
    print(f"checkpoint_path={result.checkpoint_path}")
    print(f"metrics={result.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
