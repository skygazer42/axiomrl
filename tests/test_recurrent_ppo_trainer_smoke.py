from pathlib import Path

from rl_training.experiment.config import TrainConfig
from rl_training.runtime.recurrent_ppo_trainer import train_recurrent_ppo
from rl_training.runtime.workflows import evaluate_checkpoint


def test_train_recurrent_ppo_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="recurrent_ppo",
        env_id="CartPole-v1",
        seed=71,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "sequence_length": 8,
            "sequences_per_batch": 4,
            "encoder_hidden_sizes": (16,),
            "head_hidden_sizes": (16,),
            "features_dim": 32,
            "recurrent_hidden_size": 32,
            "recurrent_num_layers": 1,
        },
    )

    result = train_recurrent_ppo(config, run_suffix="smoke")
    metrics = evaluate_checkpoint(result.checkpoint_path, num_episodes=1)

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 64
    assert "eval_return_mean" in metrics
