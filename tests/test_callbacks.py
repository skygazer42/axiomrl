from pathlib import Path

from axiomrl.algorithms.base import UpdateResult
from axiomrl.api import DQN, PPO
from axiomrl.experiment.config import TrainConfig
from axiomrl.runtime.collector import CollectResult
from axiomrl.runtime.controls import EarlyStoppingCallback, EarlyStoppingConfig
from axiomrl.runtime.trainer import TrainerState, TrainResult
from axiomrl.runtime.types import MetricDict


class RecordingCallback:
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_train_start(self, trainer: object) -> None:
        self.events.append(f"train_start:{getattr(trainer, 'algorithm', 'unknown')}")

    def on_collect_end(self, trainer: object, result: CollectResult) -> None:
        self.events.append(f"collect:{result.num_env_steps}")

    def on_update_end(self, trainer: object, result: UpdateResult) -> None:
        self.events.append(f"update:{result.num_gradient_steps}")

    def on_eval_end(self, trainer: object, metrics: MetricDict) -> None:
        self.events.append(f"eval:{int(metrics['eval_episodes'])}")

    def on_train_end(self, trainer: object, result: TrainResult) -> None:
        self.events.append(f"train_end:{int(result.metrics['global_step'])}")


def test_public_api_training_emits_callbacks_for_ppo_and_dqn(tmp_path: Path) -> None:
    ppo = PPO(
        TrainConfig(
            algo="ppo",
            env_id="CartPole-v1",
            seed=67,
            total_timesteps=64,
            output_dir=tmp_path / "ppo-runs",
            num_envs=2,
            eval_episodes=1,
            algo_kwargs={
                "num_steps": 32,
                "update_epochs": 1,
                "minibatch_size": 32,
                "hidden_sizes": (16, 16),
            },
        )
    )
    dqn = DQN(
        TrainConfig(
            algo="dqn",
            env_id="CartPole-v1",
            seed=71,
            total_timesteps=96,
            output_dir=tmp_path / "dqn-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 256,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "target_update_interval": 16,
                "hidden_sizes": (16, 16),
            },
        )
    )

    ppo_callback = RecordingCallback()
    dqn_callback = RecordingCallback()

    ppo.learn(callbacks=[ppo_callback])
    dqn.learn(callbacks=[dqn_callback])

    assert any(event.startswith("train_start:ppo") for event in ppo_callback.events)
    assert any(event.startswith("collect:") for event in ppo_callback.events)
    assert any(event.startswith("update:") for event in ppo_callback.events)
    assert any(event.startswith("eval:") for event in ppo_callback.events)
    assert any(event.startswith("train_end:") for event in ppo_callback.events)

    assert any(event.startswith("train_start:dqn") for event in dqn_callback.events)
    assert any(event.startswith("collect:") for event in dqn_callback.events)
    assert any(event.startswith("update:") for event in dqn_callback.events)
    assert any(event.startswith("eval:") for event in dqn_callback.events)
    assert any(event.startswith("train_end:") for event in dqn_callback.events)


def test_early_stopping_callback_implements_callback_protocol(tmp_path: Path) -> None:
    callback = EarlyStoppingCallback(EarlyStoppingConfig())
    trainer = TrainerState(algorithm="ppo", run_dir=tmp_path, global_step=1)

    callback.on_train_start(trainer)
    callback.on_collect_end(trainer, CollectResult(num_env_steps=1, num_episodes=0, metrics={}, last_obs=None))
    callback.on_update_end(trainer, UpdateResult(metrics={}, num_gradient_steps=0))
    callback.on_eval_end(trainer, {"eval_return_mean": 0.0, "eval_episodes": 1.0})
    callback.on_train_end(trainer, TrainResult(run_dir=tmp_path, checkpoint_path=None, metrics={"global_step": 0.0}))

    assert trainer.global_step == 1
