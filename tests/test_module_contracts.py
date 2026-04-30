from pathlib import Path

import torch

from axiomrl.algorithms.base import UpdateResult
from axiomrl.data import RecurrentRolloutBuffer
from axiomrl.experiment.config import TrainConfig
from axiomrl.experiment.runs import RunContext
from axiomrl.models import CNNDrQv2Model, LSTMActorCritic, NatureCNN
from axiomrl.policies.base import PolicyOutput
from axiomrl.runtime.collector import CollectResult
from axiomrl.runtime.evaluator import EvalResult
from axiomrl.runtime.trainer import TrainResult


def test_contract_dataclasses_are_instantiable(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=1,
        total_timesteps=128,
        output_dir=tmp_path,
    )
    context = RunContext(
        run_id="run-1",
        run_dir=tmp_path / "run-1",
        checkpoints_dir=tmp_path / "run-1" / "checkpoints",
        tensorboard_dir=tmp_path / "run-1" / "tb",
        config_path=tmp_path / "run-1" / "config.yaml",
        metadata_path=tmp_path / "run-1" / "metadata.json",
    )
    policy_output = PolicyOutput(actions=None, logprobs=None, values=None, entropy=None, state=None)
    collect_result = CollectResult(num_env_steps=8, num_episodes=2, metrics={"return_mean": 1.0})
    update_result = UpdateResult(metrics={"loss": 0.5}, num_gradient_steps=1)
    eval_result = EvalResult(num_episodes=4, metrics={"return_mean": 2.0})
    train_result = TrainResult(
        run_dir=context.run_dir,
        checkpoint_path=None,
        metrics={"global_step": 8},
    )

    assert config.algo == "ppo"
    assert context.run_id == "run-1"
    assert policy_output.state is None
    assert collect_result.num_env_steps == 8
    assert update_result.num_gradient_steps == 1
    assert eval_result.num_episodes == 4
    assert train_result.metrics["global_step"] == 8


def test_image_and_recurrent_module_contracts_are_instantiable() -> None:
    cnn = NatureCNN(obs_shape=(4, 84, 84), features_dim=128)
    drqv2_model = CNNDrQv2Model(
        obs_shape=(9, 84, 84),
        action_dim=1,
        features_dim=64,
        actor_hidden_sizes=(32,),
        critic_hidden_sizes=(32,),
    )
    recurrent_buffer = RecurrentRolloutBuffer(
        num_steps=4,
        num_envs=2,
        obs_shape=(4,),
        hidden_size=16,
        num_layers=1,
    )
    recurrent_policy = LSTMActorCritic(
        obs_shape=(4,),
        action_dim=2,
        features_dim=32,
        encoder_hidden_sizes=(16,),
        head_hidden_sizes=(16,),
        hidden_size=32,
        num_layers=1,
    )

    features = cnn(torch.zeros((2, 4, 84, 84), dtype=torch.uint8))
    drqv2_actions = drqv2_model.actor(torch.zeros((2, 9, 84, 84), dtype=torch.uint8))
    initial_state = recurrent_policy.initial_state(2)

    assert features.shape == (2, 128)
    assert drqv2_actions.shape == (2, 1)
    assert recurrent_buffer.obs.shape == (4, 2, 4)
    assert initial_state[0].shape == (1, 2, 32)
    assert initial_state[1].shape == (1, 2, 32)
