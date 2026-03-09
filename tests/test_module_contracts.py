from pathlib import Path

from rl_training.algorithms.base import UpdateResult
from rl_training.experiment.config import TrainConfig
from rl_training.experiment.runs import RunContext
from rl_training.policies.base import PolicyOutput
from rl_training.runtime.collector import CollectResult
from rl_training.runtime.evaluator import EvalResult
from rl_training.runtime.trainer import TrainResult


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
