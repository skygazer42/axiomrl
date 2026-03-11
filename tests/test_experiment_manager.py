from pathlib import Path

from rl_training.experiment.config import TrainConfig
from rl_training.experiment.default_manager import DefaultExperimentManager
from rl_training.experiment.registry import get_algorithm_spec


def test_builtin_algorithm_specs_are_registered() -> None:
    a2c = get_algorithm_spec("a2c")
    ppo = get_algorithm_spec("ppo")
    dqn = get_algorithm_spec("dqn")
    iql = get_algorithm_spec("iql")
    sac = get_algorithm_spec("sac")
    cql = get_algorithm_spec("cql")
    td3_bc = get_algorithm_spec("td3_bc")
    tqc = get_algorithm_spec("tqc")
    redq = get_algorithm_spec("redq")
    td3 = get_algorithm_spec("td3")

    assert a2c.name == "a2c"
    assert ppo.name == "ppo"
    assert dqn.name == "dqn"
    assert iql.name == "iql"
    assert sac.name == "sac"
    assert cql.name == "cql"
    assert td3_bc.name == "td3_bc"
    assert tqc.name == "tqc"
    assert redq.name == "redq"
    assert td3.name == "td3"
    assert callable(a2c.train_fn)
    assert callable(a2c.evaluate_fn)
    assert callable(a2c.predict_fn)
    assert callable(ppo.train_fn)
    assert callable(ppo.evaluate_fn)
    assert callable(ppo.predict_fn)
    assert callable(dqn.train_fn)
    assert callable(dqn.evaluate_fn)
    assert callable(dqn.predict_fn)
    assert callable(iql.train_fn)
    assert callable(iql.evaluate_fn)
    assert callable(iql.predict_fn)
    assert callable(sac.train_fn)
    assert callable(sac.evaluate_fn)
    assert callable(sac.predict_fn)
    assert callable(cql.train_fn)
    assert callable(cql.evaluate_fn)
    assert callable(cql.predict_fn)
    assert callable(td3_bc.train_fn)
    assert callable(td3_bc.evaluate_fn)
    assert callable(td3_bc.predict_fn)
    assert callable(tqc.train_fn)
    assert callable(tqc.evaluate_fn)
    assert callable(tqc.predict_fn)
    assert callable(redq.train_fn)
    assert callable(redq.evaluate_fn)
    assert callable(redq.predict_fn)
    assert callable(td3.train_fn)
    assert callable(td3.evaluate_fn)
    assert callable(td3.predict_fn)


def test_default_experiment_manager_setup_runs_training(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=43,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "hidden_sizes": (16, 16),
        },
    )

    trainer = manager.setup(config)
    result = trainer.train()

    assert result.run_dir.exists()
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_default_experiment_manager_resume_advances_training(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="dqn",
        env_id="CartPole-v1",
        seed=47,
        total_timesteps=96,
        output_dir=tmp_path,
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

    initial = manager.setup(config).train()
    resumed = manager.resume(initial.checkpoint_path, total_timesteps=160).train()

    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 160
