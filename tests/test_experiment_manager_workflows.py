from pathlib import Path
from types import SimpleNamespace

import rl_training.experiment.default_manager as default_manager_module
from rl_training.experiment.config import TrainConfig
from rl_training.experiment.default_manager import DefaultExperimentManager
from rl_training.experiment.sweeps import SeedSweepPlan
from rl_training.runtime import runner as runner_module
from rl_training.runtime.runner import FunctionRunner
from rl_training.runtime.trainer import TrainResult


def test_default_experiment_manager_setup_runner_executes_training(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=91,
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

    manager = DefaultExperimentManager()
    runner = manager.setup_runner(config)

    result = runner.run()

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


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


def test_default_experiment_manager_handles_bc(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="bc",
        env_id="Pendulum-v1",
        seed=61,
        total_timesteps=16,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 13,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "eval_interval": 8,
        },
    )

    result = manager.setup(config).train()

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_default_experiment_manager_handles_awr(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="awr",
        env_id="Pendulum-v1",
        seed=62,
        total_timesteps=16,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 14,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "beta": 1.0,
            "max_weight": 20.0,
            "eval_interval": 8,
        },
    )

    result = manager.setup(config).train()

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_default_experiment_manager_handles_marwil(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="marwil",
        env_id="Pendulum-v1",
        seed=62,
        total_timesteps=16,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 14,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "beta": 1.0,
            "vf_coeff": 1.0,
            "moving_average_sqd_adv_norm_start": 100.0,
            "moving_average_sqd_adv_norm_update_rate": 0.05,
            "eval_interval": 8,
        },
    )

    result = manager.setup(config).train()

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_default_experiment_manager_handles_awac(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="awac",
        env_id="Pendulum-v1",
        seed=63,
        total_timesteps=16,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 15,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "eval_interval": 8,
        },
    )

    result = manager.setup(config).train()

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_default_experiment_manager_handles_crr(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="crr",
        env_id="Pendulum-v1",
        seed=64,
        total_timesteps=16,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 18,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "beta": 1.0,
            "n_action_samples": 4,
            "max_weight": 20.0,
            "advantage_type": "mean",
            "weight_type": "exp",
            "eval_interval": 8,
        },
    )

    result = manager.setup(config).train()

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_default_experiment_manager_handles_rebrac(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="rebrac",
        env_id="Pendulum-v1",
        seed=65,
        total_timesteps=16,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 19,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_delay": 2,
            "actor_bc_weight": 1.0,
            "critic_bc_weight": 1.0,
            "actor_q_weight": 1.0,
            "eval_interval": 8,
        },
    )

    result = manager.setup(config).train()

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_default_experiment_manager_handles_cal_ql(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="cal_ql",
        env_id="Pendulum-v1",
        seed=66,
        total_timesteps=16,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 20,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
            "cql_alpha": 5.0,
            "num_cql_samples": 10,
            "eval_interval": 8,
        },
    )

    result = manager.setup(config).train()

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_default_experiment_manager_handles_edac(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="edac",
        env_id="Pendulum-v1",
        seed=66,
        total_timesteps=16,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 20,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
            "num_critics": 4,
            "eta": 1.0,
            "eval_interval": 8,
        },
    )

    result = manager.setup(config).train()

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_default_experiment_manager_handles_rlpd(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="rlpd",
        env_id="Pendulum-v1",
        seed=67,
        total_timesteps=16,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 21,
            "buffer_capacity": 256,
            "batch_size": 16,
            "learning_starts": 8,
            "train_frequency": 1,
            "gradient_updates_per_step": 2,
            "offline_pretrain_updates": 4,
            "offline_batch_ratio": 0.5,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
            "eval_interval": 8,
        },
    )

    result = manager.setup(config).train()

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_default_experiment_manager_handles_xql(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="xql",
        env_id="Pendulum-v1",
        seed=67,
        total_timesteps=16,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 21,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "beta": 3.0,
            "loss_temperature": 1.0,
            "max_advantage_weight": 100.0,
            "max_value_diff_exp": 5.0,
            "eval_interval": 8,
        },
    )

    result = manager.setup(config).train()

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_default_experiment_manager_handles_bear(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="bear",
        env_id="Pendulum-v1",
        seed=64,
        total_timesteps=16,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 16,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "latent_dim": 2,
            "mmd_sigma": 20.0,
            "mmd_alpha": 10.0,
            "num_mmd_action_samples": 10,
            "eval_interval": 8,
        },
    )

    result = manager.setup(config).train()

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_default_experiment_manager_handles_bcq(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="bcq",
        env_id="Pendulum-v1",
        seed=64,
        total_timesteps=16,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "dataset_kind": "random",
            "dataset_size": 64,
            "dataset_seed": 17,
            "batch_size": 16,
            "hidden_sizes": (16, 16),
            "latent_dim": 2,
            "num_action_samples": 10,
            "perturbation_scale": 0.05,
            "eval_interval": 8,
        },
    )

    result = manager.setup(config).train()

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_default_experiment_manager_handles_crossq(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="crossq",
        env_id="Pendulum-v1",
        seed=65,
        total_timesteps=128,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "critic_hidden_sizes": (32, 32),
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "alpha": 0.1,
            "policy_delay": 1,
            "adam_beta1": 0.5,
            "bn_momentum": 0.99,
        },
    )

    result = manager.setup(config).train()

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_default_experiment_manager_handles_her(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="her",
        env_id="RL-PointGoal1D-v0",
        seed=65,
        total_timesteps=32,
        output_dir=tmp_path,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 16,
            "learning_starts": 8,
            "hidden_sizes": (16, 16),
            "eval_interval": 8,
        },
    )

    result = manager.setup(config).train()

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_default_experiment_manager_handles_trpo(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="trpo",
        env_id="CartPole-v1",
        seed=66,
        total_timesteps=64,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "hidden_sizes": (16, 16),
            "learning_rate": 1e-3,
            "value_updates": 3,
            "max_kl": 0.01,
            "cg_iterations": 5,
            "cg_damping": 0.1,
            "line_search_steps": 5,
            "line_search_shrink": 0.8,
        },
    )

    result = manager.setup(config).train()

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()


def test_default_experiment_manager_handles_discrete_sac(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="discrete_sac",
        env_id="CartPole-v1",
        seed=67,
        total_timesteps=128,
        output_dir=tmp_path,
        num_envs=2,
        eval_episodes=1,
        algo_kwargs={
            "buffer_capacity": 512,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "hidden_sizes": (32, 32),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "alpha": 0.2,
            "tau": 0.005,
        },
    )

    result = manager.setup(config).train()

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


def test_default_experiment_manager_handles_recurrent_ppo(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="recurrent_ppo",
        env_id="CartPole-v1",
        seed=53,
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

    initial = manager.setup(config).train()
    resumed = manager.resume(initial.checkpoint_path, total_timesteps=128).train()

    assert initial.checkpoint_path is not None
    assert resumed.checkpoint_path is not None
    assert resumed.checkpoint_path.exists()
    assert resumed.metrics["global_step"] >= 128


def test_default_experiment_manager_uses_benchmark_runner_for_seed_sweeps(tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    benchmark_config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=77,
        total_timesteps=64,
        output_dir=tmp_path / "benchmark-runs",
        num_envs=1,
        eval_episodes=1,
        benchmark={"seeds": [3, 5]},
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "hidden_sizes": (16, 16),
        },
    )
    single_config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=78,
        total_timesteps=64,
        output_dir=tmp_path / "single-run",
        num_envs=1,
        eval_episodes=1,
        algo_kwargs={
            "num_steps": 32,
            "update_epochs": 1,
            "minibatch_size": 32,
            "hidden_sizes": (16, 16),
        },
    )

    benchmark_runner = manager.setup_runner(benchmark_config)
    single_runner = manager.setup_runner(single_config)

    benchmark_runner_cls = getattr(runner_module, "BenchmarkRunner", None)
    assert benchmark_runner_cls is not None
    assert isinstance(benchmark_runner, benchmark_runner_cls)
    assert isinstance(benchmark_runner.seed_sweep, SeedSweepPlan)
    assert benchmark_runner.seed_sweep.seeds == (3, 5)
    assert callable(benchmark_runner.run)
    assert isinstance(single_runner, FunctionRunner)


def test_default_experiment_manager_clones_callbacks_per_seed_run(monkeypatch, tmp_path: Path) -> None:
    manager = DefaultExperimentManager()
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=100,
        total_timesteps=64,
        output_dir=tmp_path / "benchmark-runs",
        num_envs=1,
        eval_episodes=1,
        benchmark={"seeds": [3, 5]},
    )
    captured_callbacks: list[tuple[object, ...] | None] = []

    def fake_train_fn(train_config: TrainConfig, *, callbacks=None):
        run_dir = config.output_dir / f"seed-{train_config.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        captured_callbacks.append(tuple(callbacks) if callbacks is not None else None)
        return TrainResult(
            run_dir=run_dir,
            checkpoint_path=None,
            metrics={"eval_return_mean": float(train_config.seed), "global_step": 64.0},
        )

    monkeypatch.setattr(
        default_manager_module,
        "get_algorithm_spec",
        lambda _: SimpleNamespace(train_fn=fake_train_fn),
    )

    class StatefulCallback:
        def __init__(self) -> None:
            self.history: list[str] = []

        def on_train_start(self, trainer: object) -> None:
            self.history.append("start")

        def on_collect_end(self, trainer: object, result: object) -> None:
            self.history.append("collect")

        def on_update_end(self, trainer: object, result: object) -> None:
            self.history.append("update")

        def on_eval_end(self, trainer: object, metrics: object) -> None:
            self.history.append("eval")

        def on_train_end(self, trainer: object, result: object) -> None:
            self.history.append("end")

    original_callback = StatefulCallback()
    runner = manager.setup_runner(config, callbacks=(original_callback,))
    runner.run()

    assert len(captured_callbacks) == 2
    assert captured_callbacks[0] is not None
    assert captured_callbacks[1] is not None

    first_callback = captured_callbacks[0][0]
    second_callback = captured_callbacks[1][0]
    assert first_callback is not original_callback
    assert second_callback is not original_callback
    assert first_callback is not second_callback
