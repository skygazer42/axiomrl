from pathlib import Path

from rl_training.api import A2C, C51DQN, DDPG, DQN, DoubleDQN, DuelingDQN, NoisyDQN, NStepDQN, PPO, PrioritizedDQN, QRDQN, RainbowDQN, SAC, TD3
from rl_training.experiment.config import TrainConfig


def test_ppo_public_api_supports_learn_save_load_and_evaluate(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=53,
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

    algo = PPO(config)
    result = algo.learn()
    exported = algo.save(tmp_path / "exports" / "ppo.pt")
    loaded = PPO.load(exported)
    action = loaded.predict([0.0, 0.0, 0.0, 0.0])
    metrics = loaded.evaluate(num_episodes=1)

    assert result.checkpoint_path is not None
    assert exported.exists()
    assert isinstance(action, int)
    assert set(metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_off_policy_public_apis_support_learn_and_evaluate(tmp_path: Path) -> None:
    a2c = A2C(
        TrainConfig(
            algo="a2c",
            env_id="CartPole-v1",
            seed=57,
            total_timesteps=64,
            output_dir=tmp_path / "a2c-runs",
            num_envs=2,
            eval_episodes=1,
            algo_kwargs={
                "num_steps": 32,
                "hidden_sizes": (16, 16),
                "learning_rate": 3e-4,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "gamma": 0.99,
                "gae_lambda": 0.95,
            },
        )
    )
    dqn = DQN(
        TrainConfig(
            algo="dqn",
            env_id="CartPole-v1",
            seed=59,
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
    double_dqn = DoubleDQN(
        TrainConfig(
            algo="double_dqn",
            env_id="CartPole-v1",
            seed=60,
            total_timesteps=96,
            output_dir=tmp_path / "double-dqn-runs",
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
    dueling_dqn = DuelingDQN(
        TrainConfig(
            algo="dueling_dqn",
            env_id="CartPole-v1",
            seed=60,
            total_timesteps=96,
            output_dir=tmp_path / "dueling-dqn-runs",
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
    noisy_dqn = NoisyDQN(
        TrainConfig(
            algo="noisy_dqn",
            env_id="CartPole-v1",
            seed=60,
            total_timesteps=96,
            output_dir=tmp_path / "noisy-dqn-runs",
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
    prioritized_dqn = PrioritizedDQN(
        TrainConfig(
            algo="prioritized_dqn",
            env_id="CartPole-v1",
            seed=60,
            total_timesteps=96,
            output_dir=tmp_path / "prioritized-dqn-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 256,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "target_update_interval": 16,
                "hidden_sizes": (16, 16),
                "prioritized_alpha": 0.6,
                "prioritized_beta_start": 0.4,
            },
        )
    )
    rainbow_dqn = RainbowDQN(
        TrainConfig(
            algo="rainbow_dqn",
            env_id="CartPole-v1",
            seed=60,
            total_timesteps=96,
            output_dir=tmp_path / "rainbow-dqn-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 256,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "target_update_interval": 16,
                "hidden_sizes": (16, 16),
                "epsilon_start": 0.0,
                "epsilon_end": 0.0,
                "exploration_fraction": 0.0,
                "prioritized_alpha": 0.6,
                "prioritized_beta_start": 0.4,
            },
        )
    )
    c51_dqn = C51DQN(
        TrainConfig(
            algo="c51_dqn",
            env_id="CartPole-v1",
            seed=60,
            total_timesteps=96,
            output_dir=tmp_path / "c51-dqn-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 256,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "target_update_interval": 16,
                "hidden_sizes": (16, 16),
                "v_min": 0.0,
                "v_max": 200.0,
                "num_atoms": 51,
            },
        )
    )
    n_step_dqn = NStepDQN(
        TrainConfig(
            algo="n_step_dqn",
            env_id="CartPole-v1",
            seed=60,
            total_timesteps=96,
            output_dir=tmp_path / "n-step-dqn-runs",
            eval_episodes=1,
            algo_kwargs={
                "n_step": 3,
                "gamma": 0.99,
                "buffer_capacity": 256,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "target_update_interval": 16,
                "hidden_sizes": (16, 16),
            },
        )
    )
    qr_dqn = QRDQN(
        TrainConfig(
            algo="qr_dqn",
            env_id="CartPole-v1",
            seed=60,
            total_timesteps=96,
            output_dir=tmp_path / "qr-dqn-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 256,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "target_update_interval": 16,
                "hidden_sizes": (16, 16),
                "gamma": 0.99,
                "num_quantiles": 51,
                "kappa": 1.0,
            },
        )
    )
    sac = SAC(
        TrainConfig(
            algo="sac",
            env_id="Pendulum-v1",
            seed=61,
            total_timesteps=96,
            output_dir=tmp_path / "sac-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 512,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "hidden_sizes": (32, 32),
                "alpha": 0.2,
                "tau": 0.005,
            },
        )
    )
    ddpg = DDPG(
        TrainConfig(
            algo="ddpg",
            env_id="Pendulum-v1",
            seed=62,
            total_timesteps=96,
            output_dir=tmp_path / "ddpg-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 512,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "hidden_sizes": (32, 32),
                "tau": 0.005,
            },
        )
    )
    td3 = TD3(
        TrainConfig(
            algo="td3",
            env_id="Pendulum-v1",
            seed=63,
            total_timesteps=96,
            output_dir=tmp_path / "td3-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 512,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "hidden_sizes": (32, 32),
                "tau": 0.005,
                "policy_noise": 0.2,
                "noise_clip": 0.5,
                "policy_delay": 2,
            },
        )
    )

    a2c.learn()
    dqn.learn()
    double_dqn.learn()
    dueling_dqn.learn()
    noisy_dqn.learn()
    prioritized_dqn.learn()
    rainbow_dqn.learn()
    c51_dqn.learn()
    n_step_dqn.learn()
    qr_dqn.learn()
    sac.learn()
    ddpg.learn()
    td3.learn()

    a2c_action = a2c.predict([0.0, 0.0, 0.0, 0.0])
    dqn_action = dqn.predict([0.0, 0.0, 0.0, 0.0])
    double_dqn_action = double_dqn.predict([0.0, 0.0, 0.0, 0.0])
    dueling_dqn_action = dueling_dqn.predict([0.0, 0.0, 0.0, 0.0])
    noisy_dqn_action = noisy_dqn.predict([0.0, 0.0, 0.0, 0.0])
    prioritized_dqn_action = prioritized_dqn.predict([0.0, 0.0, 0.0, 0.0])
    rainbow_dqn_action = rainbow_dqn.predict([0.0, 0.0, 0.0, 0.0])
    c51_dqn_action = c51_dqn.predict([0.0, 0.0, 0.0, 0.0])
    n_step_dqn_action = n_step_dqn.predict([0.0, 0.0, 0.0, 0.0])
    qr_dqn_action = qr_dqn.predict([0.0, 0.0, 0.0, 0.0])
    sac_action = sac.predict([0.0, 0.0, 0.0])
    ddpg_action = ddpg.predict([0.0, 0.0, 0.0])
    td3_action = td3.predict([0.0, 0.0, 0.0])

    assert isinstance(a2c_action, int)
    assert isinstance(dqn_action, int)
    assert isinstance(double_dqn_action, int)
    assert isinstance(dueling_dqn_action, int)
    assert isinstance(noisy_dqn_action, int)
    assert isinstance(prioritized_dqn_action, int)
    assert isinstance(rainbow_dqn_action, int)
    assert isinstance(c51_dqn_action, int)
    assert isinstance(n_step_dqn_action, int)
    assert isinstance(qr_dqn_action, int)
    assert len(sac_action) == 1
    assert len(ddpg_action) == 1
    assert len(td3_action) == 1
    assert "eval_return_mean" in a2c.evaluate(num_episodes=1)
    assert "eval_return_mean" in dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in double_dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in dueling_dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in noisy_dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in prioritized_dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in rainbow_dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in c51_dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in n_step_dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in qr_dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in sac.evaluate(num_episodes=1)
    assert "eval_return_mean" in ddpg.evaluate(num_episodes=1)
    assert "eval_return_mean" in td3.evaluate(num_episodes=1)
