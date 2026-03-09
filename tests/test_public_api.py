from pathlib import Path

from rl_training.api import A2C, DQN, PPO, SAC, TD3
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
    sac.learn()
    td3.learn()

    a2c_action = a2c.predict([0.0, 0.0, 0.0, 0.0])
    dqn_action = dqn.predict([0.0, 0.0, 0.0, 0.0])
    sac_action = sac.predict([0.0, 0.0, 0.0])
    td3_action = td3.predict([0.0, 0.0, 0.0])

    assert isinstance(a2c_action, int)
    assert isinstance(dqn_action, int)
    assert len(sac_action) == 1
    assert len(td3_action) == 1
    assert "eval_return_mean" in a2c.evaluate(num_episodes=1)
    assert "eval_return_mean" in dqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in sac.evaluate(num_episodes=1)
    assert "eval_return_mean" in td3.evaluate(num_episodes=1)
