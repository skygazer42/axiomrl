from pathlib import Path

from axiomrl.api import (
    A2C,
    AWR,
    C51DQN,
    CQL,
    DDPG,
    DQN,
    EDAC,
    IQL,
    IQN,
    MARWIL,
    QRDQN,
    REDQ,
    RLPD,
    SAC,
    TD3,
    TD3BC,
    TQC,
    XQL,
    CalQL,
    DoubleDQN,
    DuelingDQN,
    NoisyDQN,
    NStepDQN,
    PrioritizedDQN,
    RainbowDQN,
)
from axiomrl.experiment.config import TrainConfig


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
    iqn = IQN(
        TrainConfig(
            algo="iqn",
            env_id="CartPole-v1",
            seed=60,
            total_timesteps=96,
            output_dir=tmp_path / "iqn-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 256,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "target_update_interval": 16,
                "hidden_sizes": (16, 16),
                "gamma": 0.99,
                "num_quantiles": 16,
                "embedding_dim": 32,
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
    cql = CQL(
        TrainConfig(
            algo="cql",
            env_id="Pendulum-v1",
            seed=62,
            total_timesteps=96,
            output_dir=tmp_path / "cql-runs",
            eval_episodes=1,
            algo_kwargs={
                "dataset_kind": "random",
                "dataset_size": 192,
                "dataset_seed": 17,
                "batch_size": 32,
                "hidden_sizes": (32, 32),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "alpha": 0.2,
                "tau": 0.005,
                "cql_alpha": 5.0,
                "num_cql_samples": 10,
            },
        )
    )
    cal_ql = CalQL(
        TrainConfig(
            algo="cal_ql",
            env_id="Pendulum-v1",
            seed=63,
            total_timesteps=96,
            output_dir=tmp_path / "cal-ql-runs",
            eval_episodes=1,
            algo_kwargs={
                "dataset_kind": "random",
                "dataset_size": 192,
                "dataset_seed": 18,
                "batch_size": 32,
                "hidden_sizes": (32, 32),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "alpha": 0.2,
                "tau": 0.005,
                "cql_alpha": 5.0,
                "num_cql_samples": 10,
            },
        )
    )
    edac = EDAC(
        TrainConfig(
            algo="edac",
            env_id="Pendulum-v1",
            seed=64,
            total_timesteps=96,
            output_dir=tmp_path / "edac-runs",
            eval_episodes=1,
            algo_kwargs={
                "dataset_kind": "random",
                "dataset_size": 192,
                "dataset_seed": 19,
                "batch_size": 32,
                "hidden_sizes": (32, 32),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "alpha": 0.2,
                "tau": 0.005,
                "num_critics": 4,
                "eta": 1.0,
            },
        )
    )
    awr = AWR(
        TrainConfig(
            algo="awr",
            env_id="Pendulum-v1",
            seed=65,
            total_timesteps=96,
            output_dir=tmp_path / "awr-runs",
            eval_episodes=1,
            algo_kwargs={
                "dataset_kind": "random",
                "dataset_size": 192,
                "dataset_seed": 20,
                "batch_size": 32,
                "hidden_sizes": (32, 32),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "beta": 1.0,
                "max_weight": 20.0,
            },
        )
    )
    marwil = MARWIL(
        TrainConfig(
            algo="marwil",
            env_id="Pendulum-v1",
            seed=66,
            total_timesteps=96,
            output_dir=tmp_path / "marwil-runs",
            eval_episodes=1,
            algo_kwargs={
                "dataset_kind": "random",
                "dataset_size": 192,
                "dataset_seed": 21,
                "batch_size": 32,
                "hidden_sizes": (32, 32),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "beta": 1.0,
                "vf_coeff": 1.0,
                "moving_average_sqd_adv_norm_start": 100.0,
                "moving_average_sqd_adv_norm_update_rate": 0.05,
            },
        )
    )
    rlpd = RLPD(
        TrainConfig(
            algo="rlpd",
            env_id="Pendulum-v1",
            seed=67,
            total_timesteps=96,
            output_dir=tmp_path / "rlpd-runs",
            eval_episodes=1,
            algo_kwargs={
                "dataset_kind": "random",
                "dataset_size": 192,
                "dataset_seed": 22,
                "buffer_capacity": 512,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "gradient_updates_per_step": 2,
                "offline_pretrain_updates": 8,
                "offline_batch_ratio": 0.5,
                "hidden_sizes": (32, 32),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "alpha": 0.2,
                "tau": 0.005,
            },
        )
    )
    tqc = TQC(
        TrainConfig(
            algo="tqc",
            env_id="Pendulum-v1",
            seed=62,
            total_timesteps=96,
            output_dir=tmp_path / "tqc-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 512,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "hidden_sizes": (32, 32),
                "alpha": 0.2,
                "tau": 0.005,
                "num_critics": 3,
                "num_quantiles": 7,
                "top_quantiles_to_drop_per_net": 1,
                "kappa": 1.0,
            },
        )
    )
    redq = REDQ(
        TrainConfig(
            algo="redq",
            env_id="Pendulum-v1",
            seed=63,
            total_timesteps=96,
            output_dir=tmp_path / "redq-runs",
            eval_episodes=1,
            algo_kwargs={
                "buffer_capacity": 512,
                "batch_size": 32,
                "learning_starts": 32,
                "train_frequency": 1,
                "gradient_updates_per_step": 2,
                "hidden_sizes": (32, 32),
                "alpha": 0.2,
                "tau": 0.005,
                "num_critics": 5,
                "subset_size": 2,
            },
        )
    )
    iql = IQL(
        TrainConfig(
            algo="iql",
            env_id="Pendulum-v1",
            seed=64,
            total_timesteps=96,
            output_dir=tmp_path / "iql-runs",
            eval_episodes=1,
            algo_kwargs={
                "dataset_kind": "random",
                "dataset_size": 192,
                "dataset_seed": 21,
                "batch_size": 32,
                "hidden_sizes": (32, 32),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "expectile": 0.7,
                "beta": 3.0,
                "max_advantage_weight": 100.0,
            },
        )
    )
    xql = XQL(
        TrainConfig(
            algo="xql",
            env_id="Pendulum-v1",
            seed=65,
            total_timesteps=96,
            output_dir=tmp_path / "xql-runs",
            eval_episodes=1,
            algo_kwargs={
                "dataset_kind": "random",
                "dataset_size": 192,
                "dataset_seed": 22,
                "batch_size": 32,
                "hidden_sizes": (32, 32),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "beta": 3.0,
                "loss_temperature": 1.0,
                "max_advantage_weight": 100.0,
                "max_value_diff_exp": 5.0,
            },
        )
    )
    td3_bc = TD3BC(
        TrainConfig(
            algo="td3_bc",
            env_id="Pendulum-v1",
            seed=65,
            total_timesteps=96,
            output_dir=tmp_path / "td3-bc-runs",
            eval_episodes=1,
            algo_kwargs={
                "dataset_kind": "random",
                "dataset_size": 192,
                "dataset_seed": 23,
                "batch_size": 32,
                "hidden_sizes": (32, 32),
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "policy_noise": 0.2,
                "noise_clip": 0.5,
                "policy_delay": 2,
                "bc_alpha": 2.5,
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
    iqn.learn()
    sac.learn()
    cql.learn()
    cal_ql.learn()
    edac.learn()
    awr.learn()
    marwil.learn()
    rlpd.learn()
    tqc.learn()
    redq.learn()
    iql.learn()
    xql.learn()
    td3_bc.learn()
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
    iqn_action = iqn.predict([0.0, 0.0, 0.0, 0.0])
    sac_action = sac.predict([0.0, 0.0, 0.0])
    cql_action = cql.predict([0.0, 0.0, 0.0])
    cal_ql_action = cal_ql.predict([0.0, 0.0, 0.0])
    edac_action = edac.predict([0.0, 0.0, 0.0])
    awr_action = awr.predict([0.0, 0.0, 0.0])
    marwil_action = marwil.predict([0.0, 0.0, 0.0])
    rlpd_action = rlpd.predict([0.0, 0.0, 0.0])
    tqc_action = tqc.predict([0.0, 0.0, 0.0])
    redq_action = redq.predict([0.0, 0.0, 0.0])
    iql_action = iql.predict([0.0, 0.0, 0.0])
    xql_action = xql.predict([0.0, 0.0, 0.0])
    td3_bc_action = td3_bc.predict([0.0, 0.0, 0.0])
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
    assert isinstance(iqn_action, int)
    assert len(sac_action) == 1
    assert len(cql_action) == 1
    assert len(cal_ql_action) == 1
    assert len(edac_action) == 1
    assert len(awr_action) == 1
    assert len(marwil_action) == 1
    assert len(rlpd_action) == 1
    assert len(tqc_action) == 1
    assert len(redq_action) == 1
    assert len(iql_action) == 1
    assert len(xql_action) == 1
    assert len(td3_bc_action) == 1
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
    assert "eval_return_mean" in iqn.evaluate(num_episodes=1)
    assert "eval_return_mean" in sac.evaluate(num_episodes=1)
    assert "eval_return_mean" in cql.evaluate(num_episodes=1)
    assert "eval_return_mean" in cal_ql.evaluate(num_episodes=1)
    assert "eval_return_mean" in edac.evaluate(num_episodes=1)
    assert "eval_return_mean" in awr.evaluate(num_episodes=1)
    assert "eval_return_mean" in marwil.evaluate(num_episodes=1)
    assert "eval_return_mean" in rlpd.evaluate(num_episodes=1)
    assert "eval_return_mean" in tqc.evaluate(num_episodes=1)
    assert "eval_return_mean" in redq.evaluate(num_episodes=1)
    assert "eval_return_mean" in iql.evaluate(num_episodes=1)
    assert "eval_return_mean" in xql.evaluate(num_episodes=1)
    assert "eval_return_mean" in td3_bc.evaluate(num_episodes=1)
    assert "eval_return_mean" in ddpg.evaluate(num_episodes=1)
    assert "eval_return_mean" in td3.evaluate(num_episodes=1)
