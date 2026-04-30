from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from axiomrl.runtime import (
    a2c_trainer,
    agent57_trainer,
    apex_dqn_trainer,
    appo_trainer,
    ars_trainer,
    awac_trainer,
    awr_trainer,
    bc_trainer,
    bcq_trainer,
    bear_trainer,
    cal_ql_trainer,
    cql_trainer,
    crossq_trainer,
    crr_trainer,
    curl_trainer,
    d4pg_trainer,
    ddpg_trainer,
    decision_transformer_trainer,
    discrete_sac_trainer,
    dreamer_trainer,
    drq_trainer,
    drqn_trainer,
    drqv2_trainer,
    edac_trainer,
    efficientzero_trainer,
    gail_trainer,
    her_trainer,
    impala_trainer,
    iql_trainer,
    marwil_trainer,
    mbpo_trainer,
    mopo_trainer,
    muzero_trainer,
    naf_trainer,
    openai_es_trainer,
    pets_trainer,
    ppg_trainer,
    r2d2_trainer,
    rebrac_trainer,
    recurrent_ppo_trainer,
    redq_trainer,
    rlpd_trainer,
    sac_trainer,
    td3_bc_trainer,
    td3_trainer,
    tqc_trainer,
    xql_trainer,
)
from tests.support.runtime_foundation import (
    ExpectedEvaluationSupport,
    _make_a2c_config,
    _make_agent57_config,
    _make_apex_dqn_config,
    _make_appo_config,
    _make_ars_config,
    _make_awac_config,
    _make_awr_config,
    _make_bc_config,
    _make_bcq_config,
    _make_bear_config,
    _make_cal_ql_config,
    _make_cql_config,
    _make_crossq_config,
    _make_crr_config,
    _make_curl_config,
    _make_d4pg_config,
    _make_ddpg_config,
    _make_decision_transformer_config,
    _make_discrete_sac_config,
    _make_dreamer_config,
    _make_drq_config,
    _make_drqn_config,
    _make_drqv2_config,
    _make_edac_config,
    _make_efficientzero_config,
    _make_gail_config,
    _make_her_config,
    _make_impala_config,
    _make_iql_config,
    _make_marwil_config,
    _make_mbpo_config,
    _make_mopo_config,
    _make_muzero_config,
    _make_naf_config,
    _make_openai_es_config,
    _make_pets_config,
    _make_ppg_config,
    _make_r2d2_config,
    _make_rebrac_config,
    _make_recurrent_ppo_config,
    _make_redq_config,
    _make_rlpd_config,
    _make_sac_config,
    _make_td3_bc_config,
    _make_td3_config,
    _make_tqc_config,
    _make_xql_config,
    _raise_expected_eval_support,
    _raise_legacy_eval_path,
)


def test_a2c_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(a2c_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(a2c_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    policy = SimpleNamespace(
        act=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([0])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        a2c_trainer._evaluate_policy(
            policy,
            _make_a2c_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_discrete_sac_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(discrete_sac_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(discrete_sac_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        sample_actions=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([0])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        discrete_sac_trainer._evaluate_discrete_sac_policy(
            model,
            _make_discrete_sac_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_ddpg_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ddpg_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(ddpg_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]]),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        ddpg_trainer._evaluate_ddpg_policy(
            model,
            _make_ddpg_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_td3_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(td3_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(td3_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]]),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        td3_trainer._evaluate_td3_policy(
            model,
            _make_td3_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_redq_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(redq_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(redq_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        sample_actions=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([[0.0]])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        redq_trainer._evaluate_redq_policy(
            model,
            _make_redq_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_sac_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sac_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(sac_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        sample_actions=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([[0.0]])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        sac_trainer._evaluate_sac_policy(
            model,
            _make_sac_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_crossq_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crossq_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(crossq_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        sample_actions=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([[0.0]])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        crossq_trainer._evaluate_crossq_policy(
            model,
            _make_crossq_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_tqc_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tqc_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(tqc_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        sample_actions=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([[0.0]])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        tqc_trainer._evaluate_tqc_policy(
            model,
            _make_tqc_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_d4pg_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(d4pg_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(d4pg_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]]),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        d4pg_trainer._evaluate_d4pg_policy(
            model,
            _make_d4pg_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_naf_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(naf_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(naf_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]]),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        naf_trainer._evaluate_naf_policy(
            model,
            _make_naf_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_drq_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(drq_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(drq_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]]),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        drq_trainer._evaluate_drq_policy(
            model,
            _make_drq_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_drqv2_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(drqv2_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(drqv2_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]]),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        drqv2_trainer._evaluate_drqv2_policy(
            model,
            _make_drqv2_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_curl_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(curl_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(curl_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]]),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        curl_trainer._evaluate_curl_policy(
            model,
            _make_curl_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_impala_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(impala_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(impala_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    policy = SimpleNamespace(
        act=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([0])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        impala_trainer._evaluate_impala_policy(
            policy,
            _make_impala_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_dreamer_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dreamer_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(dreamer_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        act=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([[0]])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        dreamer_trainer._evaluate_policy(
            model,
            _make_dreamer_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_apex_dqn_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(apex_dqn_trainer, "_evaluate_q_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        apex_dqn_trainer.train_apex_dqn(_make_apex_dqn_config(tmp_path), run_suffix="eval-helper")


def test_iql_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(iql_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(iql_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        sample_actions=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([[0.0]])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        iql_trainer._evaluate_iql_policy(
            model,
            _make_iql_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_gail_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gail_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(gail_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    policy = SimpleNamespace(
        act=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([[0]])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        gail_trainer._evaluate_policy(
            policy,
            _make_gail_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_mbpo_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mbpo_trainer, "_evaluate_sac_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        mbpo_trainer.train_mbpo(_make_mbpo_config(tmp_path), run_suffix="eval-helper")


def test_mopo_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mopo_trainer, "_evaluate_sac_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        mopo_trainer.train_mopo(_make_mopo_config(tmp_path), run_suffix="eval-helper")


def test_muzero_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(muzero_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(muzero_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    algorithm = SimpleNamespace(
        set_eval_mode=lambda: None,
        act=lambda obs, deterministic=True: SimpleNamespace(actions=torch.tensor([0])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        muzero_trainer._evaluate_muzero_policy(
            algorithm,
            _make_muzero_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_decision_transformer_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        decision_transformer_trainer,
        "evaluate_continuous_episodes",
        _raise_expected_eval_support,
        raising=False,
    )
    monkeypatch.setattr(decision_transformer_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        predict_last_action=lambda **kwargs: torch.tensor([[0.0]]),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        decision_transformer_trainer._evaluate_decision_transformer_policy(
            model,
            _make_decision_transformer_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
            context_length=4,
            target_return=0.0,
            max_timestep=64,
            gamma=0.99,
        )


def test_her_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(her_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(her_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]]),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        her_trainer._evaluate_her_policy(
            model,
            _make_her_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_efficientzero_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        efficientzero_trainer, "_maybe_run_muzero_evaluation", _raise_expected_eval_support, raising=False
    )

    with pytest.raises(ExpectedEvaluationSupport):
        efficientzero_trainer.train_efficientzero(_make_efficientzero_config(tmp_path), run_suffix="eval-helper")


def test_appo_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(appo_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(appo_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    policy = SimpleNamespace(
        act=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([0])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        appo_trainer._evaluate_appo_policy(
            policy,
            _make_appo_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_ppg_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ppg_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(ppg_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        act=lambda obs_tensor, deterministic=True: SimpleNamespace(actions=torch.tensor([0])),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        ppg_trainer._evaluate_ppg_policy(
            model,
            _make_ppg_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_ars_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ars_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(ars_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]], dtype=torch.float32),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        ars_trainer._evaluate_ars_policy(
            model,
            _make_ars_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_openai_es_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_es_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(openai_es_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]], dtype=torch.float32),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        openai_es_trainer._evaluate_openai_es_policy(
            model,
            _make_openai_es_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_pets_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pets_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(pets_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    algorithm = SimpleNamespace(
        set_eval_mode=lambda: None,
        plan_action=lambda *args, **kwargs: np.zeros((1,), dtype=np.float32),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        pets_trainer._evaluate_pets_policy(
            algorithm,
            _make_pets_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_awr_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(awr_trainer, "_evaluate_iql_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        awr_trainer.train_awr(_make_awr_config(tmp_path), run_suffix="eval-helper")


def test_awac_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(awac_trainer, "_evaluate_sac_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        awac_trainer.train_awac(_make_awac_config(tmp_path), run_suffix="eval-helper")


def test_cal_ql_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cal_ql_trainer, "_evaluate_sac_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        cal_ql_trainer.train_cal_ql(_make_cal_ql_config(tmp_path), run_suffix="eval-helper")


def test_bc_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bc_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(bc_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        actor=lambda obs_tensor: torch.tensor([[0.0]], dtype=torch.float32),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        bc_trainer._evaluate_bc_policy(
            model,
            _make_bc_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_bcq_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bcq_trainer, "evaluate_continuous_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(bcq_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    model = SimpleNamespace(
        select_actions=lambda obs_tensor, num_action_samples, deterministic=True: torch.tensor(
            [[0.0]], dtype=torch.float32
        ),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        bcq_trainer._evaluate_bcq_policy(
            model,
            _make_bcq_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
            num_action_samples=10,
        )


def test_bear_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bear_trainer, "_evaluate_sac_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        bear_trainer.train_bear(_make_bear_config(tmp_path), run_suffix="eval-helper")


def test_cql_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cql_trainer, "_evaluate_sac_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        cql_trainer.train_cql(_make_cql_config(tmp_path), run_suffix="eval-helper")


def test_edac_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(edac_trainer, "_evaluate_redq_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        edac_trainer.train_edac(_make_edac_config(tmp_path), run_suffix="eval-helper")


def test_rebrac_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rebrac_trainer, "_evaluate_td3_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        rebrac_trainer.train_rebrac(_make_rebrac_config(tmp_path), run_suffix="eval-helper")


def test_xql_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(xql_trainer, "_evaluate_iql_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        xql_trainer.train_xql(_make_xql_config(tmp_path), run_suffix="eval-helper")


def test_crr_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crr_trainer, "_evaluate_sac_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        crr_trainer.train_crr(_make_crr_config(tmp_path), run_suffix="eval-helper")


def test_marwil_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(marwil_trainer, "_evaluate_iql_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        marwil_trainer.train_marwil(_make_marwil_config(tmp_path), run_suffix="eval-helper")


def test_td3_bc_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(td3_bc_trainer, "_evaluate_td3_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        td3_bc_trainer.train_td3_bc(_make_td3_bc_config(tmp_path), run_suffix="eval-helper")


def test_rlpd_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rlpd_trainer, "_evaluate_sac_policy", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        rlpd_trainer.train_rlpd(_make_rlpd_config(tmp_path), run_suffix="eval-helper")


def test_agent57_evaluation_delegates_to_shared_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent57_trainer, "_maybe_run_r2d2_evaluation", _raise_expected_eval_support, raising=False)

    with pytest.raises(ExpectedEvaluationSupport):
        agent57_trainer.train_agent57(_make_agent57_config(tmp_path), run_suffix="eval-helper")


def test_recurrent_ppo_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recurrent_ppo_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False
    )
    monkeypatch.setattr(recurrent_ppo_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    policy = SimpleNamespace(
        initial_state=lambda batch_size, device=None: ("state", "cell"),
        act=lambda obs_tensor, state=None, deterministic=True: SimpleNamespace(
            actions=torch.tensor([0]),
            state=state,
        ),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        recurrent_ppo_trainer._evaluate_recurrent_policy(
            policy,
            _make_recurrent_ppo_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_drqn_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(drqn_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(drqn_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    q_network = SimpleNamespace(
        initial_state=lambda batch_size, device=None: ("state", "cell"),
        act=lambda obs_tensor, state=None, epsilon=0.0, deterministic=True, episode_starts=None: SimpleNamespace(
            actions=torch.tensor([0]),
            state=state,
        ),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        drqn_trainer._evaluate_drqn_policy(
            q_network,
            _make_drqn_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )


def test_r2d2_evaluation_uses_shared_evaluation_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(r2d2_trainer, "evaluate_discrete_episodes", _raise_expected_eval_support, raising=False)
    monkeypatch.setattr(r2d2_trainer, "build_env", _raise_legacy_eval_path, raising=False)

    q_network = SimpleNamespace(
        initial_state=lambda batch_size, device=None: ("state", "cell"),
        act=lambda obs_tensor, state=None, epsilon=0.0, deterministic=True, episode_starts=None: SimpleNamespace(
            actions=torch.tensor([0]),
            state=state,
        ),
    )

    with pytest.raises(ExpectedEvaluationSupport):
        r2d2_trainer._evaluate_r2d2_policy(
            q_network,
            _make_r2d2_config(tmp_path),
            device=torch.device("cpu"),
            num_episodes=1,
        )
