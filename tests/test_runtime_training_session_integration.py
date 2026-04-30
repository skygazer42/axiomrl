from __future__ import annotations

from pathlib import Path

import pytest

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
    td3_bc_trainer,
    td3_trainer,
    tqc_trainer,
    trpo_trainer,
    xql_trainer,
)
from tests.support.runtime_foundation import (
    ExpectedSharedSession,
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
    _make_td3_bc_config,
    _make_td3_config,
    _make_tqc_config,
    _make_trpo_config,
    _make_xql_config,
    _raise_legacy_run_setup,
    _raise_shared_session,
)


def test_train_a2c_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(a2c_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(a2c_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        a2c_trainer.train_a2c(_make_a2c_config(tmp_path), run_suffix="session")


def test_train_trpo_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trpo_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(trpo_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        trpo_trainer.train_trpo(_make_trpo_config(tmp_path), run_suffix="session")


def test_train_discrete_sac_uses_shared_training_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(discrete_sac_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(discrete_sac_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        discrete_sac_trainer.train_discrete_sac(_make_discrete_sac_config(tmp_path), run_suffix="session")


def test_train_ddpg_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ddpg_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(ddpg_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        ddpg_trainer.train_ddpg(_make_ddpg_config(tmp_path), run_suffix="session")


def test_train_td3_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td3_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(td3_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        td3_trainer.train_td3(_make_td3_config(tmp_path), run_suffix="session")


def test_train_redq_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(redq_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(redq_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        redq_trainer.train_redq(_make_redq_config(tmp_path), run_suffix="session")


def test_train_crossq_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crossq_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(crossq_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        crossq_trainer.train_crossq(_make_crossq_config(tmp_path), run_suffix="session")


def test_train_tqc_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tqc_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(tqc_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        tqc_trainer.train_tqc(_make_tqc_config(tmp_path), run_suffix="session")


def test_train_d4pg_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(d4pg_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(d4pg_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        d4pg_trainer.train_d4pg(_make_d4pg_config(tmp_path), run_suffix="session")


def test_train_naf_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(naf_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(naf_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        naf_trainer.train_naf(_make_naf_config(tmp_path), run_suffix="session")


def test_train_drq_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drq_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(drq_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        drq_trainer.train_drq(_make_drq_config(tmp_path), run_suffix="session")


def test_train_drqv2_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drqv2_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(drqv2_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        drqv2_trainer.train_drqv2(_make_drqv2_config(tmp_path), run_suffix="session")


def test_train_curl_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(curl_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(curl_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        curl_trainer.train_curl(_make_curl_config(tmp_path), run_suffix="session")


def test_train_impala_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(impala_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(impala_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        impala_trainer.train_impala(_make_impala_config(tmp_path), run_suffix="session")


def test_train_dreamer_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dreamer_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(dreamer_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        dreamer_trainer.train_dreamer(_make_dreamer_config(tmp_path), run_suffix="session")


def test_train_apex_dqn_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apex_dqn_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(apex_dqn_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        apex_dqn_trainer.train_apex_dqn(_make_apex_dqn_config(tmp_path), run_suffix="session")


def test_train_iql_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(iql_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(iql_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        iql_trainer.train_iql(_make_iql_config(tmp_path), run_suffix="session")


def test_train_gail_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gail_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(gail_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        gail_trainer.train_gail(_make_gail_config(tmp_path), run_suffix="session")


def test_train_mbpo_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mbpo_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(mbpo_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        mbpo_trainer.train_mbpo(_make_mbpo_config(tmp_path), run_suffix="session")


def test_train_mopo_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mopo_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(mopo_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        mopo_trainer.train_mopo(_make_mopo_config(tmp_path), run_suffix="session")


def test_train_muzero_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(muzero_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(muzero_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        muzero_trainer.train_muzero(_make_muzero_config(tmp_path), run_suffix="session")


def test_train_efficientzero_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(efficientzero_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(efficientzero_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        efficientzero_trainer.train_efficientzero(_make_efficientzero_config(tmp_path), run_suffix="session")


def test_train_decision_transformer_uses_shared_training_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(decision_transformer_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(decision_transformer_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        decision_transformer_trainer.train_decision_transformer(
            _make_decision_transformer_config(tmp_path),
            run_suffix="session",
        )


def test_train_her_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(her_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(her_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        her_trainer.train_her(_make_her_config(tmp_path), run_suffix="session")


def test_train_appo_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(appo_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(appo_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        appo_trainer.train_appo(_make_appo_config(tmp_path), run_suffix="session")


def test_train_ppg_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ppg_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(ppg_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        ppg_trainer.train_ppg(_make_ppg_config(tmp_path), run_suffix="session")


def test_train_ars_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ars_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(ars_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        ars_trainer.train_ars(_make_ars_config(tmp_path), run_suffix="session")


def test_train_openai_es_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openai_es_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(openai_es_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        openai_es_trainer.train_openai_es(_make_openai_es_config(tmp_path), run_suffix="session")


def test_train_pets_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pets_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(pets_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        pets_trainer.train_pets(_make_pets_config(tmp_path), run_suffix="session")


def test_train_awr_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(awr_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(awr_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        awr_trainer.train_awr(_make_awr_config(tmp_path), run_suffix="session")


def test_train_awac_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(awac_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(awac_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        awac_trainer.train_awac(_make_awac_config(tmp_path), run_suffix="session")


def test_train_cal_ql_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cal_ql_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(cal_ql_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        cal_ql_trainer.train_cal_ql(_make_cal_ql_config(tmp_path), run_suffix="session")


def test_train_bc_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bc_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(bc_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        bc_trainer.train_bc(_make_bc_config(tmp_path), run_suffix="session")


def test_train_bcq_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bcq_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(bcq_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        bcq_trainer.train_bcq(_make_bcq_config(tmp_path), run_suffix="session")


def test_train_bear_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bear_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(bear_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        bear_trainer.train_bear(_make_bear_config(tmp_path), run_suffix="session")


def test_train_cql_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cql_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(cql_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        cql_trainer.train_cql(_make_cql_config(tmp_path), run_suffix="session")


def test_train_edac_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(edac_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(edac_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        edac_trainer.train_edac(_make_edac_config(tmp_path), run_suffix="session")


def test_train_rebrac_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rebrac_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(rebrac_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        rebrac_trainer.train_rebrac(_make_rebrac_config(tmp_path), run_suffix="session")


def test_train_xql_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xql_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(xql_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        xql_trainer.train_xql(_make_xql_config(tmp_path), run_suffix="session")


def test_train_crr_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crr_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(crr_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        crr_trainer.train_crr(_make_crr_config(tmp_path), run_suffix="session")


def test_train_marwil_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(marwil_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(marwil_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        marwil_trainer.train_marwil(_make_marwil_config(tmp_path), run_suffix="session")


def test_train_td3_bc_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td3_bc_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(td3_bc_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        td3_bc_trainer.train_td3_bc(_make_td3_bc_config(tmp_path), run_suffix="session")


def test_train_rlpd_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rlpd_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(rlpd_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        rlpd_trainer.train_rlpd(_make_rlpd_config(tmp_path), run_suffix="session")


def test_train_agent57_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent57_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(agent57_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        agent57_trainer.train_agent57(_make_agent57_config(tmp_path), run_suffix="session")


def test_train_recurrent_ppo_uses_shared_training_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recurrent_ppo_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(recurrent_ppo_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        recurrent_ppo_trainer.train_recurrent_ppo(_make_recurrent_ppo_config(tmp_path), run_suffix="session")


def test_train_drqn_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drqn_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(drqn_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        drqn_trainer.train_drqn(_make_drqn_config(tmp_path), run_suffix="session")


def test_train_r2d2_uses_shared_training_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(r2d2_trainer, "create_training_session", _raise_shared_session, raising=False)
    monkeypatch.setattr(r2d2_trainer, "create_training_run", _raise_legacy_run_setup, raising=False)

    with pytest.raises(ExpectedSharedSession):
        r2d2_trainer.train_r2d2(_make_r2d2_config(tmp_path), run_suffix="session")
