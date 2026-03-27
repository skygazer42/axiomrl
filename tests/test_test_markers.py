from tests.support.markers import classify_test_path


def test_classify_unit_test_path_by_default() -> None:
    assert classify_test_path("tests/test_release_contracts.py") == {"unit"}


def test_classify_integration_test_path() -> None:
    assert classify_test_path("tests/test_real_end_to_end_workflows.py") == {
        "integration",
        "slow",
    }


def test_classify_smoke_test_path() -> None:
    assert classify_test_path("tests/test_awac_trainer_smoke.py") == {"smoke"}


def test_classify_reference_script_path_as_smoke() -> None:
    assert classify_test_path("tests/test_openai_es_reference_script.py") == {"smoke"}


def test_classify_runtime_training_session_path_as_integration_and_slow() -> None:
    assert classify_test_path("tests/test_runtime_training_session_integration.py") == {
        "integration",
        "slow",
    }


def test_classify_runtime_evaluation_support_path_as_integration_and_slow() -> None:
    assert classify_test_path("tests/test_runtime_evaluation_support_integration.py") == {
        "integration",
        "slow",
    }


def test_classify_split_checkpoint_workflow_path_as_integration() -> None:
    assert classify_test_path("tests/test_checkpoint_resume.py") == {"integration"}
