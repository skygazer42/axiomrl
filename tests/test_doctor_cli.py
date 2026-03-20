from rl_training.cli import main


def test_doctor_command_prints_environment_info(capsys) -> None:
    exit_code = main(["doctor"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "python_version=" in captured.out
    assert "torch_version=" in captured.out
    assert "cuda_available=" in captured.out
    assert "gymnasium_version=" in captured.out

