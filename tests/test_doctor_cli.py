from axiomrl.cli import main
from axiomrl.version import __version__


def test_doctor_command_prints_environment_info(capsys) -> None:
    exit_code = main(["doctor"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"axiomrl_version={__version__}" in captured.out
    assert "python_version=" in captured.out
    assert "torch_version=" in captured.out
    assert "cuda_available=" in captured.out
    assert "gymnasium_version=" in captured.out
    assert "ale_py_version=" in captured.out
    assert "autorom_version=" in captured.out
    assert "atari_env_registration=" in captured.out
    assert "atari_roms_available=" in captured.out
