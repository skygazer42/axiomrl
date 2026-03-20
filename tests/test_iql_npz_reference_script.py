import subprocess
import sys


def test_iql_npz_reference_script_smoke_runs() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "examples/iql_pendulum_npz_reference.py",
            "--dataset-size",
            "128",
            "--total-timesteps",
            "96",
            "--eval-episodes",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "dataset_path=" in proc.stdout
    assert "checkpoint_path=" in proc.stdout
    assert "inference_action=" in proc.stdout
