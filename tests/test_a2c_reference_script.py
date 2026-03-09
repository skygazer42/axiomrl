import subprocess
import sys


def test_a2c_reference_script_smoke_runs() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "examples/a2c_cartpole_reference.py",
            "--total-timesteps",
            "64",
            "--eval-episodes",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
