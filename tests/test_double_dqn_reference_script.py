import subprocess
import sys


def test_double_dqn_reference_script_smoke_runs() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "examples/double_dqn_cartpole_reference.py",
            "--total-timesteps",
            "128",
            "--eval-episodes",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
