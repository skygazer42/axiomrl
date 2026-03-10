import subprocess
import sys


def test_ddpg_reference_script_smoke_runs() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "examples/ddpg_pendulum_reference.py",
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
