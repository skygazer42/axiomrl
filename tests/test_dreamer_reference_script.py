import subprocess
import sys


def test_dreamer_reference_script_smoke_runs() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "examples/dreamer_cartpole_pixels_reference.py",
            "--total-timesteps",
            "32",
            "--eval-episodes",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
