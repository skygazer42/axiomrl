import subprocess
import sys


def test_openai_es_reference_script_smoke_runs() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "examples/openai_es_pendulum_reference.py",
            "--total-timesteps",
            "100",
            "--eval-episodes",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
