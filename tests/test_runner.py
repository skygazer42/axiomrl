from pathlib import Path

from rl_training.runtime.runner import FunctionRunner
from rl_training.runtime.trainer import TrainResult


def test_function_runner_invokes_run_fn(tmp_path: Path) -> None:
    expected = TrainResult(
        run_dir=tmp_path,
        checkpoint_path=None,
        metrics={"global_step": 4.0},
    )
    runner = FunctionRunner(run_fn=lambda: expected)

    result = runner.run()

    assert result is expected
