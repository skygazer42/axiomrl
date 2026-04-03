from rl_training.runtime.trainer import TrainerState
from rl_training.tuning.optuna_backend import OptunaPruningCallback


class _FakeTrial:
    def __init__(self, *, should_prune: bool) -> None:
        self.should_prune_value = should_prune
        self.report_calls: list[tuple[float, int]] = []

    def report(self, value: float, step: int) -> None:
        self.report_calls.append((value, step))

    def should_prune(self) -> bool:
        return self.should_prune_value


class _FakeTrialPruned(RuntimeError):
    pass


def test_optuna_pruning_callback_reports_metric_and_requests_stop(tmp_path) -> None:
    trial = _FakeTrial(should_prune=True)
    callback = OptunaPruningCallback(
        trial=trial,
        metric="eval_return_mean",
        prune_exception=_FakeTrialPruned,
    )
    trainer = TrainerState(algorithm="ppo", run_dir=tmp_path, global_step=64)

    try:
        callback.on_eval_end(trainer, {"eval_return_mean": 12.5})
    except _FakeTrialPruned:
        pass
    else:  # pragma: no cover - sanity guard
        raise AssertionError("expected pruning callback to raise the configured prune exception")

    assert trial.report_calls == [(12.5, 64)]
    assert trainer.should_stop is True
    assert trainer.stop_reason is not None
    assert "optuna pruning" in trainer.stop_reason
