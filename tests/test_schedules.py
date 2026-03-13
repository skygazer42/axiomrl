import math

import torch

from rl_training.runtime.schedules import ScheduleSpec, apply_learning_rate_scale, resolve_schedule_spec, resolve_schedule_value


def test_resolve_schedule_spec_supports_string_and_mapping_payloads() -> None:
    constant = resolve_schedule_spec(None)
    linear = resolve_schedule_spec("linear")
    cosine = resolve_schedule_spec({"type": "cosine", "start": 1.0, "end": 0.2})

    assert constant == ScheduleSpec(kind="constant", start=1.0, end=1.0)
    assert linear == ScheduleSpec(kind="linear", start=1.0, end=0.0)
    assert cosine == ScheduleSpec(kind="cosine", start=1.0, end=0.2)


def test_resolve_schedule_value_supports_constant_linear_and_cosine_with_warmup() -> None:
    assert math.isclose(resolve_schedule_value("constant", step=0, total_steps=4), 1.0)
    assert math.isclose(resolve_schedule_value("linear", step=3, total_steps=4), 0.0)
    assert math.isclose(
        resolve_schedule_value({"type": "cosine", "start": 1.0, "end": 0.2}, step=3, total_steps=4),
        0.2,
    )
    assert math.isclose(
        resolve_schedule_value("constant", step=0, total_steps=4, warmup_steps=4),
        0.25,
    )


def test_apply_learning_rate_scale_uses_initial_learning_rate() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0)

    scaled_lr = apply_learning_rate_scale((optimizer,), scale=0.5)
    restored_lr = apply_learning_rate_scale((optimizer,), scale=1.0)

    assert math.isclose(scaled_lr, 1.5e-4, rel_tol=1e-6)
    assert math.isclose(restored_lr, 3e-4, rel_tol=1e-6)
