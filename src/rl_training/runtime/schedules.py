from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ScheduleSpec:
    kind: str = "constant"
    start: float = 1.0
    end: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {"constant", "linear", "cosine"}:
            raise ValueError(f"unsupported schedule kind: {self.kind!r}")
        if self.start < 0.0:
            raise ValueError(f"schedule start must be >= 0, got {self.start}")
        if self.end < 0.0:
            raise ValueError(f"schedule end must be >= 0, got {self.end}")


def resolve_schedule_spec(payload: object | None) -> ScheduleSpec:
    if payload in (None, False):
        return ScheduleSpec()
    if isinstance(payload, ScheduleSpec):
        return payload
    if isinstance(payload, str):
        kind = payload.strip().lower()
        default_end = 1.0 if kind == "constant" else 0.0
        return ScheduleSpec(kind=kind, start=1.0, end=default_end)
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected schedule payload to be a mapping or string, got {type(payload)!r}")

    kind = str(payload.get("type", payload.get("kind", "constant"))).strip().lower()
    start = float(payload.get("start", payload.get("value", 1.0)))
    default_end = start if kind == "constant" else 0.0
    end = float(payload.get("end", default_end))
    return ScheduleSpec(kind=kind, start=start, end=end)


def resolve_schedule_value(
    payload: ScheduleSpec | Mapping[str, object] | str | None,
    *,
    step: int,
    total_steps: int,
    warmup_steps: int = 0,
) -> float:
    if int(total_steps) < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if int(step) < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if int(warmup_steps) < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

    spec = resolve_schedule_spec(payload)
    clamped_step = min(int(step), int(total_steps) - 1)
    progress = 1.0 if total_steps <= 1 else clamped_step / float(total_steps - 1)

    if spec.kind == "constant":
        value = spec.start
    elif spec.kind == "linear":
        value = spec.start + progress * (spec.end - spec.start)
    else:
        cosine_mix = 0.5 * (1.0 + math.cos(math.pi * progress))
        value = spec.end + (spec.start - spec.end) * cosine_mix

    if warmup_steps > 0:
        warmup_scale = min((clamped_step + 1) / float(warmup_steps), 1.0)
        value *= warmup_scale
    return float(value)


def iter_optimizers(owner: object) -> tuple[torch.optim.Optimizer, ...]:
    discovered: list[torch.optim.Optimizer] = []
    seen_ids: set[int] = set()

    def _collect(candidate: object) -> None:
        if isinstance(candidate, torch.optim.Optimizer):
            candidate_id = id(candidate)
            if candidate_id not in seen_ids:
                seen_ids.add(candidate_id)
                discovered.append(candidate)
            return
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            for item in candidate:
                _collect(item)

    try:
        values = vars(owner).values()
    except TypeError:
        values = ()
    for value in values:
        _collect(value)
    return tuple(discovered)


def apply_learning_rate_scale(owner: object | Sequence[torch.optim.Optimizer], *, scale: float) -> float:
    if float(scale) < 0.0:
        raise ValueError(f"scale must be >= 0, got {scale}")

    if isinstance(owner, Sequence):
        optimizers = tuple(owner)
    else:
        optimizers = iter_optimizers(owner)
    if not optimizers:
        return 0.0

    learning_rates: list[float] = []
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            base_lr = float(param_group.setdefault("initial_lr", param_group["lr"]))
            param_group["lr"] = base_lr * float(scale)
            learning_rates.append(float(param_group["lr"]))
    if not learning_rates:
        return 0.0
    return float(sum(learning_rates) / len(learning_rates))
