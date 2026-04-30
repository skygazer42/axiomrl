from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(slots=True)
class CheckpointState:
    algorithm_state: dict[str, Any]
    buffer_state: dict[str, Any] | None
    trainer_state: dict[str, Any]
    config: dict[str, Any]
    metadata: dict[str, Any]


def save_checkpoint(path: Path, state: CheckpointState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(asdict(state), path)


def load_checkpoint(path: Path) -> CheckpointState:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return CheckpointState(**payload)
