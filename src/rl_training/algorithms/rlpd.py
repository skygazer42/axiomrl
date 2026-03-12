from __future__ import annotations

import torch

from rl_training.algorithms.sac import SAC, sac_loss


def rlpd_loss(batch: dict[str, torch.Tensor | float]) -> dict[str, float]:
    return sac_loss(batch)


class RLPD(SAC):
    """Narrow v1 RLPD learner.

    The package-specific distinction lives in the trainer: prior offline data,
    offline pretraining, and mixed offline/online updates. The learner itself
    intentionally reuses the current SAC update in this phase.
    """

