from __future__ import annotations

from axiomrl.algorithms.ddpg import DDPG, ddpg_loss


def her_loss(batch: dict) -> dict[str, float]:
    return ddpg_loss(batch)


class HER(DDPG):
    pass
