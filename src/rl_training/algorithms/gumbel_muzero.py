from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch

from rl_training.algorithms.muzero import MuZero, run_muzero_mcts


def _sample_gumbel(shape: tuple[int, ...], *, scale: float) -> np.ndarray:
    uniform = np.random.uniform(low=1e-8, high=1.0 - 1e-8, size=shape).astype(np.float32)
    gumbels = -np.log(-np.log(uniform))
    return float(scale) * gumbels


class GumbelMuZero(MuZero):
    def __init__(
        self,
        *,
        gumbel_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if float(gumbel_scale) <= 0.0:
            raise ValueError(f"gumbel_scale must be > 0, got {gumbel_scale}")
        self.gumbel_scale = float(gumbel_scale)

    def plan(
        self,
        obs: object,
        *,
        temperature: float,
        add_root_noise: bool,
        deterministic: bool,
        root_exploration_fraction: float | None = None,
        num_simulations: int | None = None,
    ) -> tuple[int, np.ndarray, float]:
        mcts_config = self.mcts_config
        if root_exploration_fraction is not None or num_simulations is not None:
            mcts_config = replace(
                mcts_config,
                root_exploration_fraction=(
                    float(root_exploration_fraction)
                    if root_exploration_fraction is not None
                    else mcts_config.root_exploration_fraction
                ),
                num_simulations=int(num_simulations) if num_simulations is not None else mcts_config.num_simulations,
            )
        probs, root_value = run_muzero_mcts(
            model=self.model,
            obs=obs,
            mcts=mcts_config,
            gamma=self.gamma,
            add_root_noise=add_root_noise,
        )
        if deterministic or temperature <= 1e-8:
            action = int(probs.argmax())
        else:
            logits = np.log(np.clip(probs, 1e-8, 1.0)) / float(temperature)
            action = int((logits + _sample_gumbel(tuple(logits.shape), scale=self.gumbel_scale)).argmax())
        return action, probs.astype(np.float32), float(root_value)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["gumbel_scale"] = self.gumbel_scale
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.gumbel_scale = float(state_dict.get("gumbel_scale", self.gumbel_scale))
