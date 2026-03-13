from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np


@dataclass(frozen=True, slots=True)
class RewardTransformConfig:
    sign: bool = False
    scale: float = 1.0
    shift: float = 0.0
    clip_min: float | None = None
    clip_max: float | None = None


class RewardSignWrapper(gym.RewardWrapper):
    def reward(self, reward: float) -> float:
        return float(np.sign(float(reward)))


class RewardScaleWrapper(gym.RewardWrapper):
    def __init__(self, env: gym.Env, *, scale: float) -> None:
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, reward: float) -> float:
        return float(reward) * self.scale


class RewardShiftWrapper(gym.RewardWrapper):
    def __init__(self, env: gym.Env, *, shift: float) -> None:
        super().__init__(env)
        self.shift = float(shift)

    def reward(self, reward: float) -> float:
        return float(reward) + self.shift


class RewardClipWrapper(gym.RewardWrapper):
    def __init__(self, env: gym.Env, *, clip_min: float | None, clip_max: float | None) -> None:
        super().__init__(env)
        self.clip_min = clip_min
        self.clip_max = clip_max

    def reward(self, reward: float) -> float:
        lower = float("-inf") if self.clip_min is None else float(self.clip_min)
        upper = float("inf") if self.clip_max is None else float(self.clip_max)
        return float(np.clip(float(reward), lower, upper))


_REWARD_PRESET_REGISTRY: dict[str, RewardTransformConfig] = {
    "sign_clip": RewardTransformConfig(sign=True),
    "clip_1": RewardTransformConfig(clip_min=-1.0, clip_max=1.0),
    "sparse_goal_zero_one": RewardTransformConfig(shift=1.0, clip_min=0.0, clip_max=1.0),
}


def _is_identity_reward_config(config: RewardTransformConfig) -> bool:
    return (
        not config.sign
        and math.isclose(config.scale, 1.0)
        and math.isclose(config.shift, 0.0)
        and config.clip_min is None
        and config.clip_max is None
    )


def resolve_reward_preset(name: str) -> RewardTransformConfig:
    normalized_name = str(name).strip().lower()
    try:
        return _REWARD_PRESET_REGISTRY[normalized_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_REWARD_PRESET_REGISTRY))
        raise ValueError(f"unsupported reward preset {name!r}; expected one of: {supported}") from exc


def resolve_reward_wrapper_config(wrapper_kwargs: Mapping[str, object]) -> RewardTransformConfig | None:
    requested = wrapper_kwargs.get("reward")
    if requested in (None, False):
        return None
    if requested is True:
        requested_payload: Mapping[str, object] = {}
    else:
        if not isinstance(requested, Mapping):
            raise TypeError(f"expected wrappers['reward'] to be a mapping or bool, got {type(requested)!r}")
        requested_payload = requested

    clip = requested_payload.get("clip")
    clip_min = requested_payload.get("clip_min")
    clip_max = requested_payload.get("clip_max")
    preset = requested_payload.get("preset")
    preset_config = RewardTransformConfig()
    if preset is not None:
        preset_config = resolve_reward_preset(str(preset))

    if clip is not None:
        if not isinstance(clip, Sequence) or len(clip) != 2:
            raise TypeError("wrappers['reward']['clip'] must be a 2-item sequence")
        clip_min = clip[0]
        clip_max = clip[1]

    config = RewardTransformConfig(
        sign=bool(requested_payload.get("sign", preset_config.sign)),
        scale=float(requested_payload.get("scale", preset_config.scale)),
        shift=float(requested_payload.get("shift", preset_config.shift)),
        clip_min=float(clip_min) if clip_min is not None else preset_config.clip_min,
        clip_max=float(clip_max) if clip_max is not None else preset_config.clip_max,
    )
    if _is_identity_reward_config(config):
        return None
    return config


def apply_reward_wrappers(env: gym.Env, reward_config: RewardTransformConfig | None) -> gym.Env:
    if reward_config is None:
        return env

    wrapped = env
    if reward_config.sign:
        wrapped = RewardSignWrapper(wrapped)
    if not math.isclose(reward_config.scale, 1.0):
        wrapped = RewardScaleWrapper(wrapped, scale=reward_config.scale)
    if not math.isclose(reward_config.shift, 0.0):
        wrapped = RewardShiftWrapper(wrapped, shift=reward_config.shift)
    if reward_config.clip_min is not None or reward_config.clip_max is not None:
        wrapped = RewardClipWrapper(
            wrapped,
            clip_min=reward_config.clip_min,
            clip_max=reward_config.clip_max,
        )
    return wrapped
