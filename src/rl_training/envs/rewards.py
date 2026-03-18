from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np

_DEFAULT_SUCCESS_KEYS: tuple[str, ...] = (
    "is_success",
    "success",
    "goal_achieved",
    "goal_reached",
    "task_success",
)


@dataclass(frozen=True, slots=True)
class RewardTransformConfig:
    sign: bool = False
    scale: float = 1.0
    shift: float = 0.0
    clip_min: float | None = None
    clip_max: float | None = None
    step_penalty: float = 0.0
    terminal_bonus: float = 0.0
    success_bonus: float = 0.0
    failure_penalty: float = 0.0
    success_keys: tuple[str, ...] = _DEFAULT_SUCCESS_KEYS


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


class RewardOutcomeWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        step_penalty: float,
        terminal_bonus: float,
        success_bonus: float,
        failure_penalty: float,
        success_keys: Sequence[str],
    ) -> None:
        super().__init__(env)
        self.step_penalty = float(step_penalty)
        self.terminal_bonus = float(terminal_bonus)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)
        self.success_keys = tuple(str(key) for key in success_keys)

    def step(self, action: object):
        observation, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = float(reward)
        if not math.isclose(self.step_penalty, 0.0):
            shaped_reward += self.step_penalty
        if terminated or truncated:
            if not math.isclose(self.terminal_bonus, 0.0):
                shaped_reward += self.terminal_bonus
            success = _extract_success_signal(info, self.success_keys)
            if success is True and not math.isclose(self.success_bonus, 0.0):
                shaped_reward += self.success_bonus
            elif success is False and not math.isclose(self.failure_penalty, 0.0):
                shaped_reward += self.failure_penalty
        return observation, float(shaped_reward), terminated, truncated, info


_REWARD_PRESET_REGISTRY: dict[str, RewardTransformConfig] = {
    "sign_clip": RewardTransformConfig(sign=True),
    "atari_clip": RewardTransformConfig(sign=True),
    "clip_1": RewardTransformConfig(clip_min=-1.0, clip_max=1.0),
    "sparse_goal_zero_one": RewardTransformConfig(shift=1.0, clip_min=0.0, clip_max=1.0),
    "survival_penalty": RewardTransformConfig(step_penalty=-0.01),
    "goal_success_bonus": RewardTransformConfig(success_bonus=1.0),
}


def _is_identity_reward_config(config: RewardTransformConfig) -> bool:
    return (
        not config.sign
        and math.isclose(config.scale, 1.0)
        and math.isclose(config.shift, 0.0)
        and config.clip_min is None
        and config.clip_max is None
        and math.isclose(config.step_penalty, 0.0)
        and math.isclose(config.terminal_bonus, 0.0)
        and math.isclose(config.success_bonus, 0.0)
        and math.isclose(config.failure_penalty, 0.0)
    )


def _resolve_success_keys(
    requested_value: object,
    *,
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    if requested_value is None:
        return fallback
    if isinstance(requested_value, str):
        normalized = requested_value.strip()
        if not normalized:
            raise ValueError("wrappers['reward']['success_keys'] must not be empty")
        return (normalized,)
    if not isinstance(requested_value, Sequence):
        raise TypeError("wrappers['reward']['success_keys'] must be a string or sequence of strings")

    resolved = tuple(str(item).strip() for item in requested_value)
    if not resolved or any(not item for item in resolved):
        raise ValueError("wrappers['reward']['success_keys'] must contain non-empty keys")
    return resolved


def _extract_success_signal(info: object, success_keys: Sequence[str]) -> bool | None:
    if not isinstance(info, Mapping):
        return None
    for key in success_keys:
        if key not in info:
            continue
        value = info[key]
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            return bool(value.reshape(-1)[0])
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) == 0:
                return None
            return bool(value[0])
        return bool(value)
    return None


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
    strategy = requested_payload.get("strategy")
    if strategy is not None:
        if preset is not None and str(strategy).strip().lower() != str(preset).strip().lower():
            raise ValueError("wrappers['reward'] cannot define conflicting 'preset' and 'strategy' values")
        preset = strategy
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
        step_penalty=float(requested_payload.get("step_penalty", preset_config.step_penalty)),
        terminal_bonus=float(requested_payload.get("terminal_bonus", preset_config.terminal_bonus)),
        success_bonus=float(requested_payload.get("success_bonus", preset_config.success_bonus)),
        failure_penalty=float(requested_payload.get("failure_penalty", preset_config.failure_penalty)),
        success_keys=_resolve_success_keys(
            requested_payload.get("success_keys"),
            fallback=preset_config.success_keys,
        ),
    )
    if _is_identity_reward_config(config):
        return None
    return config


def apply_reward_wrappers(env: gym.Env, reward_config: RewardTransformConfig | None) -> gym.Env:
    if reward_config is None:
        return env

    wrapped = env
    if (
        not math.isclose(reward_config.step_penalty, 0.0)
        or not math.isclose(reward_config.terminal_bonus, 0.0)
        or not math.isclose(reward_config.success_bonus, 0.0)
        or not math.isclose(reward_config.failure_penalty, 0.0)
    ):
        wrapped = RewardOutcomeWrapper(
            wrapped,
            step_penalty=reward_config.step_penalty,
            terminal_bonus=reward_config.terminal_bonus,
            success_bonus=reward_config.success_bonus,
            failure_penalty=reward_config.failure_penalty,
            success_keys=reward_config.success_keys,
        )
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
