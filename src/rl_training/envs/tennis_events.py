from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np


@dataclass(frozen=True, slots=True)
class TennisEventConfig:
    rally_survival_bonus: float = 0.0
    net_cross_bonus: float = 0.0
    successful_return_bonus: float = 0.0
    failure_penalty: float = 0.0
    deep_landing_bonus: float = 0.0
    wide_landing_bonus: float = 0.0
    agent_side: str = "left"
    motion_threshold: float = 8.0

    def __post_init__(self) -> None:
        if self.agent_side not in {"left", "right"}:
            raise ValueError(f"agent_side must be 'left' or 'right', got {self.agent_side!r}")


def resolve_tennis_event_wrapper_config(wrapper_kwargs: Mapping[str, object]) -> TennisEventConfig | None:
    requested = wrapper_kwargs.get("tennis_events")
    if requested in (None, False):
        return None
    if requested is True:
        payload: Mapping[str, object] = {}
    else:
        if not isinstance(requested, Mapping):
            raise TypeError(f"expected wrappers['tennis_events'] to be a mapping or bool, got {type(requested)!r}")
        payload = requested

    config = TennisEventConfig(
        rally_survival_bonus=float(payload.get("rally_survival_bonus", 0.0)),
        net_cross_bonus=float(payload.get("net_cross_bonus", 0.0)),
        successful_return_bonus=float(payload.get("successful_return_bonus", 0.0)),
        failure_penalty=float(payload.get("failure_penalty", 0.0)),
        deep_landing_bonus=float(payload.get("deep_landing_bonus", 0.0)),
        wide_landing_bonus=float(payload.get("wide_landing_bonus", 0.0)),
        agent_side=str(payload.get("agent_side", "left")).strip().lower(),
        motion_threshold=float(payload.get("motion_threshold", 8.0)),
    )
    if (
        math.isclose(config.rally_survival_bonus, 0.0)
        and math.isclose(config.net_cross_bonus, 0.0)
        and math.isclose(config.successful_return_bonus, 0.0)
        and math.isclose(config.failure_penalty, 0.0)
        and math.isclose(config.deep_landing_bonus, 0.0)
        and math.isclose(config.wide_landing_bonus, 0.0)
    ):
        return None
    return config


def _extract_frame(observation: object) -> np.ndarray:
    frame = np.asarray(observation, dtype=np.float32)
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3:
        if frame.shape[0] <= 8 and frame.shape[1] > 8 and frame.shape[2] > 8:
            return frame[-1]
        if frame.shape[-1] in (3, 4):
            return frame[..., :3].mean(axis=-1)
        if frame.shape[-1] <= 8:
            return frame[..., -1]
        return frame[0]
    if frame.ndim == 4:
        return _extract_frame(frame[-1])
    raise ValueError(f"unsupported Tennis observation shape: {frame.shape!r}")


class TennisEventRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, config: TennisEventConfig) -> None:
        super().__init__(env)
        self.config = config
        self._previous_frame: np.ndarray | None = None
        self._previous_side: str | None = None
        self._seen_ball = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._previous_frame = _extract_frame(observation)
        initial_ball_position = self._detect_ball_position(self._previous_frame)
        if initial_ball_position is None:
            self._previous_side = None
            self._seen_ball = False
        else:
            initial_ball_x, _ = initial_ball_position
            self._previous_side = self._side_for_x(initial_ball_x, frame_width=self._previous_frame.shape[1])
            self._seen_ball = True
        return observation, info

    def step(self, action: object):
        observation, reward, terminated, truncated, info = self.env.step(action)
        current_frame = _extract_frame(observation)
        shaped_reward = float(reward)

        ball_position = self._detect_ball_position(current_frame)
        if ball_position is not None:
            ball_x, ball_y = ball_position
            current_side = self._side_for_x(ball_x, frame_width=current_frame.shape[1])
            if self._seen_ball and not math.isclose(self.config.rally_survival_bonus, 0.0):
                shaped_reward += self.config.rally_survival_bonus
            if self._previous_side is not None and current_side != self._previous_side:
                shaped_reward += self.config.net_cross_bonus
                if self._previous_side == self.config.agent_side and current_side != self.config.agent_side:
                    shaped_reward += self.config.successful_return_bonus
                    shaped_reward += self._offensive_bonus(
                        ball_x=ball_x,
                        ball_y=ball_y,
                        frame_width=current_frame.shape[1],
                        frame_height=current_frame.shape[0],
                    )
            self._seen_ball = True
            self._previous_side = current_side

        if (terminated or truncated) and reward < 0.0 and not math.isclose(self.config.failure_penalty, 0.0):
            shaped_reward += self.config.failure_penalty

        self._previous_frame = current_frame
        return observation, float(shaped_reward), terminated, truncated, info

    def _detect_ball_position(self, current_frame: np.ndarray) -> tuple[float, float] | None:
        if self._previous_frame is None:
            diff = current_frame
        else:
            diff = np.abs(current_frame - self._previous_frame)

        coords = np.argwhere((diff >= self.config.motion_threshold) & (current_frame >= self.config.motion_threshold))
        if coords.size == 0:
            coords = np.argwhere(current_frame >= max(200.0, float(current_frame.max())))
        if coords.size == 0:
            return None
        return float(np.median(coords[:, 1])), float(np.median(coords[:, 0]))

    def _offensive_bonus(
        self,
        *,
        ball_x: float,
        ball_y: float,
        frame_width: int,
        frame_height: int,
    ) -> float:
        bonus = 0.0
        net_x = frame_width / 2.0
        if self.config.agent_side == "left":
            opponent_depth = max(0.0, ball_x - net_x)
        else:
            opponent_depth = max(0.0, net_x - ball_x)

        if opponent_depth >= frame_width * 0.25:
            bonus += self.config.deep_landing_bonus

        center_y = frame_height / 2.0
        if abs(ball_y - center_y) >= frame_height * 0.3:
            bonus += self.config.wide_landing_bonus

        return float(bonus)

    @staticmethod
    def _side_for_x(ball_x: float, *, frame_width: int) -> str:
        return "left" if ball_x < (frame_width / 2.0) else "right"


def apply_tennis_event_wrapper(env: gym.Env, tennis_event_config: TennisEventConfig | None) -> gym.Env:
    if tennis_event_config is None:
        return env
    return TennisEventRewardWrapper(env, tennis_event_config)
