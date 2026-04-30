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
    point_win_bonus: float = 0.0
    point_loss_penalty: float = 0.0
    deep_landing_bonus: float = 0.0
    wide_landing_bonus: float = 0.0
    attack_window_steps: int = 0
    attack_conversion_bonus: float = 0.0
    failed_attack_penalty: float = 0.0
    agent_side: str = "left"
    motion_threshold: float = 8.0
    min_cross_delta_x_px: float = 6.0
    cross_cooldown_steps: int = 2
    max_step_shaping_abs: float = 0.25
    emit_info_metrics: bool = True

    def __post_init__(self) -> None:
        if self.agent_side not in {"left", "right"}:
            raise ValueError(f"agent_side must be 'left' or 'right', got {self.agent_side!r}")
        if self.motion_threshold < 0.0:
            raise ValueError(f"motion_threshold must be >= 0, got {self.motion_threshold!r}")
        if self.min_cross_delta_x_px < 0.0:
            raise ValueError(f"min_cross_delta_x_px must be >= 0, got {self.min_cross_delta_x_px!r}")
        if self.cross_cooldown_steps < 0:
            raise ValueError(f"cross_cooldown_steps must be >= 0, got {self.cross_cooldown_steps!r}")
        if self.attack_window_steps < 0:
            raise ValueError(f"attack_window_steps must be >= 0, got {self.attack_window_steps!r}")
        if self.max_step_shaping_abs <= 0.0:
            raise ValueError(f"max_step_shaping_abs must be > 0, got {self.max_step_shaping_abs!r}")
        if self.point_win_bonus < 0.0:
            raise ValueError(f"point_win_bonus must be >= 0, got {self.point_win_bonus!r}")
        if self.point_loss_penalty < 0.0:
            raise ValueError(f"point_loss_penalty must be >= 0, got {self.point_loss_penalty!r}")
        if self.attack_conversion_bonus < 0.0:
            raise ValueError(f"attack_conversion_bonus must be >= 0, got {self.attack_conversion_bonus!r}")
        if self.failed_attack_penalty < 0.0:
            raise ValueError(f"failed_attack_penalty must be >= 0, got {self.failed_attack_penalty!r}")


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
        point_win_bonus=float(payload.get("point_win_bonus", 0.0)),
        point_loss_penalty=float(payload.get("point_loss_penalty", 0.0)),
        deep_landing_bonus=float(payload.get("deep_landing_bonus", 0.0)),
        wide_landing_bonus=float(payload.get("wide_landing_bonus", 0.0)),
        attack_window_steps=int(payload.get("attack_window_steps", 0)),
        attack_conversion_bonus=float(payload.get("attack_conversion_bonus", 0.0)),
        failed_attack_penalty=float(payload.get("failed_attack_penalty", 0.0)),
        agent_side=str(payload.get("agent_side", "left")).strip().lower(),
        motion_threshold=float(payload.get("motion_threshold", 8.0)),
        min_cross_delta_x_px=float(payload.get("min_cross_delta_x_px", 6.0)),
        cross_cooldown_steps=int(payload.get("cross_cooldown_steps", 2)),
        max_step_shaping_abs=float(payload.get("max_step_shaping_abs", 0.25)),
        emit_info_metrics=bool(payload.get("emit_info_metrics", True)),
    )
    if (
        math.isclose(config.rally_survival_bonus, 0.0)
        and math.isclose(config.net_cross_bonus, 0.0)
        and math.isclose(config.successful_return_bonus, 0.0)
        and math.isclose(config.failure_penalty, 0.0)
        and math.isclose(config.point_win_bonus, 0.0)
        and math.isclose(config.point_loss_penalty, 0.0)
        and math.isclose(config.deep_landing_bonus, 0.0)
        and math.isclose(config.wide_landing_bonus, 0.0)
        and math.isclose(config.attack_conversion_bonus, 0.0)
        and math.isclose(config.failed_attack_penalty, 0.0)
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
        self._previous_ball_x: float | None = None
        self._seen_ball = False
        self._cross_cooldown_remaining = 0
        self._attack_window_remaining = 0
        self._episode_step_count = 0
        self._ball_detected_steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._previous_frame = _extract_frame(observation)
        self._cross_cooldown_remaining = 0
        self._attack_window_remaining = 0
        self._episode_step_count = 0
        self._ball_detected_steps = 0
        initial_ball_position = self._detect_ball_position(self._previous_frame)
        if initial_ball_position is None:
            self._previous_side = None
            self._previous_ball_x = None
            self._seen_ball = False
        else:
            initial_ball_x, _ = initial_ball_position
            self._previous_side = self._side_for_x(initial_ball_x, frame_width=self._previous_frame.shape[1])
            self._previous_ball_x = float(initial_ball_x)
            self._seen_ball = True
        return observation, info

    def step(self, action: object):
        observation, reward, terminated, truncated, info = self.env.step(action)
        current_frame = _extract_frame(observation)
        base_reward = float(reward)
        self._episode_step_count += 1
        if self._cross_cooldown_remaining > 0:
            self._cross_cooldown_remaining -= 1

        rally_bonus = 0.0
        net_cross_bonus = 0.0
        return_bonus = 0.0
        offense_bonus = 0.0
        failure_penalty_applied = 0.0
        point_outcome_bonus = 0.0
        attack_conversion_bonus = 0.0
        attack_failure_penalty_applied = 0.0
        attack_window_started = 0.0
        attack_triggered = False

        ball_position = self._detect_ball_position(current_frame)
        if ball_position is not None:
            self._ball_detected_steps += 1
            ball_x, ball_y = ball_position
            current_side = self._side_for_x(ball_x, frame_width=current_frame.shape[1])
            if self._seen_ball and not math.isclose(self.config.rally_survival_bonus, 0.0):
                rally_bonus = self.config.rally_survival_bonus
            can_trigger_cross = (
                self._seen_ball
                and self._previous_side is not None
                and self._previous_ball_x is not None
                and current_side != self._previous_side
                and abs(float(ball_x) - float(self._previous_ball_x)) >= self.config.min_cross_delta_x_px
                and self._cross_cooldown_remaining == 0
            )
            if can_trigger_cross:
                net_cross_bonus = self.config.net_cross_bonus
                if self._previous_side == self.config.agent_side and current_side != self.config.agent_side:
                    return_bonus = self.config.successful_return_bonus
                    offense_bonus, attack_triggered = self._offensive_bonus(
                        ball_x=ball_x,
                        ball_y=ball_y,
                        frame_width=current_frame.shape[1],
                        frame_height=current_frame.shape[0],
                    )
                    if attack_triggered and self.config.attack_window_steps > 0:
                        self._attack_window_remaining = self.config.attack_window_steps
                        attack_window_started = float(self._attack_window_remaining)
                self._cross_cooldown_remaining = self.config.cross_cooldown_steps
            self._seen_ball = True
            self._previous_side = current_side
            self._previous_ball_x = float(ball_x)
        else:
            # Keep events tied to contiguous ball tracking windows only.
            self._seen_ball = False
            self._previous_ball_x = None

        if (terminated or truncated) and reward < 0.0 and not math.isclose(self.config.failure_penalty, 0.0):
            failure_penalty_applied = self.config.failure_penalty

        if base_reward > 0.0 and not math.isclose(self.config.point_win_bonus, 0.0):
            point_outcome_bonus = self.config.point_win_bonus
        elif base_reward < 0.0 and not math.isclose(self.config.point_loss_penalty, 0.0):
            point_outcome_bonus = -self.config.point_loss_penalty

        if self._attack_window_remaining > 0:
            if base_reward > 0.0 and not math.isclose(self.config.attack_conversion_bonus, 0.0):
                attack_conversion_bonus = self.config.attack_conversion_bonus
                self._attack_window_remaining = 0
            elif base_reward < 0.0 and not math.isclose(self.config.failed_attack_penalty, 0.0):
                attack_failure_penalty_applied = -self.config.failed_attack_penalty
                self._attack_window_remaining = 0
            else:
                self._attack_window_remaining = max(0, self._attack_window_remaining - 1)

        raw_shaping_total = (
            rally_bonus
            + net_cross_bonus
            + return_bonus
            + offense_bonus
            + failure_penalty_applied
            + point_outcome_bonus
            + attack_conversion_bonus
            + attack_failure_penalty_applied
        )
        shaping_total = float(
            np.clip(
                raw_shaping_total,
                -self.config.max_step_shaping_abs,
                self.config.max_step_shaping_abs,
            )
        )
        shaped_reward = base_reward + shaping_total

        self._previous_frame = current_frame
        if self.config.emit_info_metrics:
            info_dict = dict(info)
            info_dict["tennis_events/shaping_total"] = float(shaping_total)
            info_dict["tennis_events/rally_bonus"] = float(rally_bonus)
            info_dict["tennis_events/net_cross_bonus"] = float(net_cross_bonus)
            info_dict["tennis_events/return_bonus"] = float(return_bonus)
            info_dict["tennis_events/offense_bonus"] = float(offense_bonus)
            info_dict["tennis_events/failure_penalty_applied"] = float(failure_penalty_applied)
            info_dict["tennis_events/point_outcome_bonus"] = float(point_outcome_bonus)
            info_dict["tennis_events/attack_conversion_bonus"] = float(attack_conversion_bonus)
            info_dict["tennis_events/attack_failure_penalty_applied"] = float(attack_failure_penalty_applied)
            info_dict["tennis_events/attack_window_started"] = float(attack_window_started)
            info_dict["tennis_events/attack_window_remaining"] = float(self._attack_window_remaining)
            info_dict["tennis_events/attack_triggered"] = float(1.0 if attack_triggered else 0.0)
            info_dict["tennis_events/base_reward"] = float(base_reward)
            info_dict["tennis_events/ball_detected_ratio"] = float(
                self._ball_detected_steps / max(1, self._episode_step_count)
            )
            info = info_dict
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
    ) -> tuple[float, bool]:
        bonus = 0.0
        net_x = frame_width / 2.0
        if self.config.agent_side == "left":
            opponent_depth = max(0.0, ball_x - net_x)
        else:
            opponent_depth = max(0.0, net_x - ball_x)

        deep_attack = opponent_depth >= frame_width * 0.25
        if deep_attack:
            bonus += self.config.deep_landing_bonus

        center_y = frame_height / 2.0
        wide_attack = abs(ball_y - center_y) >= frame_height * 0.3
        if wide_attack:
            bonus += self.config.wide_landing_bonus

        return float(bonus), bool(deep_attack or wide_attack)

    @staticmethod
    def _side_for_x(ball_x: float, *, frame_width: int) -> str:
        return "left" if ball_x < (frame_width / 2.0) else "right"


def apply_tennis_event_wrapper(env: gym.Env, tennis_event_config: TennisEventConfig | None) -> gym.Env:
    if tennis_event_config is None:
        return env
    return TennisEventRewardWrapper(env, tennis_event_config)
