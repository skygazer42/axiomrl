from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass(frozen=True, slots=True)
class AtariWrapperConfig:
    screen_size: int = 84
    frame_skip: int = 4
    noop_max: int = 30
    grayscale_obs: bool = True
    scale_obs: bool = False
    frame_stack: int = 4
    terminal_on_life_loss: bool = False
    clip_reward: bool = True
    channel_first: bool = True


class SignedRewardWrapper(gym.RewardWrapper):
    def reward(self, reward: float) -> float:
        return float(np.sign(reward))


def _channel_first_observation(observation: np.ndarray) -> np.ndarray:
    obs = np.asarray(observation)

    if obs.ndim == 2:
        return obs[np.newaxis, ...]

    if obs.ndim == 3:
        if obs.shape[-1] in (1, 3) and obs.shape[0] not in (1, 3, 4):
            return np.moveaxis(obs, -1, 0)
        return obs

    if obs.ndim == 4 and obs.shape[-1] in (1, 3):
        moved = np.moveaxis(obs, -1, 1)
        return moved.reshape(moved.shape[0] * moved.shape[1], *moved.shape[2:])

    return obs


class ChannelFirstObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"channel-first wrapper requires Box observation space, got {type(obs_space)!r}")

        low = _channel_first_observation(np.asarray(obs_space.low))
        high = _channel_first_observation(np.asarray(obs_space.high))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=obs_space.dtype)

    def observation(self, observation: object) -> np.ndarray:
        return _channel_first_observation(np.asarray(observation))


def looks_like_atari_env(env_id: str) -> bool:
    lowered = env_id.lower()
    return env_id.startswith("ALE/") or "noframeskip" in lowered or lowered.endswith("-ram-v5")


def ensure_atari_env_registered(*, env_id: str | None = None) -> None:
    if env_id is not None and not looks_like_atari_env(env_id):
        return

    try:
        ale_py = importlib.import_module("ale_py")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Atari environments require 'ale-py'. Install Atari support with "
            "`pip install \"axiomrl[atari]\"` and download ROMs with `AutoROM --accept-license`."
        ) from exc

    register_envs = getattr(gym, "register_envs", None)
    if callable(register_envs):
        register_envs(ale_py)


def probe_atari_runtime(*, env_id: str = "ALE/Tennis-v5") -> dict[str, object]:
    status: dict[str, object] = {
        "atari_env_registration": "unknown",
        "atari_roms_available": False,
        "atari_probe_env_id": env_id,
    }

    try:
        ensure_atari_env_registered(env_id=env_id)
    except ModuleNotFoundError as exc:
        status["atari_env_registration"] = "missing_dependency"
        status["atari_probe_error"] = str(exc)
        return status
    except Exception as exc:  # pragma: no cover - defensive probe path
        status["atari_env_registration"] = "error"
        status["atari_probe_error"] = str(exc)
        return status

    status["atari_env_registration"] = "ready"
    env = None
    try:
        env = gym.make(
            env_id,
            frameskip=1,
            repeat_action_probability=0.0,
            full_action_space=False,
        )
    except Exception as exc:
        status["atari_probe_error"] = str(exc)
        return status
    finally:
        if env is not None:
            env.close()

    status["atari_roms_available"] = True
    status["atari_probe_error"] = "none"
    return status


def split_env_kwargs(env_kwargs: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    base_kwargs = dict(env_kwargs)
    wrapper_kwargs = base_kwargs.pop("wrappers", {})
    if wrapper_kwargs in (None, False):
        return base_kwargs, {}
    if not isinstance(wrapper_kwargs, Mapping):
        raise TypeError(f"expected env_kwargs['wrappers'] to be a mapping, got {type(wrapper_kwargs)!r}")
    return base_kwargs, dict(wrapper_kwargs)


def resolve_atari_wrapper_config(
    *,
    env_id: str,
    tags: Sequence[str],
    wrapper_kwargs: Mapping[str, object],
    evaluation: bool,
    reward_wrapper_active: bool = False,
) -> AtariWrapperConfig | None:
    requested = wrapper_kwargs.get("atari")
    use_defaults = "atari" in tags or looks_like_atari_env(env_id)

    if requested is False:
        return None
    if requested is None and not use_defaults:
        return None
    if requested is True or requested is None:
        requested_payload: Mapping[str, object] = {}
    else:
        if not isinstance(requested, Mapping):
            raise TypeError(f"expected wrappers['atari'] to be a mapping or bool, got {type(requested)!r}")
        requested_payload = requested

    default_clip_reward = not evaluation and not reward_wrapper_active
    clip_reward = bool(requested_payload.get("clip_reward", default_clip_reward))

    return AtariWrapperConfig(
        screen_size=int(requested_payload.get("screen_size", 84)),
        frame_skip=int(requested_payload.get("frame_skip", 4)),
        noop_max=int(requested_payload.get("noop_max", 30)),
        grayscale_obs=bool(requested_payload.get("grayscale_obs", True)),
        scale_obs=bool(requested_payload.get("scale_obs", False)),
        frame_stack=int(requested_payload.get("frame_stack", 4)),
        terminal_on_life_loss=bool(requested_payload.get("terminal_on_life_loss", False)),
        clip_reward=clip_reward,
        channel_first=bool(requested_payload.get("channel_first", True)),
    )


def apply_atari_wrappers(
    env: gym.Env,
    atari_config: AtariWrapperConfig | None,
) -> gym.Env:
    if atari_config is None:
        return env

    wrapped = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=atari_config.noop_max,
        frame_skip=atari_config.frame_skip,
        screen_size=atari_config.screen_size,
        terminal_on_life_loss=atari_config.terminal_on_life_loss,
        grayscale_obs=atari_config.grayscale_obs,
        scale_obs=atari_config.scale_obs,
    )

    if atari_config.frame_stack > 1:
        wrapped = gym.wrappers.FrameStackObservation(wrapped, stack_size=atari_config.frame_stack)
    if atari_config.channel_first:
        wrapped = ChannelFirstObservation(wrapped)
    if atari_config.clip_reward:
        wrapped = SignedRewardWrapper(wrapped)
    return wrapped
