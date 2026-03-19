from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from functools import partial

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec

from rl_training.envs.atari import apply_atari_wrappers, resolve_atari_wrapper_config, split_env_kwargs
from rl_training.envs.goals import register_builtin_goal_envs
from rl_training.envs.pixels import apply_pixel_wrappers, resolve_pixel_wrapper_config
from rl_training.envs.rewards import apply_reward_wrappers, resolve_reward_wrapper_config
from rl_training.envs.video import apply_video_wrapper, resolve_video_wrapper_config
from rl_training.experiment.config import TrainConfig
from rl_training.runtime.vector_envs import resolve_worker_backend


EnvFactory = Callable[[], gym.Env]


def _merge_env_kwargs(base: Mapping[str, object], override: Mapping[str, object]) -> dict[str, object]:
    merged: dict[str, object] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_env_kwargs(existing, value)
        else:
            merged[key] = value
    return merged


def resolve_mode_env_kwargs(env_kwargs: Mapping[str, object], *, evaluation: bool) -> dict[str, object]:
    base_kwargs = dict(env_kwargs)
    training_overrides = base_kwargs.pop("training", None)
    evaluation_overrides = base_kwargs.pop("evaluation", None)

    selected_overrides = evaluation_overrides if evaluation else training_overrides
    if selected_overrides is None:
        return base_kwargs
    if not isinstance(selected_overrides, Mapping):
        mode = "evaluation" if evaluation else "training"
        raise TypeError(f"expected env_kwargs['{mode}'] to be a mapping, got {type(selected_overrides)!r}")
    return _merge_env_kwargs(base_kwargs, selected_overrides)


def _resolve_registered_env_spec(env_id: str) -> EnvSpec | None:
    try:
        return gym.spec(env_id)
    except gym.error.Error:
        return None


def _ensure_registered_env_spec(env_spec: EnvSpec | None) -> None:
    if env_spec is None:
        return
    if _resolve_registered_env_spec(env_spec.id) is not None:
        return
    gym.register(
        id=env_spec.id,
        entry_point=env_spec.entry_point,
        reward_threshold=env_spec.reward_threshold,
        nondeterministic=env_spec.nondeterministic,
        max_episode_steps=env_spec.max_episode_steps,
        order_enforce=env_spec.order_enforce,
        disable_env_checker=env_spec.disable_env_checker,
        additional_wrappers=tuple(env_spec.additional_wrappers),
        vector_entry_point=env_spec.vector_entry_point,
        kwargs=deepcopy(env_spec.kwargs),
    )


def build_env(
    config: TrainConfig,
    env_index: int,
    *,
    evaluation: bool = False,
    parent_env_spec: EnvSpec | None = None,
) -> gym.Env:
    register_builtin_goal_envs()
    _ensure_registered_env_spec(parent_env_spec)
    env_kwargs, wrapper_kwargs = split_env_kwargs(resolve_mode_env_kwargs(config.env_kwargs, evaluation=evaluation))
    reward_config = resolve_reward_wrapper_config(wrapper_kwargs)
    env = gym.make(config.env_id, **env_kwargs)
    env = apply_atari_wrappers(
        env,
        resolve_atari_wrapper_config(
            env_id=config.env_id,
            tags=config.tags,
            wrapper_kwargs=wrapper_kwargs,
            evaluation=evaluation,
            reward_wrapper_active=reward_config is not None,
        ),
    )
    env = apply_pixel_wrappers(env, resolve_pixel_wrapper_config(wrapper_kwargs))
    env = apply_reward_wrappers(env, reward_config)
    env = apply_video_wrapper(
        env,
        resolve_video_wrapper_config(wrapper_kwargs),
        output_dir=config.output_dir,
        env_index=env_index,
        evaluation=evaluation,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(config.seed + env_index)
    if getattr(env, "observation_space", None) is not None:
        env.observation_space.seed(config.seed + env_index)
    return env


def make_env(config: TrainConfig, env_index: int, *, evaluation: bool = False) -> EnvFactory:
    parent_env_spec = _resolve_registered_env_spec(config.env_id)
    return partial(build_env, config, env_index, evaluation=evaluation, parent_env_spec=parent_env_spec)


def make_vector_env(config: TrainConfig, *, evaluation: bool = False) -> gym.vector.VectorEnv:
    env_fns = [make_env(config, env_index, evaluation=evaluation) for env_index in range(config.num_envs)]
    backend = resolve_worker_backend(config.execution_backend)
    return backend.make_vector_env(env_fns)
