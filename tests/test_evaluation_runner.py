from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_training.experiment.config import TrainConfig
from rl_training.runtime import evaluation_runner as evaluation_runner_module
from rl_training.runtime.evaluation_runner import EvaluationRunner


def test_evaluation_runner_returns_eval_result_for_discrete_policy(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=13,
        total_timesteps=64,
        output_dir=tmp_path,
        eval_episodes=1,
    )

    runner = EvaluationRunner(
        config=config,
        device=torch.device("cpu"),
        action_fn=lambda obs_tensor: 0,
    )

    result = runner.evaluate(num_episodes=1)

    assert result.num_episodes == 1
    assert set(result.metrics) >= {"eval_return_mean", "eval_return_std", "eval_episodes"}


def test_evaluation_runner_resets_stateful_action_fn_between_episodes(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=17,
        total_timesteps=64,
        output_dir=tmp_path,
        eval_episodes=2,
    )

    class StatefulActionFn:
        def __init__(self) -> None:
            self.reset_count = 0

        def reset(self) -> None:
            self.reset_count += 1

        def __call__(self, obs_tensor: torch.Tensor) -> int:
            del obs_tensor
            return 0

    action_fn = StatefulActionFn()
    runner = EvaluationRunner(
        config=config,
        device=torch.device("cpu"),
        action_fn=action_fn,
    )

    result = runner.evaluate(num_episodes=2)

    assert result.num_episodes == 2
    assert action_fn.reset_count == 2


def test_evaluation_runner_binds_env_aware_action_fn(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=19,
        total_timesteps=64,
        output_dir=tmp_path,
        eval_episodes=1,
    )

    class EnvAwareActionFn:
        def __init__(self) -> None:
            self.bound_action_space_shape: tuple[int, ...] | None = None

        def bind_env(self, env) -> None:  # type: ignore[no-untyped-def]
            self.bound_action_space_shape = tuple(env.action_space.shape)

        def __call__(self, obs_tensor: torch.Tensor) -> int:
            del obs_tensor
            return 0

    action_fn = EnvAwareActionFn()
    runner = EvaluationRunner(
        config=config,
        device=torch.device("cpu"),
        action_fn=action_fn,
    )

    result = runner.evaluate(num_episodes=1)

    assert result.num_episodes == 1
    assert action_fn.bound_action_space_shape == ()


def test_evaluation_runner_prepares_raw_observations_before_action_selection(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="GoalEvalTest-v0",
        seed=23,
        total_timesteps=64,
        output_dir=tmp_path,
        eval_episodes=1,
    )

    class GoalLikeEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                    "achieved_goal": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                    "desired_goal": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                }
            )
            self.action_space = gym.spaces.Discrete(2)
            self._step = 0

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            del options
            self._step = 0
            return {
                "observation": np.zeros(2, dtype=np.float32),
                "achieved_goal": np.zeros(1, dtype=np.float32),
                "desired_goal": np.ones(1, dtype=np.float32),
            }, {}

        def step(self, action: int):
            del action
            self._step += 1
            return (
                {
                    "observation": np.full(2, float(self._step), dtype=np.float32),
                    "achieved_goal": np.ones(1, dtype=np.float32),
                    "desired_goal": np.ones(1, dtype=np.float32),
                },
                1.0,
                True,
                False,
                {},
            )

    class GoalAwareActionFn:
        def __init__(self) -> None:
            self.prepare_calls = 0

        def prepare_observation(self, obs) -> torch.Tensor:  # type: ignore[no-untyped-def]
            self.prepare_calls += 1
            flat_obs = np.concatenate(
                [
                    np.asarray(obs["observation"], dtype=np.float32),
                    np.asarray(obs["achieved_goal"], dtype=np.float32),
                    np.asarray(obs["desired_goal"], dtype=np.float32),
                ]
            )
            return torch.as_tensor(flat_obs, dtype=torch.float32)

        def __call__(self, obs_tensor: torch.Tensor) -> int:
            assert tuple(obs_tensor.shape) == (4,)
            return 0

    original_build_env = evaluation_runner_module.build_env
    evaluation_runner_module.build_env = lambda *args, **kwargs: GoalLikeEnv()
    try:
        action_fn = GoalAwareActionFn()
        runner = EvaluationRunner(
            config=config,
            device=torch.device("cpu"),
            action_fn=action_fn,
        )

        result = runner.evaluate(num_episodes=1)
    finally:
        evaluation_runner_module.build_env = original_build_env

    assert result.num_episodes == 1
    assert action_fn.prepare_calls == 1


def test_evaluation_runner_aggregates_action_fn_episode_metrics(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="EpisodeMetricEvalTest-v0",
        seed=29,
        total_timesteps=64,
        output_dir=tmp_path,
        eval_episodes=2,
    )

    class SingleStepEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            del options
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action: int):
            del action
            return np.ones(2, dtype=np.float32), 1.0, True, False, {"is_success": True}

    class MetricActionFn:
        def __init__(self) -> None:
            self.after_step_calls = 0
            self.success = 0.0

        def reset(self) -> None:
            self.success = 0.0

        def __call__(self, obs_tensor: torch.Tensor) -> int:
            del obs_tensor
            return 0

        def after_step(self, next_obs, reward: float, done: bool, truncated: bool, info) -> None:  # type: ignore[no-untyped-def]
            del next_obs, reward
            self.after_step_calls += 1
            self.success = float(done and not truncated and bool(info.get("is_success", False)))

        def episode_metrics(self) -> dict[str, float]:
            return {"eval_success_rate": self.success}

    original_build_env = evaluation_runner_module.build_env
    evaluation_runner_module.build_env = lambda *args, **kwargs: SingleStepEnv()
    try:
        action_fn = MetricActionFn()
        runner = EvaluationRunner(
            config=config,
            device=torch.device("cpu"),
            action_fn=action_fn,
        )

        result = runner.evaluate(num_episodes=2)
    finally:
        evaluation_runner_module.build_env = original_build_env

    assert result.num_episodes == 2
    assert action_fn.after_step_calls == 2
    assert result.metrics["eval_success_rate"] == 1.0
