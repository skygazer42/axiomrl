import gymnasium as gym
import numpy as np

from rl_training.data import HERReplayBuffer
from rl_training.envs import POINT_GOAL_ENV_ID, register_builtin_goal_envs


class _NeverTerminateGoalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del seed, options
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        del action
        return np.zeros((1,), dtype=np.float32), 0.0, False, False, {}

    def compute_reward(self, achieved_goal: object, desired_goal: object, info: object) -> float:
        del achieved_goal, desired_goal, info
        return -1.0

    def compute_terminated(self, achieved_goal: object, desired_goal: object, info: object) -> bool:
        del achieved_goal, desired_goal, info
        return False


def _make_obs(observation: float, desired_goal: float) -> dict[str, np.ndarray]:
    value = np.asarray([observation], dtype=np.float32)
    goal = np.asarray([desired_goal], dtype=np.float32)
    return {
        "observation": value,
        "achieved_goal": value.copy(),
        "desired_goal": goal,
    }


def test_her_replay_buffer_samples_goal_relabelled_batches() -> None:
    register_builtin_goal_envs()
    env = gym.make(POINT_GOAL_ENV_ID)
    buffer = HERReplayBuffer(
        capacity=32,
        num_envs=1,
        obs_shape=(1,),
        goal_shape=(1,),
        action_shape=(1,),
        her_ratio=1.0,
    )

    buffer.add(
        env_index=0,
        obs=_make_obs(0.0, 1.0),
        actions=np.asarray([0.5], dtype=np.float32),
        rewards=-1.0,
        next_obs=_make_obs(0.5, 1.0),
        terminated=False,
        truncated=False,
    )
    buffer.add(
        env_index=0,
        obs=_make_obs(0.5, 1.0),
        actions=np.asarray([0.5], dtype=np.float32),
        rewards=-1.0,
        next_obs=_make_obs(1.0, 1.0),
        terminated=True,
        truncated=False,
    )

    batch = buffer.sample(2, env=env)
    reloaded = HERReplayBuffer(
        capacity=32,
        num_envs=1,
        obs_shape=(1,),
        goal_shape=(1,),
        action_shape=(1,),
    )
    reloaded.load_state_dict(buffer.state_dict())

    assert len(buffer) == 2
    assert batch["obs"].shape[-1] == 2
    assert batch["next_obs"].shape[-1] == 2
    assert len(reloaded) == 2

    env.close()


def test_her_replay_buffer_relabels_single_transition_episodes() -> None:
    register_builtin_goal_envs()
    env = gym.make(POINT_GOAL_ENV_ID)
    buffer = HERReplayBuffer(
        capacity=8,
        num_envs=1,
        obs_shape=(1,),
        goal_shape=(1,),
        action_shape=(1,),
        her_ratio=1.0,
    )

    buffer.add(
        env_index=0,
        obs=_make_obs(0.0, 1.0),
        actions=np.asarray([0.25], dtype=np.float32),
        rewards=-1.0,
        next_obs=_make_obs(0.5, 1.0),
        terminated=False,
        truncated=True,
    )

    batch = buffer.sample(4, env=env)

    assert np.allclose(batch["obs"][:, 1].cpu().numpy(), 0.5)
    assert np.allclose(batch["next_obs"][:, 1].cpu().numpy(), 0.5)
    assert np.allclose(batch["rewards"].cpu().numpy(), 0.0)
    assert np.allclose(batch["dones"].cpu().numpy(), 1.0)

    env.close()


def test_her_replay_buffer_preserves_time_limit_truncation_when_relabelled() -> None:
    env = _NeverTerminateGoalEnv()
    buffer = HERReplayBuffer(
        capacity=8,
        num_envs=1,
        obs_shape=(1,),
        goal_shape=(1,),
        action_shape=(1,),
        her_ratio=1.0,
    )

    buffer.add(
        env_index=0,
        obs=_make_obs(0.0, 1.0),
        actions=np.asarray([0.25], dtype=np.float32),
        rewards=-1.0,
        next_obs=_make_obs(0.25, 1.0),
        terminated=False,
        truncated=True,
    )

    batch = buffer.sample(2, env=env)

    assert np.allclose(batch["rewards"].cpu().numpy(), -1.0)
    assert np.allclose(batch["dones"].cpu().numpy(), 1.0)
