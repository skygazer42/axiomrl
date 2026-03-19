import gymnasium as gym
import numpy as np


class TinyImageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self._step = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action: int):
        del action
        self._step += 1
        obs = np.full(self.observation_space.shape, fill_value=self._step, dtype=np.uint8)
        terminated = self._step >= 4
        truncated = False
        return obs, 1.0, terminated, truncated, {}


class TinyImageDiscreteEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, num_actions: int = 2, reward_action: int = 1) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(num_actions)
        self.reward_action = int(reward_action)
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self._step = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action: int):
        action_int = int(action)
        self._step += 1
        obs = np.full(self.observation_space.shape, fill_value=self._step + action_int, dtype=np.uint8)
        terminated = self._step >= 4
        truncated = False
        reward = float(action_int == self.reward_action)
        return obs, reward, terminated, truncated, {}


class TinyRenderContinuousEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._step = 0
        self._state = np.zeros(3, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self._step = 0
        self._state.fill(0.0)
        return self._state.copy(), {}

    def step(self, action: np.ndarray):
        action_value = float(np.asarray(action).reshape(-1)[0])
        self._step += 1
        self._state = np.array([action_value, self._step / 4.0, -action_value], dtype=np.float32)
        terminated = self._step >= 4
        truncated = False
        reward = 1.0 - abs(action_value)
        return self._state.copy(), reward, terminated, truncated, {}

    def render(self) -> np.ndarray:
        canvas = np.zeros((96, 96, 3), dtype=np.uint8)
        action_intensity = int(np.clip((self._state[0] + 1.0) * 127.5, 0, 255))
        canvas[..., 0] = np.uint8(self._step * 32)
        canvas[16:80, 16:80, 1] = np.uint8(action_intensity)
        canvas[32:64, 32:64, 2] = np.uint8(255 - action_intensity)
        return canvas


class TinyRenderDiscreteEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self._step = 0
        self._state = np.zeros(2, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self._step = 0
        self._state.fill(0.0)
        return self._state.copy(), {}

    def step(self, action: int):
        action_int = int(action)
        self._step += 1
        self._state = np.array([action_int / 3.0, self._step / 6.0], dtype=np.float32)
        terminated = self._step >= 6
        truncated = False
        reward = 1.0 if action_int == (self._step % self.action_space.n) else 0.0
        return self._state.copy(), reward, terminated, truncated, {}

    def render(self) -> np.ndarray:
        canvas = np.zeros((96, 96, 3), dtype=np.uint8)
        canvas[..., 0] = np.uint8(self._step * 24)
        canvas[16:80, 16:80, 1] = np.uint8(np.clip(self._state[0] * 255, 0, 255))
        canvas[32:64, 32:64, 2] = np.uint8(np.clip(self._state[1] * 255, 0, 255))
        return canvas
