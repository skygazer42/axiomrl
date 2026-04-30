from __future__ import annotations

from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch


def _to_checkpoint_value(value: object) -> object:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return {
            "__resume_kind__": "ndarray",
            "dtype": str(value.dtype),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {
            "__resume_kind__": "tuple",
            "items": [_to_checkpoint_value(item) for item in value],
        }
    if isinstance(value, list):
        return [_to_checkpoint_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_checkpoint_value(item) for key, item in value.items()}
    return deepcopy(value)


def _from_checkpoint_value(value: object) -> object:
    if isinstance(value, dict):
        kind = value.get("__resume_kind__")
        if kind == "ndarray":
            return np.asarray(value["data"], dtype=np.dtype(str(value["dtype"])))
        if kind == "tuple":
            return tuple(_from_checkpoint_value(item) for item in value["items"])
        return {key: _from_checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_checkpoint_value(item) for item in value]
    return deepcopy(value)


def capture_resume_value(value: object) -> object:
    return _to_checkpoint_value(value)


def restore_resume_value(value: object) -> object:
    return _from_checkpoint_value(value)


def move_resume_value_to_device(value: object, *, device: torch.device | str) -> object:
    target_device = torch.device(device)
    if torch.is_tensor(value):
        return value.to(device=target_device)
    if isinstance(value, tuple):
        return tuple(move_resume_value_to_device(item, device=target_device) for item in value)
    if isinstance(value, list):
        return [move_resume_value_to_device(item, device=target_device) for item in value]
    if isinstance(value, dict):
        return {key: move_resume_value_to_device(item, device=target_device) for key, item in value.items()}
    return deepcopy(value)


def capture_global_random_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    payload: dict[str, object] = {
        "numpy": {
            "algorithm": str(numpy_state[0]),
            "keys": _to_checkpoint_value(np.asarray(numpy_state[1], dtype=np.uint32)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu().clone(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
    return payload


def restore_global_random_state(payload: dict[str, object]) -> None:
    numpy_payload = payload.get("numpy")
    if isinstance(numpy_payload, dict):
        keys = np.asarray(_from_checkpoint_value(numpy_payload["keys"]), dtype=np.uint32)
        np.random.set_state(
            (
                str(numpy_payload["algorithm"]),
                keys,
                int(numpy_payload["position"]),
                int(numpy_payload["has_gauss"]),
                float(numpy_payload["cached_gaussian"]),
            )
        )

    torch_cpu_state = payload.get("torch_cpu")
    if torch.is_tensor(torch_cpu_state):
        torch.set_rng_state(torch_cpu_state.cpu())

    torch_cuda_states = payload.get("torch_cuda")
    if torch.cuda.is_available() and isinstance(torch_cuda_states, list):
        torch.cuda.set_rng_state_all([state.cpu() for state in torch_cuda_states if torch.is_tensor(state)])


def _capture_rng_state(generator: object) -> object | None:
    if generator is None or not hasattr(generator, "bit_generator"):
        return None
    return _to_checkpoint_value(deepcopy(generator.bit_generator.state))


def _restore_rng_state(generator: object, state: object) -> None:
    if generator is None or state is None or not hasattr(generator, "bit_generator"):
        return
    generator.bit_generator.state = deepcopy(_from_checkpoint_value(state))


def _iter_env_chain(env: gym.Env):
    current = env
    while True:
        yield current
        if not isinstance(current, gym.Wrapper):
            break
        current = current.env


def _capture_env_snapshot(env: gym.Env) -> list[dict[str, object]]:
    snapshots: list[dict[str, object]] = []
    for current in _iter_env_chain(env):
        snapshot: dict[str, object] = {}
        if hasattr(current, "_elapsed_steps"):
            elapsed_steps = getattr(current, "_elapsed_steps")
            snapshot["elapsed_steps"] = None if elapsed_steps is None else int(elapsed_steps)
        if hasattr(current, "obs_queue"):
            obs_queue = getattr(current, "obs_queue")
            snapshot["obs_queue"] = [_to_checkpoint_value(item) for item in list(obs_queue)]
        if hasattr(current, "stacked_obs"):
            snapshot["stacked_obs"] = _to_checkpoint_value(getattr(current, "stacked_obs"))
        if hasattr(current, "padding_value"):
            snapshot["padding_value"] = _to_checkpoint_value(getattr(current, "padding_value"))
        if current is env.unwrapped:
            state = getattr(current, "state", None)
            if state is not None:
                snapshot["state"] = _to_checkpoint_value(np.asarray(state))
            for attr_name in ("_state", "_goal"):
                attr_value = getattr(current, attr_name, None)
                if attr_value is not None:
                    snapshot[attr_name] = _to_checkpoint_value(np.asarray(attr_value))
            for attr_name in ("_step", "_step_count"):
                if hasattr(current, attr_name):
                    snapshot[attr_name] = int(getattr(current, attr_name))
            if hasattr(current, "steps_beyond_terminated"):
                steps_beyond_terminated = getattr(current, "steps_beyond_terminated")
                snapshot["steps_beyond_terminated"] = (
                    None if steps_beyond_terminated is None else int(steps_beyond_terminated)
                )
            snapshot["env_rng_state"] = _capture_rng_state(getattr(current, "np_random", None))
            snapshot["action_space_rng_state"] = _capture_rng_state(getattr(current.action_space, "np_random", None))
        snapshots.append(snapshot)
    return snapshots


def _restore_env_snapshot(env: gym.Env, snapshots: list[dict[str, object]]) -> None:
    for current, snapshot in zip(_iter_env_chain(env), snapshots):
        if "elapsed_steps" in snapshot and hasattr(current, "_elapsed_steps"):
            setattr(current, "_elapsed_steps", snapshot["elapsed_steps"])
        if "obs_queue" in snapshot and hasattr(current, "obs_queue"):
            restored_obs_queue = [_from_checkpoint_value(item) for item in snapshot["obs_queue"]]
            maxlen = getattr(getattr(current, "obs_queue"), "maxlen", None)
            setattr(current, "obs_queue", deque(restored_obs_queue, maxlen=maxlen))
        if "stacked_obs" in snapshot and hasattr(current, "stacked_obs"):
            setattr(current, "stacked_obs", _from_checkpoint_value(snapshot["stacked_obs"]))
        if "padding_value" in snapshot and hasattr(current, "padding_value"):
            setattr(current, "padding_value", _from_checkpoint_value(snapshot["padding_value"]))
        if current is env.unwrapped:
            if "state" in snapshot:
                restored_state = np.asarray(_from_checkpoint_value(snapshot["state"]), dtype=np.float64)
                setattr(current, "state", restored_state)
            for attr_name in ("_state", "_goal"):
                if attr_name in snapshot:
                    setattr(current, attr_name, np.asarray(_from_checkpoint_value(snapshot[attr_name])))
            for attr_name in ("_step", "_step_count"):
                if attr_name in snapshot:
                    setattr(current, attr_name, int(snapshot[attr_name]))
            if "steps_beyond_terminated" in snapshot and hasattr(current, "steps_beyond_terminated"):
                setattr(current, "steps_beyond_terminated", snapshot["steps_beyond_terminated"])
            _restore_rng_state(getattr(current, "np_random", None), snapshot.get("env_rng_state"))
            _restore_rng_state(getattr(current.action_space, "np_random", None), snapshot.get("action_space_rng_state"))


class ResumeStateWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.resume_snapshot: dict[str, object] | None = None
        self._last_observation: object | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_observation = _to_checkpoint_value(obs)
        return obs, info

    def step(self, action: object):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_observation = _to_checkpoint_value(obs)
        return obs, reward, terminated, truncated, info

    def capture_resume_state(self) -> dict[str, object]:
        return {
            "observation": deepcopy(self._last_observation),
            "env_snapshot": _capture_env_snapshot(self.env),
        }

    def restore_resume_state(self) -> object | None:
        if self.resume_snapshot is None:
            return None
        snapshot = self.resume_snapshot
        env_snapshot = snapshot.get("env_snapshot")
        if isinstance(env_snapshot, list):
            _restore_env_snapshot(self.env, env_snapshot)
        restored_observation = _from_checkpoint_value(snapshot.get("observation"))
        self._last_observation = snapshot.get("observation")
        self.resume_snapshot = None
        return restored_observation


def capture_env_resume_state(env: gym.Env) -> dict[str, object] | None:
    if not hasattr(env, "capture_resume_state"):
        return None
    snapshot = env.capture_resume_state()  # type: ignore[attr-defined]
    return snapshot if isinstance(snapshot, dict) else None


def restore_env_resume_state(env: gym.Env, payload: dict[str, object]) -> object | None:
    if not hasattr(env, "restore_resume_state") or not hasattr(env, "resume_snapshot"):
        return None
    env.resume_snapshot = payload  # type: ignore[attr-defined]
    return env.restore_resume_state()  # type: ignore[attr-defined]


def capture_vector_env_resume_state(envs: gym.vector.VectorEnv) -> dict[str, object]:
    snapshots = envs.call("capture_resume_state")
    payload: dict[str, object] = {
        "env_snapshots": [snapshot for snapshot in snapshots if isinstance(snapshot, dict)],
    }
    vector_state: dict[str, object] = {}
    for attr_name in ("_autoreset_envs", "_env_obs", "_observations", "_rewards", "_terminations", "_truncations"):
        if hasattr(envs, attr_name):
            vector_state[attr_name] = _to_checkpoint_value(getattr(envs, attr_name))
    if vector_state:
        payload["vector_state"] = vector_state
    return payload


def _stack_restored_observations(restored_observations: list[object]) -> object:
    first_observation = restored_observations[0]
    if isinstance(first_observation, dict):
        return {
            key: np.stack([np.asarray(obs[key]) for obs in restored_observations if isinstance(obs, dict)])
            for key in first_observation
        }
    return np.stack([np.asarray(obs) for obs in restored_observations])


def restore_vector_env_resume_state(
    envs: gym.vector.VectorEnv,
    payload: dict[str, object],
) -> object | None:
    snapshots = payload.get("env_snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        return None
    vector_state = payload.get("vector_state")
    if not snapshots:
        return None
    envs.set_attr("resume_snapshot", snapshots)
    restored = envs.call("restore_resume_state")
    restored_observations = [obs for obs in restored if obs is not None]
    if len(restored_observations) != len(snapshots):
        return None
    if isinstance(vector_state, dict):
        for attr_name, value in vector_state.items():
            if hasattr(envs, attr_name):
                setattr(envs, attr_name, _from_checkpoint_value(value))
    if isinstance(vector_state, dict) and "_observations" in vector_state:
        return _from_checkpoint_value(vector_state["_observations"])
    return _stack_restored_observations(restored_observations)
