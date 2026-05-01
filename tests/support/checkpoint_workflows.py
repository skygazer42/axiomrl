import gymnasium as gym


def register_tiny_render_env() -> str:
    env_id = "RLTrainingTest/CheckpointDrQv2-v0"
    try:
        gym.spec(env_id)
    except gym.error.Error:
        gym.register(id=env_id, entry_point="tests.support.envs:TinyRenderContinuousEnv")
    return env_id


def register_tiny_render_discrete_env() -> str:
    env_id = "RLTrainingTest/CheckpointDreamer-v0"
    try:
        gym.spec(env_id)
    except gym.error.Error:
        gym.register(id=env_id, entry_point="tests.support.envs:TinyRenderDiscreteEnv")
    return env_id
