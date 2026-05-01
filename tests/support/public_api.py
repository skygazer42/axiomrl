import gymnasium as gym


def register_tiny_render_env() -> str:
    env_id = "RLTrainingTest/PublicAPIDrQv2-v0"
    try:
        gym.spec(env_id)
    except gym.error.Error:
        gym.register(id=env_id, entry_point="tests.support.envs:TinyRenderContinuousEnv")
    return env_id
