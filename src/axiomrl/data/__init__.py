from __future__ import annotations

from importlib import import_module


_EXPORTS: dict[str, tuple[str, str]] = {
    "HERReplayBuffer": ("rl_training.data.her_replay_buffer", "HERReplayBuffer"),
    "NStepAccumulator": ("rl_training.data.n_step", "NStepAccumulator"),
    "PrioritizedReplayBuffer": ("rl_training.data.prioritized_replay_buffer", "PrioritizedReplayBuffer"),
    "PrioritizedRecurrentReplayBuffer": (
        "rl_training.data.prioritized_recurrent_replay_buffer",
        "PrioritizedRecurrentReplayBuffer",
    ),
    "RecurrentReplayBuffer": ("rl_training.data.recurrent_replay_buffer", "RecurrentReplayBuffer"),
    "RecurrentRolloutBuffer": ("rl_training.data.recurrent_rollout_buffer", "RecurrentRolloutBuffer"),
    "ReplayBuffer": ("rl_training.data.replay_buffer", "ReplayBuffer"),
    "RolloutBuffer": ("rl_training.data.rollout_buffer", "RolloutBuffer"),
    "RunningMeanStd": ("rl_training.data.running_mean_std", "RunningMeanStd"),
    "TrajectoryWindowDataset": ("rl_training.data.trajectory_windows", "TrajectoryWindowDataset"),
    "TransitionDataset": ("rl_training.data.offline_dataset", "TransitionDataset"),
    "collect_random_transition_dataset": ("rl_training.data.rollout_export", "collect_random_transition_dataset"),
    "compute_discounted_returns_to_go": ("rl_training.data.offline_dataset", "compute_discounted_returns_to_go"),
    "export_random_transition_dataset": ("rl_training.data.rollout_export", "export_random_transition_dataset"),
    "load_transition_dataset": ("rl_training.data.dataset_loaders", "load_transition_dataset"),
    "mix_transition_datasets": ("rl_training.data.offline_mixers", "mix_transition_datasets"),
    "save_transition_dataset_npz": ("rl_training.data.rollout_export", "save_transition_dataset_npz"),
}


def __getattr__(name: str):
    spec = _EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, export_name = spec
    module = import_module(module_name)
    value = getattr(module, export_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)

