from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "HERReplayBuffer": ("axiomrl.data.her_replay_buffer", "HERReplayBuffer"),
    "NStepAccumulator": ("axiomrl.data.n_step", "NStepAccumulator"),
    "PrioritizedReplayBuffer": ("axiomrl.data.prioritized_replay_buffer", "PrioritizedReplayBuffer"),
    "PrioritizedRecurrentReplayBuffer": (
        "axiomrl.data.prioritized_recurrent_replay_buffer",
        "PrioritizedRecurrentReplayBuffer",
    ),
    "RecurrentReplayBuffer": ("axiomrl.data.recurrent_replay_buffer", "RecurrentReplayBuffer"),
    "RecurrentRolloutBuffer": ("axiomrl.data.recurrent_rollout_buffer", "RecurrentRolloutBuffer"),
    "ReplayBuffer": ("axiomrl.data.replay_buffer", "ReplayBuffer"),
    "RolloutBuffer": ("axiomrl.data.rollout_buffer", "RolloutBuffer"),
    "RunningMeanStd": ("axiomrl.data.running_mean_std", "RunningMeanStd"),
    "TrajectoryWindowDataset": ("axiomrl.data.trajectory_windows", "TrajectoryWindowDataset"),
    "TransitionDataset": ("axiomrl.data.offline_dataset", "TransitionDataset"),
    "collect_random_transition_dataset": ("axiomrl.data.rollout_export", "collect_random_transition_dataset"),
    "compute_discounted_returns_to_go": ("axiomrl.data.offline_dataset", "compute_discounted_returns_to_go"),
    "export_random_transition_dataset": ("axiomrl.data.rollout_export", "export_random_transition_dataset"),
    "load_transition_dataset": ("axiomrl.data.dataset_loaders", "load_transition_dataset"),
    "mix_transition_datasets": ("axiomrl.data.offline_mixers", "mix_transition_datasets"),
    "save_transition_dataset_npz": ("axiomrl.data.rollout_export", "save_transition_dataset_npz"),
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
