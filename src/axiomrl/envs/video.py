from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym


@dataclass(frozen=True, slots=True)
class VideoWrapperConfig:
    video_folder: str | None = None
    name_prefix: str | None = None
    episode_trigger_every: int | None = 1
    step_trigger_every: int | None = None
    video_length: int = 0
    fps: int | None = None
    disable_logger: bool = True


def _resolve_trigger_every(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"wrappers['video']['{field_name}'] must be a positive integer")
    return resolved


def resolve_video_wrapper_config(wrapper_kwargs: Mapping[str, object]) -> VideoWrapperConfig | None:
    requested = wrapper_kwargs.get("video")
    if requested in (None, False):
        return None
    if requested is True:
        requested_payload: Mapping[str, object] = {}
    else:
        if not isinstance(requested, Mapping):
            raise TypeError(f"expected wrappers['video'] to be a mapping or bool, got {type(requested)!r}")
        requested_payload = requested

    episode_trigger_every = _resolve_trigger_every(
        requested_payload.get("episode_trigger_every"),
        field_name="episode_trigger_every",
    )
    step_trigger_every = _resolve_trigger_every(
        requested_payload.get("step_trigger_every"),
        field_name="step_trigger_every",
    )
    if episode_trigger_every is not None and step_trigger_every is not None:
        raise ValueError("wrappers['video'] cannot define both 'episode_trigger_every' and 'step_trigger_every'")
    if episode_trigger_every is None and step_trigger_every is None:
        episode_trigger_every = 1

    video_folder = requested_payload.get("video_folder")
    name_prefix = requested_payload.get("name_prefix")
    fps = requested_payload.get("fps")

    return VideoWrapperConfig(
        video_folder=None if video_folder is None else str(video_folder),
        name_prefix=None if name_prefix is None else str(name_prefix),
        episode_trigger_every=episode_trigger_every,
        step_trigger_every=step_trigger_every,
        video_length=int(requested_payload.get("video_length", 0)),
        fps=None if fps is None else int(fps),
        disable_logger=bool(requested_payload.get("disable_logger", True)),
    )


def apply_video_wrapper(
    env: gym.Env,
    video_config: VideoWrapperConfig | None,
    *,
    output_dir: Path,
    env_index: int,
    evaluation: bool,
) -> gym.Env:
    if video_config is None:
        return env

    if video_config.video_folder is None:
        video_dir = output_dir / "videos" / ("evaluation" if evaluation else "training")
    else:
        requested_dir = Path(video_config.video_folder)
        video_dir = requested_dir if requested_dir.is_absolute() else (output_dir / requested_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    episode_trigger = None
    if video_config.episode_trigger_every is not None:
        episode_interval = int(video_config.episode_trigger_every)
        episode_trigger = lambda episode_id: episode_id % episode_interval == 0

    step_trigger = None
    if video_config.step_trigger_every is not None:
        step_interval = int(video_config.step_trigger_every)
        step_trigger = lambda step_id: step_id % step_interval == 0

    name_prefix = video_config.name_prefix
    if name_prefix is None:
        name_prefix = f"{'evaluation' if evaluation else 'training'}-env{env_index}"

    return gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        episode_trigger=episode_trigger,
        step_trigger=step_trigger,
        video_length=video_config.video_length,
        name_prefix=name_prefix,
        fps=video_config.fps,
        disable_logger=video_config.disable_logger,
    )
