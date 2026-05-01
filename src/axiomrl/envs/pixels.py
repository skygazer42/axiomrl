from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import gymnasium as gym

from axiomrl.envs.atari import ChannelFirstObservation


@dataclass(frozen=True, slots=True)
class PixelObservationConfig:
    resize_shape: tuple[int, int] | None = (84, 84)
    grayscale: bool = False
    frame_stack: int = 3
    channel_first: bool = True
    render_only: bool = True


def _coerce_resize_shape(value: object) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise TypeError(f"expected pixel resize_shape to be an int, pair, or null, got {type(value)!r}")


def resolve_pixel_wrapper_config(wrapper_kwargs: Mapping[str, object]) -> PixelObservationConfig | None:
    requested = wrapper_kwargs.get("pixels")
    if requested in (None, False):
        return None

    if requested is True:
        requested_payload: Mapping[str, object] = {}
    else:
        if not isinstance(requested, Mapping):
            raise TypeError(f"expected wrappers['pixels'] to be a mapping or bool, got {type(requested)!r}")
        requested_payload = requested

    return PixelObservationConfig(
        resize_shape=_coerce_resize_shape(requested_payload.get("resize_shape", (84, 84))),
        grayscale=bool(requested_payload.get("grayscale", False)),
        frame_stack=int(requested_payload.get("frame_stack", 3)),
        channel_first=bool(requested_payload.get("channel_first", True)),
        render_only=bool(requested_payload.get("render_only", True)),
    )


def apply_pixel_wrappers(
    env: gym.Env,
    pixel_config: PixelObservationConfig | None,
) -> gym.Env:
    if pixel_config is None:
        return env
    if not pixel_config.render_only:
        raise ValueError("pixel wrapper currently requires render_only=True")

    wrapped = gym.wrappers.AddRenderObservation(env, render_only=pixel_config.render_only)
    if pixel_config.resize_shape is not None:
        wrapped = gym.wrappers.ResizeObservation(wrapped, pixel_config.resize_shape)
    if pixel_config.grayscale:
        wrapped = gym.wrappers.GrayscaleObservation(wrapped, keep_dim=True)
    if pixel_config.frame_stack > 1:
        wrapped = gym.wrappers.FrameStackObservation(wrapped, stack_size=pixel_config.frame_stack)
    if pixel_config.channel_first:
        wrapped = ChannelFirstObservation(wrapped)
    return wrapped
