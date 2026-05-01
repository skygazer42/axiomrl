import torch


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    if advantages.numel() <= 1:
        return advantages

    mean = advantages.mean()
    centered = advantages - mean
    std = centered.square().mean().sqrt()
    if std < 1e-8:
        return centered
    return centered / (std + 1e-8)
