from __future__ import annotations

import torch


class RunningMeanStd:
    def __init__(self) -> None:
        self._mean = torch.tensor(0.0, dtype=torch.float32)
        self._var = torch.tensor(1.0, dtype=torch.float32)
        self._count = 0

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def var(self) -> torch.Tensor:
        return self._var

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self._var)

    @property
    def count(self) -> int:
        return int(self._count)

    def update(self, x: object) -> None:
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        if x_tensor.numel() == 0:
            return

        if x_tensor.ndim == 0:
            batch_count = 1
            batch_mean = x_tensor
            batch_var = torch.zeros_like(batch_mean)
        elif x_tensor.ndim == 1:
            batch_count = 1
            batch_mean = x_tensor
            batch_var = torch.zeros_like(batch_mean)
        else:
            batch_count = int(x_tensor.shape[0])
            if batch_count == 0:
                return
            batch_mean = x_tensor.mean(dim=0)
            batch_var = x_tensor.var(dim=0, unbiased=False)

        if self._count == 0:
            self._mean = batch_mean.detach().clone()
            self._var = batch_var.detach().clone()
            self._count = int(batch_count)
            return

        if batch_mean.shape != self._mean.shape:
            raise ValueError(f"shape mismatch: expected {tuple(self._mean.shape)}, got {tuple(batch_mean.shape)}")

        if batch_mean.device != self._mean.device:
            batch_mean = batch_mean.to(device=self._mean.device)
            batch_var = batch_var.to(device=self._mean.device)

        count_a = float(self._count)
        count_b = float(batch_count)
        total_count = count_a + count_b
        delta = batch_mean - self._mean

        new_mean = self._mean + delta * (count_b / total_count)
        m_a = self._var * count_a
        m_b = batch_var * count_b
        m2 = m_a + m_b + delta.pow(2) * (count_a * count_b / total_count)
        new_var = m2 / total_count

        self._mean = new_mean
        self._var = new_var
        self._count += int(batch_count)

