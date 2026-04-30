import torch


def test_running_mean_std_updates_mean_and_var_for_batch() -> None:
    from axiomrl.data.running_mean_std import RunningMeanStd

    rms = RunningMeanStd()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    rms.update(x)

    assert rms.count == 2
    assert torch.allclose(rms.mean, torch.tensor([2.0, 3.0]))
    assert torch.allclose(rms.var, torch.tensor([1.0, 1.0]))
    assert torch.allclose(rms.std, torch.tensor([1.0, 1.0]))


def test_running_mean_std_matches_combined_statistics_across_updates() -> None:
    from axiomrl.data.running_mean_std import RunningMeanStd

    rms = RunningMeanStd()
    rms.update(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))
    rms.update(torch.tensor([[5.0, 6.0]], dtype=torch.float32))

    combined = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    expected_mean = combined.mean(dim=0)
    expected_var = combined.var(dim=0, unbiased=False)

    assert rms.count == 3
    assert torch.allclose(rms.mean, expected_mean)
    assert torch.allclose(rms.var, expected_var)


def test_running_mean_std_supports_single_sample_vector() -> None:
    from axiomrl.data.running_mean_std import RunningMeanStd

    rms = RunningMeanStd()
    rms.update(torch.tensor([10.0, 20.0], dtype=torch.float32))

    assert rms.count == 1
    assert torch.allclose(rms.mean, torch.tensor([10.0, 20.0]))
    assert torch.allclose(rms.var, torch.tensor([0.0, 0.0]))
