import pytest

from rl_training.data.n_step import NStepAccumulator


def test_n_step_accumulator_emits_transition_after_n_steps() -> None:
    accumulator = NStepAccumulator(num_envs=1, n_step=3, gamma=1.0)

    assert accumulator.add(0, "s0", 0, 1.0, "s1", False) == []
    assert accumulator.add(0, "s1", 1, 2.0, "s2", False) == []

    transitions = accumulator.add(0, "s2", 2, 3.0, "s3", False)

    assert len(transitions) == 1
    transition = transitions[0]
    assert transition["obs"] == "s0"
    assert transition["actions"] == 0
    assert transition["rewards"] == pytest.approx(6.0)
    assert transition["next_obs"] == "s3"
    assert transition["dones"] == pytest.approx(0.0)


def test_n_step_accumulator_applies_discounting() -> None:
    accumulator = NStepAccumulator(num_envs=1, n_step=3, gamma=0.5)

    accumulator.add(0, "s0", 0, 1.0, "s1", False)
    accumulator.add(0, "s1", 1, 2.0, "s2", False)
    transitions = accumulator.add(0, "s2", 2, 3.0, "s3", False)

    transition = transitions[0]
    assert abs(transition["rewards"] - (1.0 + 0.5 * 2.0 + 0.25 * 3.0)) < 1e-12


def test_n_step_accumulator_flushes_remaining_transitions_on_done() -> None:
    accumulator = NStepAccumulator(num_envs=1, n_step=3, gamma=1.0)

    assert accumulator.add(0, "s0", 0, 1.0, "s1", False) == []
    assert accumulator.add(0, "s1", 1, 2.0, "s2", False) == []

    flushed = accumulator.add(0, "s2", 2, 3.0, "terminal", True)

    assert [transition["obs"] for transition in flushed] == ["s0", "s1", "s2"]
    assert [transition["actions"] for transition in flushed] == [0, 1, 2]
    assert [transition["rewards"] for transition in flushed] == pytest.approx([6.0, 5.0, 3.0])
    assert [transition["next_obs"] for transition in flushed] == ["terminal", "terminal", "terminal"]
    assert [transition["dones"] for transition in flushed] == pytest.approx([1.0, 1.0, 1.0])


def test_n_step_accumulator_is_isolated_per_env() -> None:
    accumulator = NStepAccumulator(num_envs=2, n_step=2, gamma=1.0)

    assert accumulator.add(0, "s0", 0, 1.0, "s1", False) == []
    assert accumulator.add(1, "t0", 3, 5.0, "t1", False) == []

    transitions_env0 = accumulator.add(0, "s1", 1, 2.0, "s2", False)
    transitions_env1 = accumulator.add(1, "t1", 4, 7.0, "t2", False)

    assert len(transitions_env0) == 1
    assert transitions_env0[0]["obs"] == "s0"
    assert transitions_env0[0]["rewards"] == pytest.approx(3.0)

    assert len(transitions_env1) == 1
    assert transitions_env1[0]["obs"] == "t0"
    assert transitions_env1[0]["rewards"] == pytest.approx(12.0)
