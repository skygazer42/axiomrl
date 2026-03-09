# TensorBoard Logging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add real TensorBoard scalar logging to training runs while keeping the existing JSON/JSONL artifacts intact.

**Architecture:** Extend the experiment logger so one logger instance writes config and scalar metrics to both the existing filesystem artifacts and TensorBoard event files inside each run's `tensorboard/` directory. Keep trainer code unchanged by preserving the current `Logger` interface and wiring the TensorBoard writer through the existing run setup helper.

**Tech Stack:** Python 3.10+, PyTorch `SummaryWriter`, PyTest

---

### Task 1: Add failing tests for TensorBoard-backed logging

**Files:**
- Modify: `tests/test_run_utils.py`
- Modify: `tests/test_experiment_contracts.py`

**Step 1: Write the failing tests**

```python
def test_create_training_run_emits_tensorboard_event_file(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=13,
        total_timesteps=64,
        output_dir=tmp_path,
    )

    artifacts = create_training_run(config, run_suffix="tb")
    try:
        artifacts.logger.log_metrics({"loss": 1.25}, step=7)
    finally:
        artifacts.close()

    event_files = list(artifacts.run_context.tensorboard_dir.glob("events.out.tfevents.*"))
    assert event_files
```

```python
def test_create_run_context_exposes_tensorboard_directory(tmp_path: Path) -> None:
    config = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=7,
        total_timesteps=128,
        output_dir=tmp_path,
    )

    context = create_run_context(config, run_suffix="manual")
    assert context.tensorboard_dir.exists()
```

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_run_utils.py tests/test_experiment_contracts.py`
Expected: FAIL because the logger does not write TensorBoard event files yet

**Step 3: Write minimal implementation**

- extend the logger to open a `SummaryWriter` rooted at `RunContext.tensorboard_dir`
- keep JSON/JSONL output behavior intact
- ensure `close()` flushes and closes the writer

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_run_utils.py tests/test_experiment_contracts.py`
Expected: PASS

### Task 2: Wire TensorBoard logging through run setup and dependencies

**Files:**
- Modify: `src/rl_training/experiment/logging.py`
- Modify: `src/rl_training/runtime/run_utils.py`
- Modify: `pyproject.toml`

**Step 1: Use the failing tests**

Run: `pytest -q tests/test_run_utils.py tests/test_experiment_contracts.py`
Expected: FAIL before wiring is complete

**Step 2: Write minimal implementation**

- accept a `tensorboard_dir` when constructing the concrete logger
- log scalar numeric metrics to TensorBoard using the same `step`
- preserve config logging and JSONL metrics logging
- declare the runtime `tensorboard` dependency

**Step 3: Run focused tests to verify behavior**

Run: `pytest -q tests/test_run_utils.py tests/test_experiment_contracts.py`
Expected: PASS

### Task 3: Verify the package after logger integration

**Files:**
- No new files

**Step 1: Run focused training tests**

Run: `pytest -q tests/test_run_utils.py tests/test_experiment_contracts.py tests/test_trainer_smoke.py tests/test_dqn_trainer_smoke.py tests/test_sac_trainer_smoke.py tests/test_a2c_trainer_smoke.py tests/test_td3_trainer_smoke.py`
Expected: PASS

**Step 2: Run the full suite**

Run: `pytest -q`
Expected: PASS
