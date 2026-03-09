# Eval and Resume Workflows Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn checkpoint-based evaluation and resume into real package workflows so `eval` and `resume` are no longer placeholders.

**Architecture:** Keep algorithm-specific training loops in their own trainer modules, but add shared checkpoint-driven workflow helpers that can reconstruct PPO or DQN from saved config and algorithm state. CLI dispatch should stay thin and call package functions rather than own recovery logic directly.

**Tech Stack:** Python 3.10+, PyTorch, Gymnasium, NumPy, PyTest

---

### Task 1: Add function-level checkpoint workflows

**Files:**
- Create: `src/rl_training/runtime/workflows.py`
- Modify: `src/rl_training/runtime/ppo_trainer.py`
- Modify: `src/rl_training/runtime/dqn_trainer.py`
- Create: `tests/test_checkpoint_workflows.py`

**Step 1: Write the failing test**

- Verify evaluating a PPO checkpoint returns metrics
- Verify resuming a DQN checkpoint returns a new checkpoint and advances `global_step`

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_checkpoint_workflows.py`
Expected: FAIL because workflow helpers do not exist

**Step 3: Write minimal implementation**

- Load `CheckpointState`
- Rebuild `TrainConfig`
- Rebuild PPO or DQN algorithm objects from config
- Load saved algorithm state
- Evaluate from checkpoint
- Resume from checkpoint with overridden `total_timesteps`

**Step 4: Run the test to verify it passes**

Run: `pytest -q tests/test_checkpoint_workflows.py`
Expected: PASS

### Task 2: Wire the CLI `eval` and `resume` commands

**Files:**
- Modify: `src/rl_training/cli.py`
- Modify: `tests/test_cli.py`

**Step 1: Extend the failing test**

- Verify `main(["eval", ...])` returns `0`
- Verify `main(["resume", ...])` returns `0`

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_cli.py`
Expected: FAIL because commands still raise `NotImplementedError`

**Step 3: Write minimal implementation**

- `eval --checkpoint --num-episodes`
- `resume --checkpoint --total-timesteps`
- Dispatch through workflow helpers

**Step 4: Run the test to verify it passes**

Run: `pytest -q tests/test_cli.py`
Expected: PASS

### Task 3: Verify the package with the new workflows

**Files:**
- No new files

**Step 1: Run focused workflow tests**

Run: `pytest -q tests/test_checkpoint_workflows.py tests/test_cli.py`
Expected: PASS

**Step 2: Run the full suite**

Run: `pytest -q`
Expected: PASS
