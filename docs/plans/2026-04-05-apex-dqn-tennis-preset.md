# Apex DQN Tennis Preset Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an `apex_dqn_tennis` preset that resolves through both repo-local config files and packaged assets, then launch a Tennis training run with the existing package CLI.

**Architecture:** Reuse the existing `apex_dqn` trainer and Breakout preset shape. Add Tennis-specific YAML config and zoo preset files in both the repo tree and packaged asset tree, extend the tennis manifest/tests to cover the new preset, then start training with a local output directory override.

**Tech Stack:** Python, pytest, YAML preset configs, AxiomRL CLI

---

### Task 1: Add the failing preset tests

**Files:**
- Modify: `tests/test_zoo_presets.py`

**Step 1: Write the failing test**

Extend the tennis benchmark expected preset list to include `apex_dqn_tennis`, and extend the packaged tennis preset parametrized test to load `apex_dqn_tennis` with expected algorithm `apex_dqn`.

**Step 2: Run test to verify it fails**

Run: `./.env/bin/pytest -q tests/test_zoo_presets.py -k apex_dqn_tennis`
Expected: FAIL because the tennis manifest/preset files do not exist yet.

### Task 2: Add the minimal preset implementation

**Files:**
- Create: `configs/apex_dqn/tennis_atari.yaml`
- Create: `zoo/atari/apex_dqn_tennis.yaml`
- Create: `src/rl_training/assets/configs/apex_dqn/tennis_atari.yaml`
- Create: `src/rl_training/assets/zoo/atari/apex_dqn_tennis.yaml`
- Modify: `zoo/atari/tennis_benchmark.yaml`
- Modify: `src/rl_training/assets/zoo/atari/tennis_benchmark.yaml`

**Step 1: Copy the Tennis protocol shape**

Use the existing Tennis presets for:
- `env_id: ALE/Tennis-v5`
- training/evaluation sticky-action split
- checkpoint/eval cadence

**Step 2: Copy the Ape-X hyperparameter shape**

Use the existing `apex_dqn` Breakout preset as the base algorithm config, then scale it to the Tennis long-run budget with checkpoint/eval intervals.

### Task 3: Verify the preset resolves

**Files:**
- Test: `tests/test_zoo_presets.py`

**Step 1: Run focused tests**

Run: `./.env/bin/pytest -q tests/test_zoo_presets.py -k 'tennis or apex_dqn_tennis'`
Expected: PASS

### Task 4: Launch the new training run

**Files:**
- Run only

**Step 1: Start training**

Run:

```bash
./.env/bin/python -s -m rl_training.cli train \
  --config zoo/atari/apex_dqn_tennis.yaml \
  --output-dir /data/ax/axiomrl/.worktrees/tennis-rainbow/runs/tennis-apex-main
```

**Step 2: Verify startup**

Confirm:
- process exists
- run directory created
- log file shows ALE startup
- tensorboard event file appears
