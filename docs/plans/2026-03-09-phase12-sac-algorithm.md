# SAC Algorithm Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the first continuous-control algorithm layer to the package by implementing a minimal SAC actor-critic model and single-step update path.

**Architecture:** Keep SAC scoped to the algorithm and model layer for this slice. Reuse the existing replay-buffer-based off-policy boundary, but delay the full trainer path until the actor, critics, target update logic, and loss surface are stable under tests.

**Tech Stack:** Python 3.10+, PyTorch, Gymnasium, NumPy, PyTest

---

### Task 1: Add the SAC model layer

**Files:**
- Create: `src/rl_training/models/mlp_sac.py`
- Modify: `src/rl_training/models/__init__.py`
- Create: `tests/test_sac_update.py`

**Step 1: Write the failing test**

- Verify sampled continuous actions have the right shape
- Verify actions are bounded to `[-1, 1]` before environment scaling

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_sac_update.py`
Expected: FAIL because the SAC model does not exist

**Step 3: Write minimal implementation**

- Gaussian actor with tanh squashing
- Twin Q critics
- Small dataclass-style output for sampled actions and log-probs

**Step 4: Run the test to verify it passes**

Run: `pytest -q tests/test_sac_update.py`
Expected: PASS

### Task 2: Add the SAC algorithm layer

**Files:**
- Create: `src/rl_training/algorithms/sac.py`
- Modify: `src/rl_training/algorithms/__init__.py`
- Modify: `tests/test_sac_update.py`

**Step 1: Extend the failing test**

- Verify `sac_loss(...)` returns named metrics
- Verify `SAC.update(...)` returns `UpdateResult`

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_sac_update.py`
Expected: FAIL because SAC algorithm code does not exist

**Step 3: Write minimal implementation**

- fixed-alpha SAC update
- twin critic targets
- soft target-network updates
- metrics for actor loss, critic loss, target Q, and entropy term

**Step 4: Run the test to verify it passes**

Run: `pytest -q tests/test_sac_update.py`
Expected: PASS

### Task 3: Verify the new continuous-control algorithm slice

**Files:**
- No new files

**Step 1: Run focused tests**

Run: `pytest -q tests/test_sac_update.py`
Expected: PASS

**Step 2: Run the full suite**

Run: `pytest -q`
Expected: PASS
