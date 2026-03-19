# Test Env Registration Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify test-only Gym environment registration so custom env ids remain importable under Windows `spawn` and async vector workers.

**Architecture:** Keep production behavior unchanged except for the already-added worker-side `EnvSpec` propagation, and standardize test registrations onto importable support-module entry points. Reuse a shared `tests.support.envs` module for tiny image/render envs instead of relying on file-local classes that only exist in the parent process.

**Tech Stack:** Python, pytest, Gymnasium

---

### Task 1: Expand shared test support envs

**Files:**
- Modify: `tests/support/envs.py`

**Step 1: Write the failing test**

Use the existing async custom-env regression in `tests/test_envs.py`.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_envs.py -k "parent_registered_custom_env_with_async_backend"`

**Step 3: Write minimal implementation**

Add any missing reusable tiny env classes needed by other tests to `tests/support/envs.py`.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_envs.py -k "parent_registered_custom_env_with_async_backend"`

**Step 5: Commit**

Skip commit in this session unless explicitly requested.

### Task 2: Repoint duplicated dynamic registrations

**Files:**
- Modify: `tests/test_atari_dqn_trainer_smoke.py`
- Modify: `tests/test_atari_ppo_trainer_smoke.py`
- Modify: `tests/test_checkpoint_workflows.py`
- Modify: `tests/test_curl_trainer_smoke.py`
- Modify: `tests/test_dreamer_trainer_smoke.py`
- Modify: `tests/test_drqv2_trainer_smoke.py`
- Modify: `tests/test_gail_trainer_smoke.py`
- Modify: `tests/test_public_api.py`

**Step 1: Write the failing test**

Use focused smoke/public-api coverage that exercises these registrations.

**Step 2: Run test to verify it fails**

Only if a new focused regression is needed; otherwise rely on the red test already captured in Task 1.

**Step 3: Write minimal implementation**

Change `gym.register(..., entry_point=LocalClass)` to importable string entry points from `tests.support.envs`, passing `kwargs` where behavior differs.

**Step 4: Run test to verify it passes**

Run targeted pytest coverage for the touched files.

**Step 5: Commit**

Skip commit in this session unless explicitly requested.

### Task 3: Verify no regressions

**Files:**
- Verify: `tests/test_envs.py`
- Verify: `tests/test_*trainer_smoke.py`
- Verify: `tests/test_checkpoint_workflows.py`
- Verify: `tests/test_public_api.py`

**Step 1: Write the failing test**

Not needed; this task is verification-only.

**Step 2: Run test to verify it fails**

Not applicable.

**Step 3: Write minimal implementation**

None.

**Step 4: Run test to verify it passes**

Run focused pytest coverage, then full `pytest -q` if the focused suite is green.

**Step 5: Commit**

Skip commit in this session unless explicitly requested.
