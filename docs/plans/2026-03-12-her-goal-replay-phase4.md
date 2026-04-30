# HER Goal Replay Phase 4 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first usable `HER` baseline to `axiomrl` by combining goal-conditioned observation handling, future-goal replay relabeling, and a DDPG-based training path.

**Architecture:** Keep the policy/runtime split simple. The environment continues to emit dict observations for goal-conditioned tasks, the HER replay buffer stores goal components separately, and the policy still consumes flat `observation + desired_goal` vectors. The first release uses a thin `HER` wrapper around the existing `DDPG` algorithm so the new package capability is replay relabeling, not a second actor-critic implementation.

**Tech Stack:** Python 3.10, PyTorch, Gymnasium, pytest, setuptools

---

### Task 1: Add built-in goal-conditioned env support and helpers

**Files:**
- Create: `src/axiomrl/envs/goals.py`
- Modify: `src/axiomrl/envs/factory.py`
- Modify: `src/axiomrl/envs/__init__.py`
- Create: `tests/test_goal_envs.py`

**Step 1: Write the failing test**

Add coverage for:

- built-in goal env registration
- dict observations with `observation`, `achieved_goal`, `desired_goal`
- helper flattening for single and vectorized goal observations
- reward / termination recomputation helpers

**Step 2: Run test to verify it fails**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Implement:

- `PointGoal1DEnv`
- env registration helper invoked by the env factory
- goal observation flattening / extraction helpers

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.

### Task 2: Add episodic HER replay relabeling

**Files:**
- Create: `src/axiomrl/data/her_replay_buffer.py`
- Modify: `src/axiomrl/data/__init__.py`
- Create: `tests/test_her_replay_buffer.py`

**Step 1: Write the failing test**

Add coverage for:

- storing completed goal-conditioned episodes
- sampling relabelled transitions
- future-goal substitution changing desired goals and recomputed rewards
- replay state round-tripping through `state_dict`

**Step 2: Run test to verify it fails**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Implement:

- episodic HER replay buffer
- `future` goal sampling strategy
- reward / done recomputation hooks

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.

### Task 3: Add `HER` algorithm surface and trainer

**Files:**
- Create: `src/axiomrl/algorithms/her.py`
- Create: `src/axiomrl/runtime/her_trainer.py`
- Modify: `src/axiomrl/algorithms/__init__.py`
- Modify: `src/axiomrl/experiment/registry.py`
- Modify: `src/axiomrl/api/algorithms.py`
- Modify: `src/axiomrl/api/__init__.py`
- Modify: `src/axiomrl/__init__.py`
- Create: `configs/her/point_goal.yaml`
- Create: `src/axiomrl/assets/configs/her/point_goal.yaml`
- Create: `tests/test_her_trainer_smoke.py`

**Step 1: Write the failing test**

Add coverage for:

- low-level `HER` algorithm export
- HER trainer writing checkpoints and metrics
- evaluation / prediction from goal-conditioned checkpoints

**Step 2: Run test to verify it fails**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Implement:

- `HER` low-level class as the goal-conditioned DDPG backend surface
- `train_her(...)`
- registry load / evaluate / predict paths
- packaged config for the built-in point-goal env

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.

### Task 4: Product surface polish

**Files:**
- Modify: `README.md`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_experiment_manager.py`
- Modify: `tests/test_package_api_exports.py`
- Modify: `tests/test_package_smoke.py`
- Modify: `tests/test_cli.py`
- Modify: `docs/plans/2026-03-12-rl-expansion-roadmap-design.md`

**Step 1: Write the failing test**

Add or extend coverage so it asserts:

- `HER` is exported from root and API packages
- packaged configs include the point-goal HER preset
- README documents the first goal-conditioned workflow
- roadmap snapshot reflects that `HER` is now landing

**Step 2: Run test to verify it fails**

Deferred until the user allows testing.

**Step 3: Write minimal implementation**

Add:

- concise HER README example
- roadmap snapshot update for the new goal-conditioned surface

**Step 4: Run focused tests to verify they pass**

Deferred until the user allows testing.
