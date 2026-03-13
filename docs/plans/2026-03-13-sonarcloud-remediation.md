# SonarCloud Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the current open SonarCloud issues for `skygazer42_RL` without changing training behavior.

**Architecture:** Apply low-risk mechanical fixes first, then address float-comparison findings with tolerant checks, and finally refactor the complexity-heavy trainer functions by extracting helpers. Keep file ownership narrow per change set and verify each batch with focused pytest runs before moving on.

**Tech Stack:** Python, PyTorch, NumPy, pytest, Gymnasium.

---

### Task 1: Mechanical Sonar fixes

**Files:**
- Modify: `src/rl_training/algorithms/*.py`
- Modify: `src/rl_training/contrib/recurrent_ppo.py`
- Modify: `src/rl_training/data/her_replay_buffer.py`
- Modify: `src/rl_training/data/running_mean_std.py`
- Modify: `src/rl_training/api/algorithms.py`
- Modify: `src/rl_training/runtime/workflows.py`
- Modify: `src/rl_training/cli.py`
- Modify: `src/rl_training/runtime/iql_trainer.py`
- Modify: `src/rl_training/zoo_cli.py`
- Modify: `src/rl_training/experiment/runs.py`
- Test: `tests/test_running_mean_std.py`
- Test: `tests/test_dataset_loaders.py`

**Step 1:** Update tests or add focused regression coverage for any behavior-sensitive changes.
**Step 2:** Run the focused tests and confirm the new assertions fail before the production change when applicable.
**Step 3:** Apply the mechanical fixes: explicit `weight_decay=0.0`, replace flagged variance arguments, modernize NumPy RNG, extract duplicated literals/constants, fix type hints, remove invariant-return structure, and make timezone-aware run IDs.
**Step 4:** Run focused pytest commands for touched areas.

### Task 2: Float-comparison fixes

**Files:**
- Modify: `src/rl_training/algorithms/marwil.py`
- Modify: `src/rl_training/data/offline_dataset.py`
- Modify: `src/rl_training/data/prioritized_replay_buffer.py`
- Modify: `src/rl_training/envs/rewards.py`
- Modify: `tests/test_atari_envs.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_goal_envs.py`
- Modify: `tests/test_reward_wrappers.py`
- Modify: `tests/test_schedules.py`
- Modify: `tests/test_dqn_trainer_smoke.py`
- Modify: `tests/test_dqn_update.py`
- Modify: `tests/test_n_step_accumulator.py`
- Modify: `tests/test_replay_buffer.py`
- Modify: `tests/test_run_utils.py`
- Modify: `tests/test_dataset_loaders.py`

**Step 1:** Replace direct float equality assertions with `pytest.approx`, `math.isclose`, or tolerant tensor checks.
**Step 2:** Replace production float equality branches with tolerance-based logic that preserves intent.
**Step 3:** Run focused pytest coverage for these modules.

### Task 3: Cognitive complexity refactors

**Files:**
- Modify: `src/rl_training/algorithms/trpo.py`
- Modify: `src/rl_training/data/offline_mixers.py`
- Modify: `src/rl_training/runtime/crossq_trainer.py`
- Modify: `src/rl_training/runtime/ddpg_trainer.py`
- Modify: `src/rl_training/runtime/dqn_trainer.py`
- Modify: `src/rl_training/runtime/drqv2_trainer.py`
- Modify: `src/rl_training/runtime/her_trainer.py`
- Modify: `src/rl_training/runtime/redq_trainer.py`
- Modify: `src/rl_training/runtime/rlpd_trainer.py`
- Modify: `src/rl_training/runtime/sac_trainer.py`
- Modify: `src/rl_training/runtime/td3_trainer.py`
- Modify: `src/rl_training/runtime/tqc_trainer.py`

**Step 1:** Extract helpers to flatten nested control flow without changing behavior.
**Step 2:** Run targeted trainer tests and smoke tests.
**Step 3:** Run a broader pytest pass once all refactors land.
