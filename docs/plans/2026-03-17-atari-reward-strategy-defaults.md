# Atari and Game Reward Strategy Defaults Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make reward strategy configuration practical for Atari and other games by adding reusable scenario presets and ensuring explicit generic reward strategies are not silently overridden by Atari's default reward clipping.

**Architecture:** Extend the reward preset registry in `src/axiomrl/envs/rewards.py` with scenario-oriented presets such as Atari clipping and survival/goal shaping. Update Atari wrapper resolution so default `clip_reward` is disabled when a non-identity generic reward wrapper is configured, while preserving explicit `wrappers.atari.clip_reward` overrides. Cover the behavior in Atari env tests and reward wrapper tests, then document the recommended YAML recipes.

**Tech Stack:** Python 3.10+, Gymnasium wrappers, existing env factory/config system, pytest.

---

### Task 1: Add failing scenario and Atari interaction tests

**Files:**
- Modify: `tests/test_reward_wrappers.py`
- Modify: `tests/test_atari_envs.py`

**Step 1: Write the failing tests**
- Add preset coverage for `atari_clip`, `survival_penalty`, and `goal_success_bonus`.
- Add Atari env tests showing:
  - generic reward config disables default Atari clipping when `clip_reward` is omitted
  - explicit `clip_reward: true` still wins

**Step 2: Run test to verify it fails**
- Run: `pytest -q tests/test_reward_wrappers.py tests/test_atari_envs.py`
- Expected: failures because the new presets and override logic do not exist yet.

### Task 2: Implement preset registry and Atari override behavior

**Files:**
- Modify: `src/axiomrl/envs/rewards.py`
- Modify: `src/axiomrl/envs/atari.py`
- Modify: `src/axiomrl/envs/factory.py`

**Step 1: Write minimal implementation**
- Add scenario presets for Atari and generic game shaping.
- Thread resolved generic reward config into Atari wrapper resolution.
- Disable default Atari `clip_reward` only when a non-identity generic reward wrapper is active and the Atari config did not explicitly request clipping.

**Step 2: Run focused tests**
- Run: `pytest -q tests/test_reward_wrappers.py tests/test_atari_envs.py`

### Task 3: Document recommended game recipes

**Files:**
- Modify: `README.md`

**Step 1: Add docs**
- Document Atari reward strategy behavior and the automatic clip override.
- Add YAML recipes for Atari clipping, sparse-goal shaping, and survival/per-step penalties.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_reward_wrappers.py tests/test_atari_envs.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
