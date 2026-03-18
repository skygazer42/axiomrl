# Environment Mode Overrides and Evaluation Protocol Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add declarative per-mode environment overrides so training and evaluation can use different Atari/game protocols from the same config file.

**Architecture:** Keep `TrainConfig` unchanged and extend `env_kwargs` parsing in `src/rl_training/envs/factory.py`. Base `env_kwargs` remain the shared scenario definition, while nested `training` and `evaluation` mappings provide mode-specific overrides that are recursively merged before wrapper resolution. This keeps existing configs backward compatible and creates a stable foundation for Atari evaluation protocols, video capture, sticky-action settings, and benchmark recipes.

**Tech Stack:** Python 3.10+, Gymnasium env factory/wrappers, existing config system, pytest.

---

### Task 1: Add failing mode-override tests

**Files:**
- Modify: `tests/test_atari_envs.py`
- Modify: `tests/test_envs.py`

**Step 1: Write the failing tests**
- Add coverage for `env_kwargs.evaluation` overriding base Atari kwargs like `repeat_action_probability`.
- Add coverage for `env_kwargs.training` overriding base values for train env construction.
- Add coverage for recursive wrapper merges so evaluation-specific `wrappers.atari.clip_reward` overrides do not discard base `wrappers.atari.frame_stack`.

**Step 2: Run test to verify it fails**
- Run: `pytest -q tests/test_atari_envs.py tests/test_envs.py`
- Expected: failures because mode-specific overrides are not resolved yet.

### Task 2: Implement env mode override resolution

**Files:**
- Modify: `src/rl_training/envs/factory.py`

**Step 1: Write minimal implementation**
- Add a recursive mapping merge helper.
- Add mode-aware env kwarg resolution for base + `training` or `evaluation`.
- Keep backward compatibility for configs without mode overrides.

**Step 2: Run focused tests**
- Run: `pytest -q tests/test_atari_envs.py tests/test_envs.py`

### Task 3: Document Atari/game protocol usage

**Files:**
- Modify: `README.md`

**Step 1: Add docs**
- Show a single config file defining both train-time and eval-time Atari settings.
- Clarify that evaluation envs can use different sticky-action and reward-clipping behavior without duplicating the full scenario config.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_atari_envs.py tests/test_envs.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
