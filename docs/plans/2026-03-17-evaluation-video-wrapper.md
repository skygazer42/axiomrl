# Evaluation Video Wrapper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add declarative evaluation-time video recording so Atari and other games can capture reproducible rollout videos directly from the env config.

**Architecture:** Introduce a dedicated video wrapper resolver in `src/axiomrl/envs/video.py` and wire it into `src/axiomrl/envs/factory.py`. Users opt in via `env_kwargs.wrappers.video`, and mode-specific config already added in `env_kwargs.evaluation` keeps video capture isolated to evaluation by default. The wrapper should default to writing under `output_dir/videos/<mode>` and allow simple trigger configuration without affecting existing training code paths.

**Tech Stack:** Python 3.10+, Gymnasium `RecordVideo`, existing env factory/config system, pytest.

---

### Task 1: Add failing video wrapper tests

**Files:**
- Modify: `tests/test_envs.py`

**Step 1: Write the failing tests**
- Add coverage for resolving video wrapper config from `wrappers.video`.
- Add a build-env test proving `evaluation` mode applies `RecordVideo` with the expected folder and trigger settings.
- Add a build-env test proving the same config does not enable video during training when the wrapper is only present under `env_kwargs.evaluation`.

**Step 2: Run test to verify it fails**
- Run: `pytest -q tests/test_envs.py`
- Expected: failures because video wrapper resolution and application do not exist yet.

### Task 2: Implement video wrapper support

**Files:**
- Create: `src/axiomrl/envs/video.py`
- Modify: `src/axiomrl/envs/factory.py`
- Modify: `src/axiomrl/envs/__init__.py`

**Step 1: Write minimal implementation**
- Add a dataclass for video wrapper settings.
- Parse `video_folder`, `name_prefix`, `episode_trigger_every`, `step_trigger_every`, `video_length`, `fps`, and `disable_logger`.
- Apply `gym.wrappers.RecordVideo` with default output under `output_dir/videos/evaluation` or `output_dir/videos/training`.

**Step 2: Run focused tests**
- Run: `pytest -q tests/test_envs.py`

### Task 3: Document video capture usage

**Files:**
- Modify: `README.md`

**Step 1: Add docs**
- Show how to enable evaluation-only video capture with `env_kwargs.evaluation.wrappers.video`.
- Clarify the default output directory and trigger behavior.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_envs.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
