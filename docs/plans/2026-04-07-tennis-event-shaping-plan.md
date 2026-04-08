# Tennis Event Shaping Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reusable Tennis event-based training reward wrapper, wire it into `apex_dqn` and `rainbow_dqn` presets, and launch a clean control-vs-event-shaped comparison.

**Architecture:** Implement a new Tennis-specific environment wrapper in the `envs` layer that adds auxiliary rewards during training only. Expose it through YAML-configurable wrappers, add event-shaped presets for the surviving `apex_dqn` and `rainbow_dqn` control lines, verify preset loading, then launch the two new shaped runs alongside the existing controls.

**Tech Stack:** Python, Gymnasium wrappers, YAML configs, pytest, AxiomRL CLI

---

### Task 1: Add failing tests for Tennis event wrapper config resolution

**Files:**
- Modify: `tests/test_atari_envs.py`
- Modify: `tests/test_zoo_presets.py`

**Step 1: Write the failing tests**

Add tests for:

- a new Tennis wrapper config under `wrappers.tennis_events`
- event-shaped presets:
  - `apex_dqn_tennis_event_shaped`
  - `rainbow_dqn_tennis_event_shaped`
- a focused manifest:
  - `apex_dqn_tennis_stable_lr`
  - `apex_dqn_tennis_event_shaped`
  - `rainbow_dqn_tennis_no_early_stop`
  - `rainbow_dqn_tennis_event_shaped`

**Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src /home/kdsoft/.local/bin/pytest -q tests/test_atari_envs.py tests/test_zoo_presets.py -k 'tennis_event or tennis_event_shaped or tennis_focus'
```

Expected:
- FAIL because the wrapper and presets do not exist yet

### Task 2: Implement the Tennis event reward wrapper

**Files:**
- Create: `src/rl_training/envs/tennis_events.py`
- Modify: `src/rl_training/envs/factory.py`
- Modify: `src/rl_training/envs/__init__.py`

**Step 1: Implement a wrapper config dataclass**

Add a compact config object that supports:

- `rally_survival_bonus`
- `net_cross_bonus`
- `successful_return_bonus`
- `failure_penalty`

**Step 2: Implement config resolution**

Read the wrapper config from `wrappers.tennis_events` and return `None` when not requested.

**Step 3: Implement the wrapper**

Create a Tennis-specific wrapper that:

- inspects observations frame-by-frame
- tracks a lightweight heuristic notion of ball motion / side-of-court
- adds auxiliary reward during training steps only
- leaves evaluation untouched when not enabled

**Step 4: Integrate into the environment factory**

Apply the new wrapper after Atari observation preprocessing and before generic reward wrappers if that ordering best preserves event detection on processed observations.

**Step 5: Run focused tests**

Run:

```bash
PYTHONPATH=src /home/kdsoft/.local/bin/pytest -q tests/test_atari_envs.py -k 'tennis_event'
```

Expected:
- PASS

### Task 3: Add `apex_dqn_tennis_event_shaped`

**Files:**
- Create: `configs/apex_dqn/tennis_event_shaped.yaml`
- Create: `src/rl_training/assets/configs/apex_dqn/tennis_event_shaped.yaml`
- Create: `zoo/atari/apex_dqn_tennis_event_shaped.yaml`
- Create: `src/rl_training/assets/zoo/atari/apex_dqn_tennis_event_shaped.yaml`

**Step 1: Base from the surviving control**

Use `configs/apex_dqn/tennis_stable_lr.yaml` as the exact starting point.

**Step 2: Enable Tennis event shaping for training only**

Set training wrapper config for `tennis_events` with conservative first-pass magnitudes and keep evaluation raw.

**Step 3: Run focused preset test**

Run:

```bash
PYTHONPATH=src /home/kdsoft/.local/bin/pytest -q tests/test_zoo_presets.py -k 'apex_dqn_tennis_event_shaped'
```

Expected:
- PASS

### Task 4: Add `rainbow_dqn_tennis_event_shaped`

**Files:**
- Create: `configs/rainbow_dqn/tennis_event_shaped.yaml`
- Create: `src/rl_training/assets/configs/rainbow_dqn/tennis_event_shaped.yaml`
- Create: `zoo/atari/rainbow_dqn_tennis_event_shaped.yaml`
- Create: `src/rl_training/assets/zoo/atari/rainbow_dqn_tennis_event_shaped.yaml`

**Step 1: Base from the surviving control**

Use `configs/rainbow_dqn/tennis_no_early_stop.yaml` as the exact starting point.

**Step 2: Enable Tennis event shaping for training only**

Use the same wrapper family and keep evaluation raw.

**Step 3: Run focused preset test**

Run:

```bash
PYTHONPATH=src /home/kdsoft/.local/bin/pytest -q tests/test_zoo_presets.py -k 'rainbow_dqn_tennis_event_shaped'
```

Expected:
- PASS

### Task 5: Add the focused four-line manifest

**Files:**
- Create: `zoo/atari/tennis_focus.yaml`
- Create: `src/rl_training/assets/zoo/atari/tennis_focus.yaml`
- Modify: `tests/test_zoo_presets.py`
- Modify: `docs/plans/2026-04-06-tennis-stage1-comparison-template.md`

**Step 1: Implement the focused manifest**

List exactly:

- `apex_dqn_tennis_stable_lr`
- `apex_dqn_tennis_event_shaped`
- `rainbow_dqn_tennis_no_early_stop`
- `rainbow_dqn_tennis_event_shaped`

**Step 2: Update the comparison template**

Replace the old broad Stage 1 matrix with the focused four-line comparison.

**Step 3: Run focused tests**

Run:

```bash
PYTHONPATH=src /home/kdsoft/.local/bin/pytest -q tests/test_zoo_presets.py -k 'tennis_focus or tennis_event_shaped'
```

Expected:
- PASS

### Task 6: Stop losing lines and launch the event-shaped comparison

**Files:**
- Run only

**Step 1: Preserve the two control lines**

Keep:

- `apex_dqn_tennis_stable_lr`
- `rainbow_dqn_tennis_no_early_stop`

**Step 2: Stop non-winning auxiliary lines**

Stop:

- `apex_dqn_tennis_reward_lite`

Leave any already-dead lines alone.

**Step 3: Launch the two event-shaped lines**

Run:

```bash
./.env/bin/python -s -m rl_training.cli train --config zoo/atari/apex_dqn_tennis_event_shaped.yaml --output-dir /data/ax/axiomrl/.worktrees/tennis-rainbow/runs/tennis-apex-event-shaped
./.env/bin/python -s -m rl_training.cli train --config zoo/atari/rainbow_dqn_tennis_event_shaped.yaml --output-dir /data/ax/axiomrl/.worktrees/tennis-rainbow/runs/tennis-rainbow-event-shaped
```

**Step 4: Verify startup**

Confirm:

- process exists
- run dir exists
- log shows ALE startup
- tensorboard event file exists

### Task 7: Compare first checkpoint behavior

**Files:**
- Run only

**Step 1: Wait for the first common evaluation point**

Gather the first `metrics.jsonl` entries for all four active lines.

**Step 2: Fill the comparison template**

Record:

- latest eval
- best eval
- recent trend
- keep / drop

**Step 3: Decide continuation**

Pick the strongest `apex` line and the strongest `rainbow` line for the next budget extension.
