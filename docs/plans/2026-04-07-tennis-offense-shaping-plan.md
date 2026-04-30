# Tennis Offense Shaping Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add offensive placement bonuses on top of the current Tennis event-shaping wrapper and launch two new offense-oriented training lines for `apex_dqn` and `rainbow_dqn`.

**Architecture:** Extend the existing `tennis_events` wrapper with optional offensive bonuses, expose them through YAML configuration, add `event_offense` presets for the two surviving event-shaped families, and run them alongside the current event-shaped control lines.

**Tech Stack:** Python, Gymnasium wrappers, YAML configs, pytest, AxiomRL CLI

---

### Task 1: Add failing tests for offensive Tennis event config

**Files:**
- Modify: `tests/test_atari_envs.py`
- Modify: `tests/test_zoo_presets.py`

**Step 1: Write the failing tests**

Add tests for:

- offensive wrapper config keys:
  - `deep_landing_bonus`
  - `wide_landing_bonus`
- event offense presets:
  - `apex_dqn_tennis_event_offense`
  - `rainbow_dqn_tennis_event_offense`

**Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src:/data/ax/axiomrl/.worktrees/tennis-rainbow/.env-data/lib/python3.10/site-packages /home/kdsoft/.local/bin/pytest -q tests/test_atari_envs.py tests/test_zoo_presets.py -k 'tennis_event_offense or deep_landing_bonus or wide_landing_bonus'
```

Expected:
- FAIL because the offensive config keys / presets do not exist yet

### Task 2: Extend the Tennis event wrapper with offensive placement bonuses

**Files:**
- Modify: `src/axiomrl/envs/tennis_events.py`
- Modify: `tests/test_atari_envs.py`

**Step 1: Extend the config dataclass**

Add:

- `deep_landing_bonus`
- `wide_landing_bonus`

**Step 2: Extend config resolution**

Parse the new bonus values from `wrappers.tennis_events`.

**Step 3: Implement coarse opponent-side placement regions**

When the ball crosses onto the opponent side:

- award `deep_landing_bonus` if the detected x position is clearly deeper into opponent territory
- award `wide_landing_bonus` if the detected x position is clearly near a sideline

**Step 4: Write minimal behavioral tests**

Add one test for deep placement and one for wide placement.

**Step 5: Run tests**

Run:

```bash
PYTHONPATH=src:/data/ax/axiomrl/.worktrees/tennis-rainbow/.env-data/lib/python3.10/site-packages /home/kdsoft/.local/bin/pytest -q tests/test_atari_envs.py -k 'tennis_event'
```

Expected:
- PASS

### Task 3: Add `apex_dqn_tennis_event_offense`

**Files:**
- Create: `configs/apex_dqn/tennis_event_offense.yaml`
- Create: `src/axiomrl/assets/configs/apex_dqn/tennis_event_offense.yaml`
- Create: `zoo/atari/apex_dqn_tennis_event_offense.yaml`
- Create: `src/axiomrl/assets/zoo/atari/apex_dqn_tennis_event_offense.yaml`

**Step 1: Base from the current event-shaped control**

Use `configs/apex_dqn/tennis_event_shaped.yaml`.

**Step 2: Enable offensive bonuses**

Add conservative first-pass values for:

- `deep_landing_bonus`
- `wide_landing_bonus`

**Step 3: Keep evaluation raw**

Do not change evaluation wrappers.

### Task 4: Add `rainbow_dqn_tennis_event_offense`

**Files:**
- Create: `configs/rainbow_dqn/tennis_event_offense.yaml`
- Create: `src/axiomrl/assets/configs/rainbow_dqn/tennis_event_offense.yaml`
- Create: `zoo/atari/rainbow_dqn_tennis_event_offense.yaml`
- Create: `src/axiomrl/assets/zoo/atari/rainbow_dqn_tennis_event_offense.yaml`

**Step 1: Base from the current event-shaped control**

Use `configs/rainbow_dqn/tennis_event_shaped.yaml`.

**Step 2: Enable offensive bonuses**

Match the same offensive shaping concept used for the `apex` family.

**Step 3: Keep evaluation raw**

Do not change evaluation wrappers.

### Task 5: Add focused offense manifest coverage

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Create: `zoo/atari/tennis_offense_focus.yaml`
- Create: `src/axiomrl/assets/zoo/atari/tennis_offense_focus.yaml`

**Step 1: Add manifest tests**

The offense-focused manifest should list exactly:

- `apex_dqn_tennis_event_shaped`
- `apex_dqn_tennis_event_offense`
- `rainbow_dqn_tennis_event_shaped`
- `rainbow_dqn_tennis_event_offense`

**Step 2: Run focused preset tests**

Run:

```bash
PYTHONPATH=src:/data/ax/axiomrl/.worktrees/tennis-rainbow/.env-data/lib/python3.10/site-packages /home/kdsoft/.local/bin/pytest -q tests/test_zoo_presets.py -k 'tennis_event_offense or tennis_offense_focus'
```

Expected:
- PASS

### Task 6: Before training, present the launch plan to the user

**Files:**
- Run only after explicit confirmation

**Step 1: Summarize the intended replacement**

Before starting any new training, tell the user:

- which current lines will remain
- which lines will be replaced or stopped
- which new offense lines will be launched

**Step 2: Wait for explicit go-ahead**

Do not start the offense runs until the user confirms.

### Task 7: Launch the two offense lines after approval

**Files:**
- Run only

**Step 1: Keep the current event-shaped controls**

Keep:

- `apex_dqn_tennis_event_shaped`
- `rainbow_dqn_tennis_event_shaped`

**Step 2: Stop any control lines the user agrees to replace**

Most likely:

- `apex_dqn_tennis_stable_lr`
- `rainbow_dqn_tennis_no_early_stop`

**Step 3: Launch**

```bash
./.env/bin/python -s -m axiomrl.cli train --config zoo/atari/apex_dqn_tennis_event_offense.yaml --output-dir /data/ax/axiomrl/.worktrees/tennis-rainbow/runs/tennis-apex-event-offense
./.env/bin/python -s -m axiomrl.cli train --config zoo/atari/rainbow_dqn_tennis_event_offense.yaml --output-dir /data/ax/axiomrl/.worktrees/tennis-rainbow/runs/tennis-rainbow-event-offense
```

**Step 4: Verify startup**

Confirm:

- processes exist
- run dirs exist
- ALE startup logged
- tensorboard event files exist
