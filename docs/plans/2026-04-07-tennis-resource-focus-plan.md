# Tennis Resource Focus Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Collapse the Tennis experiment matrix onto two control lines, then add one stronger Tennis-specialized variant for each surviving family.

**Architecture:** Treat `apex_dqn_tennis_stable_lr` and `rainbow_dqn_tennis_no_early_stop` as the only surviving control lines. Stop all other Tennis runs, add one new shaped preset under each family, test the new presets, and launch only the focused four-line comparison set.

**Tech Stack:** Python, YAML configs, pytest, AxiomRL CLI, Gymnasium Atari wrappers

---

### Task 1: Record and stop non-primary Tennis runs

**Files:**
- Run only

**Step 1: Record current status**

Capture latest metrics and PIDs for all Tennis runs that will be stopped:

- `agent57`
- `ppo`
- `impala`
- `r2d2 20M`
- old `apex_main`
- `apex_explore_tuned`
- `apex_reward_lite`
- old `rainbow_20m`
- `rainbow_stable_lr`
- `rainbow_reward_lite`

**Step 2: Stop only the losing lines**

Keep these running:

- `apex_dqn_tennis_stable_lr`
- `rainbow_dqn_tennis_no_early_stop`

Kill the rest.

**Step 3: Verify only the intended control lines remain**

Run:

```bash
ps -eo pid,cmd | rg 'tennis-apex-stable-lr|tennis-rainbow-no-early-stop|agent57|tennis-ppo-main|tennis-impala-main|tennis-r2d2-20m|tennis-apex-explore-tuned|tennis-apex-reward-lite|tennis-rainbow-stable-lr|tennis-rainbow-reward-lite|tennis-rainbow-20m|tennis-apex-main'
```

Expected:
- only `tennis-apex-stable-lr` and `tennis-rainbow-no-early-stop` remain

### Task 2: Add failing tests for the two new specialized variants

**Files:**
- Modify: `tests/test_zoo_presets.py`

**Step 1: Write the failing test**

Add coverage for:

- `apex_dqn_tennis_shaped`
- `rainbow_dqn_tennis_shaped`

Also add a focused manifest test for a compact follow-up comparison manifest that lists exactly:

- `apex_dqn_tennis_stable_lr`
- `apex_dqn_tennis_shaped`
- `rainbow_dqn_tennis_no_early_stop`
- `rainbow_dqn_tennis_shaped`

**Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src /home/kdsoft/.local/bin/pytest -q tests/test_zoo_presets.py -k 'tennis_shaped or tennis_focus'
```

Expected:
- FAIL because the new presets/manifest do not exist yet

### Task 3: Add the `apex_dqn_tennis_shaped` preset

**Files:**
- Create: `configs/apex_dqn/tennis_shaped.yaml`
- Create: `src/rl_training/assets/configs/apex_dqn/tennis_shaped.yaml`
- Create: `zoo/atari/apex_dqn_tennis_shaped.yaml`
- Create: `src/rl_training/assets/zoo/atari/apex_dqn_tennis_shaped.yaml`

**Step 1: Base from the current surviving control**

Use `configs/apex_dqn/tennis_stable_lr.yaml` as the starting point.

**Step 2: Apply stronger Tennis-specific shaping**

Implement a more assertive training-only shaping config than the current `reward_lite`, while keeping evaluation raw:

- keep Atari preprocessing
- keep evaluation `clip_reward: false`
- disable training-time Atari reward clipping
- add stronger but still bounded wrapper-level shaping

**Step 3: Add the thin zoo preset**

Point the zoo preset at the new config with `env_id: ALE/Tennis-v5`.

**Step 4: Run focused preset test**

Run:

```bash
PYTHONPATH=src /home/kdsoft/.local/bin/pytest -q tests/test_zoo_presets.py -k 'apex_dqn_tennis_shaped'
```

Expected:
- PASS

### Task 4: Add the `rainbow_dqn_tennis_shaped` preset

**Files:**
- Create: `configs/rainbow_dqn/tennis_shaped.yaml`
- Create: `src/rl_training/assets/configs/rainbow_dqn/tennis_shaped.yaml`
- Create: `zoo/atari/rainbow_dqn_tennis_shaped.yaml`
- Create: `src/rl_training/assets/zoo/atari/rainbow_dqn_tennis_shaped.yaml`

**Step 1: Base from the current surviving control**

Use `configs/rainbow_dqn/tennis_no_early_stop.yaml` as the starting point.

**Step 2: Apply stronger Tennis-specific shaping**

Implement the same training-only shaping policy:

- no early stopping
- raw evaluation reward
- stronger wrapper-level training shaping

**Step 3: Add the thin zoo preset**

Point the zoo preset at the new config with `env_id: ALE/Tennis-v5`.

**Step 4: Run focused preset test**

Run:

```bash
PYTHONPATH=src /home/kdsoft/.local/bin/pytest -q tests/test_zoo_presets.py -k 'rainbow_dqn_tennis_shaped'
```

Expected:
- PASS

### Task 5: Add the focused four-line manifest

**Files:**
- Create: `zoo/atari/tennis_focus.yaml`
- Create: `src/rl_training/assets/zoo/atari/tennis_focus.yaml`
- Modify: `tests/test_zoo_presets.py`

**Step 1: Implement the manifest**

List exactly these four presets:

- `apex_dqn_tennis_stable_lr`
- `apex_dqn_tennis_shaped`
- `rainbow_dqn_tennis_no_early_stop`
- `rainbow_dqn_tennis_shaped`

**Step 2: Run focused tests**

Run:

```bash
PYTHONPATH=src /home/kdsoft/.local/bin/pytest -q tests/test_zoo_presets.py -k 'tennis_focus or tennis_shaped'
```

Expected:
- PASS

### Task 6: Launch the focused four-line comparison set

**Files:**
- Run only

**Step 1: Keep the two control lines alive**

Do not restart these unless they have already stopped:

- `tennis-apex-stable-lr`
- `tennis-rainbow-no-early-stop`

**Step 2: Start the two shaped lines**

Use isolated output roots such as:

- `runs/tennis-apex-shaped`
- `runs/tennis-rainbow-shaped`

Run:

```bash
./.env/bin/python -s -m rl_training.cli train --config zoo/atari/apex_dqn_tennis_shaped.yaml --output-dir /data/ax/axiomrl/.worktrees/tennis-rainbow/runs/tennis-apex-shaped
./.env/bin/python -s -m rl_training.cli train --config zoo/atari/rainbow_dqn_tennis_shaped.yaml --output-dir /data/ax/axiomrl/.worktrees/tennis-rainbow/runs/tennis-rainbow-shaped
```

**Step 3: Verify startup**

Confirm:

- process exists
- run directory exists
- ALE startup logged
- tensorboard event file exists

### Task 7: Update the comparison template

**Files:**
- Modify: `docs/plans/2026-04-06-tennis-stage1-comparison-template.md`

**Step 1: Replace the six-line matrix with the focused four-line comparison**

Rows should become:

- `apex_dqn_tennis_stable_lr`
- `apex_dqn_tennis_shaped`
- `rainbow_dqn_tennis_no_early_stop`
- `rainbow_dqn_tennis_shaped`

**Step 2: Verify the template still matches the active experiment set**

Manually compare the template rows against the `tennis_focus.yaml` manifest.
