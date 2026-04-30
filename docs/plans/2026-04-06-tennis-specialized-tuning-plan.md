# Tennis Specialized Tuning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and launch a focused Stage 1 Tennis tuning matrix centered on `apex_dqn` and `rainbow_dqn`, while de-prioritizing weak lines and preserving comparability across runs.

**Architecture:** Keep the existing AxiomRL training stack and add six new experiment presets: three for `apex_dqn`, three for `rainbow_dqn`. Reuse the current Atari/Tennis config and zoo preset structure, encode tuning choices in YAML, test that the new presets resolve correctly, then stop weak foreground runs and launch the new Stage 1 matrix with isolated output directories.

**Tech Stack:** Python, YAML configs, pytest, AxiomRL CLI, Gymnasium Atari wrappers

---

### Task 1: Add failing preset tests for the Stage 1 tuning matrix

**Files:**
- Modify: `tests/test_zoo_presets.py`

**Step 1: Write the failing test**

Extend Tennis preset coverage to include these six new Stage 1 presets:

- `apex_dqn_tennis_stable_lr`
- `apex_dqn_tennis_explore_tuned`
- `apex_dqn_tennis_reward_lite`
- `rainbow_dqn_tennis_stable_lr`
- `rainbow_dqn_tennis_no_early_stop`
- `rainbow_dqn_tennis_reward_lite`

Add manifest assertions and packaged preset-loading coverage.

**Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src /home/kdsoft/.local/bin/pytest -q tests/test_zoo_presets.py -k 'tennis and (apex_dqn_tennis_stable_lr or rainbow_dqn_tennis_no_early_stop or reward_lite)'
```

Expected:
- FAIL because the new preset/config files do not exist yet

**Step 3: Commit**

```bash
git add tests/test_zoo_presets.py
git commit -m "test: cover tennis tuning preset matrix"
```

### Task 2: Add `apex_dqn` Stage 1 tuning configs and presets

**Files:**
- Create: `configs/apex_dqn/tennis_stable_lr.yaml`
- Create: `configs/apex_dqn/tennis_explore_tuned.yaml`
- Create: `configs/apex_dqn/tennis_reward_lite.yaml`
- Create: `src/axiomrl/assets/configs/apex_dqn/tennis_stable_lr.yaml`
- Create: `src/axiomrl/assets/configs/apex_dqn/tennis_explore_tuned.yaml`
- Create: `src/axiomrl/assets/configs/apex_dqn/tennis_reward_lite.yaml`
- Create: `zoo/atari/apex_dqn_tennis_stable_lr.yaml`
- Create: `zoo/atari/apex_dqn_tennis_explore_tuned.yaml`
- Create: `zoo/atari/apex_dqn_tennis_reward_lite.yaml`
- Create: `src/axiomrl/assets/zoo/atari/apex_dqn_tennis_stable_lr.yaml`
- Create: `src/axiomrl/assets/zoo/atari/apex_dqn_tennis_explore_tuned.yaml`
- Create: `src/axiomrl/assets/zoo/atari/apex_dqn_tennis_reward_lite.yaml`

**Step 1: Write the failing test**

Use the Task 1 failing tests as the red state for these files.

**Step 2: Implement `stable-lr`**

Create a Tennis config derived from the current `apex_dqn` Tennis baseline, but make optimization calmer:

- lower learning rate
- slower target update interval
- leave reward semantics unchanged

**Step 3: Implement `explore-tuned`**

Create a Tennis config derived from the same baseline, but retune exploration:

- adjust `actor_epsilon_base`
- adjust `actor_epsilon_alpha`
- optionally adjust replay beta schedule conservatively

**Step 4: Implement `reward-lite`**

Create a Tennis config derived from `stable-lr`, with training-only reward shaping:

- keep evaluation rewards unshaped
- use only a small `wrappers.reward.step_penalty` or similarly light shaping

**Step 5: Add zoo preset wrappers**

Add one thin zoo YAML per config so each preset is loadable through `load_config("zoo/atari/...")`.

**Step 6: Commit**

```bash
git add configs/apex_dqn/tennis_stable_lr.yaml configs/apex_dqn/tennis_explore_tuned.yaml configs/apex_dqn/tennis_reward_lite.yaml
git add src/axiomrl/assets/configs/apex_dqn/tennis_stable_lr.yaml src/axiomrl/assets/configs/apex_dqn/tennis_explore_tuned.yaml src/axiomrl/assets/configs/apex_dqn/tennis_reward_lite.yaml
git add zoo/atari/apex_dqn_tennis_stable_lr.yaml zoo/atari/apex_dqn_tennis_explore_tuned.yaml zoo/atari/apex_dqn_tennis_reward_lite.yaml
git add src/axiomrl/assets/zoo/atari/apex_dqn_tennis_stable_lr.yaml src/axiomrl/assets/zoo/atari/apex_dqn_tennis_explore_tuned.yaml src/axiomrl/assets/zoo/atari/apex_dqn_tennis_reward_lite.yaml
git commit -m "feat: add apex tennis tuning presets"
```

### Task 3: Add `rainbow_dqn` Stage 1 tuning configs and presets

**Files:**
- Create: `configs/rainbow_dqn/tennis_stable_lr.yaml`
- Create: `configs/rainbow_dqn/tennis_no_early_stop.yaml`
- Create: `configs/rainbow_dqn/tennis_reward_lite.yaml`
- Create: `src/axiomrl/assets/configs/rainbow_dqn/tennis_stable_lr.yaml`
- Create: `src/axiomrl/assets/configs/rainbow_dqn/tennis_no_early_stop.yaml`
- Create: `src/axiomrl/assets/configs/rainbow_dqn/tennis_reward_lite.yaml`
- Create: `zoo/atari/rainbow_dqn_tennis_stable_lr.yaml`
- Create: `zoo/atari/rainbow_dqn_tennis_no_early_stop.yaml`
- Create: `zoo/atari/rainbow_dqn_tennis_reward_lite.yaml`
- Create: `src/axiomrl/assets/zoo/atari/rainbow_dqn_tennis_stable_lr.yaml`
- Create: `src/axiomrl/assets/zoo/atari/rainbow_dqn_tennis_no_early_stop.yaml`
- Create: `src/axiomrl/assets/zoo/atari/rainbow_dqn_tennis_reward_lite.yaml`

**Step 1: Write the failing test**

Reuse Task 1 failing tests.

**Step 2: Implement `stable-lr`**

Create a calmer Tennis Rainbow config:

- lower learning rate
- keep the rest of the Rainbow baseline close to the current Tennis setup

**Step 3: Implement `no-early-stop`**

Create a long-run Tennis Rainbow config with:

- no `early_stopping` block
- unchanged task reward

**Step 4: Implement `reward-lite`**

Create a Tennis Rainbow config with:

- the stable optimization setup
- only light training-time reward shaping
- raw evaluation reward

**Step 5: Add zoo preset wrappers**

Mirror the `apex_dqn` approach with thin zoo YAML files.

**Step 6: Commit**

```bash
git add configs/rainbow_dqn/tennis_stable_lr.yaml configs/rainbow_dqn/tennis_no_early_stop.yaml configs/rainbow_dqn/tennis_reward_lite.yaml
git add src/axiomrl/assets/configs/rainbow_dqn/tennis_stable_lr.yaml src/axiomrl/assets/configs/rainbow_dqn/tennis_no_early_stop.yaml src/axiomrl/assets/configs/rainbow_dqn/tennis_reward_lite.yaml
git add zoo/atari/rainbow_dqn_tennis_stable_lr.yaml zoo/atari/rainbow_dqn_tennis_no_early_stop.yaml zoo/atari/rainbow_dqn_tennis_reward_lite.yaml
git add src/axiomrl/assets/zoo/atari/rainbow_dqn_tennis_stable_lr.yaml src/axiomrl/assets/zoo/atari/rainbow_dqn_tennis_no_early_stop.yaml src/axiomrl/assets/zoo/atari/rainbow_dqn_tennis_reward_lite.yaml
git commit -m "feat: add rainbow tennis tuning presets"
```

### Task 4: Add the Stage 1 tuning manifest

**Files:**
- Create: `zoo/atari/tennis_tuning_stage1.yaml`
- Create: `src/axiomrl/assets/zoo/atari/tennis_tuning_stage1.yaml`
- Modify: `tests/test_zoo_presets.py`

**Step 1: Write the failing test**

Add a test that the Stage 1 manifest lists exactly the six tuning presets.

**Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src /home/kdsoft/.local/bin/pytest -q tests/test_zoo_presets.py -k 'tennis_tuning_stage1'
```

Expected:
- FAIL because the manifest does not exist yet

**Step 3: Implement the manifest**

List the six new tuning presets with short descriptions that encode the intent of each variant.

**Step 4: Run focused preset tests**

Run:

```bash
PYTHONPATH=src /home/kdsoft/.local/bin/pytest -q tests/test_zoo_presets.py -k 'tennis and (tuning or reward_lite or no_early_stop or stable_lr or explore_tuned)'
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add tests/test_zoo_presets.py zoo/atari/tennis_tuning_stage1.yaml src/axiomrl/assets/zoo/atari/tennis_tuning_stage1.yaml
git commit -m "feat: add tennis stage1 tuning manifest"
```

### Task 5: Shut down weak primary runs and preserve state

**Files:**
- Run only

**Step 1: Record current status**

Capture current PIDs and latest metrics for:

- `agent57`
- `ppo`
- `impala`

**Step 2: Stop only the weak primary runs**

Terminate those long-running jobs cleanly, leaving:

- `apex_dqn`
- `rainbow_dqn` lines
- `r2d2 20M` observer

**Step 3: Verify only the intended processes remain**

Run:

```bash
ps -eo pid,cmd | rg 'tennis-apex-main|tennis-rainbow|tennis-r2d2-20m|agent57_tennis|tennis-ppo-main|tennis-impala-main'
```

Expected:
- `agent57`, `ppo`, and `impala` absent
- `apex_dqn`, `rainbow_dqn`, and `r2d2-20m` still present if intended

### Task 6: Launch the Stage 1 matrix

**Files:**
- Run only

**Step 1: Create isolated output roots**

Use one output root per variant, for example:

- `runs/tennis-apex-stable-lr`
- `runs/tennis-apex-explore-tuned`
- `runs/tennis-apex-reward-lite`
- `runs/tennis-rainbow-stable-lr`
- `runs/tennis-rainbow-no-early-stop`
- `runs/tennis-rainbow-reward-lite`

**Step 2: Launch the six jobs**

Start each with the package CLI:

```bash
./.env/bin/python -s -m axiomrl.cli train --config zoo/atari/<preset>.yaml --output-dir /data/ax/axiomrl/.worktrees/tennis-rainbow/runs/<variant-dir>
```

**Step 3: Verify startup**

For each run, confirm:

- process exists
- run directory created
- ALE startup logged
- TensorBoard event file created

### Task 7: Add a comparison worksheet

**Files:**
- Create: `docs/plans/2026-04-06-tennis-stage1-comparison-template.md`

**Step 1: Write a tiny comparison template**

Include rows for:

- preset name
- latest eval
- best eval
- last 3 eval points
- keep / drop decision

**Step 2: Commit**

```bash
git add docs/plans/2026-04-06-tennis-stage1-comparison-template.md
git commit -m "docs: add tennis stage1 comparison template"
```
