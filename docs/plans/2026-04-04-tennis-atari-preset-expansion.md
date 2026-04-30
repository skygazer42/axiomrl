# Tennis Atari Preset Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add packaged `Agent57` and `EfficientZero` Tennis zoo presets and wire them into the Tennis benchmark manifest.

**Architecture:** Reuse the existing Tennis preset pattern already established by `rainbow_dqn_tennis` and `r2d2_tennis`. Implement the feature entirely through mirrored YAML assets and benchmark-manifest updates so `load_config()` and zoo commands keep working both inside and outside the repository root.

**Tech Stack:** YAML configs, Python, pytest, packaged asset resolution

---

### Task 1: Add failing Tennis preset coverage

**Files:**
- Modify: `tests/test_zoo_presets.py`

**Step 1: Write the failing test**

Add focused tests that expect:

- `zoo/atari/tennis_benchmark.yaml` to list `agent57_tennis` and `efficientzero_tennis`
- `load_config("zoo/atari/agent57_tennis.yaml")` to resolve outside the repo root
- `load_config("zoo/atari/efficientzero_tennis.yaml")` to resolve outside the repo root
- both packaged presets to inherit Tennis protocol defaults

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_zoo_presets.py -k "tennis"`

Expected: FAIL because the new preset files and manifest entries do not exist yet.

**Step 3: Write minimal implementation**

Do not implement in this task.

**Step 4: Run test to verify it passes**

Handled after Task 3.

**Step 5: Commit**

Skip commit in this session unless explicitly requested.

### Task 2: Add Tennis config and preset YAMLs

**Files:**
- Create: `configs/agent57/tennis_atari.yaml`
- Create: `configs/efficientzero/tennis_atari.yaml`
- Create: `src/axiomrl/assets/configs/agent57/tennis_atari.yaml`
- Create: `src/axiomrl/assets/configs/efficientzero/tennis_atari.yaml`
- Create: `zoo/atari/agent57_tennis.yaml`
- Create: `zoo/atari/efficientzero_tennis.yaml`
- Create: `src/axiomrl/assets/zoo/atari/agent57_tennis.yaml`
- Create: `src/axiomrl/assets/zoo/atari/efficientzero_tennis.yaml`

**Step 1: Write the failing test**

Use the new tests from Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_zoo_presets.py -k "tennis"`

**Step 3: Write minimal implementation**

Copy the Breakout configs for each algorithm, switch `env_id` to `ALE/Tennis-v5`, and align the environment protocol with the existing Tennis configs already in the tree.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_zoo_presets.py -k "tennis"`

**Step 5: Commit**

Skip commit in this session unless explicitly requested.

### Task 3: Register both presets in the Tennis benchmark

**Files:**
- Modify: `zoo/atari/tennis_benchmark.yaml`
- Modify: `src/axiomrl/assets/zoo/atari/tennis_benchmark.yaml`

**Step 1: Write the failing test**

Use the manifest-name assertions from Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_zoo_presets.py -k "tennis"`

**Step 3: Write minimal implementation**

Append both new presets with concise descriptions to the Tennis benchmark manifest and its packaged asset mirror.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_zoo_presets.py -k "tennis"`

**Step 5: Commit**

Skip commit in this session unless explicitly requested.

### Task 4: Verify no config-regression spillover

**Files:**
- Verify: `tests/test_zoo_presets.py`
- Verify: `tests/test_cli_config.py`

**Step 1: Write the failing test**

Not needed; verification only.

**Step 2: Run test to verify it fails**

Not applicable.

**Step 3: Write minimal implementation**

None.

**Step 4: Run test to verify it passes**

Run:

- `pytest -q tests/test_zoo_presets.py -k "tennis or packaged_zoo_preset"`
- `pytest -q tests/test_cli_config.py -k "linked_zoo_preset or packaged_repo_config"`

**Step 5: Commit**

Skip commit in this session unless explicitly requested.
