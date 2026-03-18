# Zoo Manifest Config Plumbing and Aggregate Reporting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Atari zoo presets benchmark-aware by automatically injecting manifest protocol and score-normalization defaults into resolved training configs, and extend the zoo report with aggregate multi-run summaries.

**Architecture:** Keep the existing preset files as lightweight launch aliases, but when `load_config(...)` resolves a zoo preset it should also discover the sibling `benchmark.yaml`, merge protocol defaults into `env_kwargs.training` / `env_kwargs.evaluation`, and merge manifest benchmark metadata into `benchmark` while preserving explicit config overrides. Extend benchmark normalization to resolve named score references such as `atari_breakout_reference`, then update the zoo report to consume run metadata and print both per-run lines and grouped aggregate summaries.

**Tech Stack:** Python 3.10+, YAML manifests, existing `TrainConfig` loader, benchmark utilities, pytest.

---

### Task 1: Add failing tests for manifest defaults and score references

**Files:**
- Modify: `tests/test_benchmarking.py`
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add a benchmark utility test proving `score_normalization.source: atari_breakout_reference` resolves to explicit random/human scores.
- Add a zoo preset loading test proving `load_config("zoo/atari/dqn_breakout.yaml")` injects:
  - benchmark suite / preset / protocol metadata
  - score-normalization defaults
  - evaluation protocol overrides such as sticky-action evaluation
- Add a zoo report test with multiple seeds that checks the new aggregate summary line.

**Step 2: Run tests to verify they fail**
- Run: `pytest -q tests/test_benchmarking.py tests/test_zoo_presets.py tests/test_cli.py`
- Expected: failures because named score references, manifest config plumbing, and aggregate summaries do not exist yet.

### Task 2: Implement score-reference resolution and manifest-aware config loading

**Files:**
- Modify: `src/rl_training/experiment/benchmarking.py`
- Modify: `src/rl_training/zoo_cli.py`
- Modify: `src/rl_training/cli.py`

**Step 1: Write minimal implementation**
- Add a small score-reference registry for named benchmark sources.
- Allow score-normalization settings to resolve scores from `source` / `game` defaults.
- When a linked zoo preset is loaded, discover its manifest and merge:
  - `protocol.training` into `env_kwargs.training`
  - `protocol.evaluation` into `env_kwargs.evaluation`
  - suite / preset / protocol benchmark metadata into `benchmark`
  - resolved score-normalization defaults into `benchmark.score_normalization`

**Step 2: Run focused tests**
- Run: `pytest -q tests/test_benchmarking.py tests/test_zoo_presets.py tests/test_cli.py`

### Task 3: Extend zoo report output with aggregates

**Files:**
- Modify: `src/rl_training/zoo_cli.py`

**Step 1: Write minimal implementation**
- Preserve the existing per-run report lines.
- Add aggregate grouping by `(algo, env_id)` with sorted seeds and summary means for latest / best benchmark metrics.

**Step 2: Run focused tests**
- Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

### Task 4: Document benchmark-aware zoo presets

**Files:**
- Modify: `README.md`
- Modify: `zoo/README.md`
- Modify: `src/rl_training/assets/zoo/README.md`

**Step 1: Add docs**
- Explain that `axiomrl train --config zoo/...` now inherits manifest benchmark defaults and train/eval protocol defaults.
- Show how `axiomrl zoo --format report` prints both per-run and aggregate summaries.

### Task 5: Verification

**Run:**
- Focused: `pytest -q tests/test_benchmarking.py tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
