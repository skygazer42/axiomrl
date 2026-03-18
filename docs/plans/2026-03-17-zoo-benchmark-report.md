# Zoo Benchmark Manifest Metadata and Report CLI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the Atari zoo benchmark self-describing and add a CLI report mode that summarizes benchmark-ready runs.

**Architecture:** Extend `zoo/atari/benchmark.yaml` with suite-level protocol and score-normalization metadata. Update `src/rl_training/zoo_cli.py` to support a new `report` format that reads run `metadata.json` files from a runs directory and prints benchmark-relevant fields such as latest return, normalized score, and best checkpoint. Keep the existing `table` and `commands` output stable.

**Tech Stack:** Python 3.10+, YAML manifest parsing, JSON run metadata, existing zoo CLI, pytest.

---

### Task 1: Add failing zoo manifest and report tests

**Files:**
- Modify: `tests/test_zoo_presets.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**
- Add assertions that the Atari benchmark manifest declares `protocol` and `score_normalization`.
- Add a zoo CLI report test that creates a fake run `metadata.json` and checks `--format report --runs-dir ...` prints normalized-score and best-checkpoint fields.

**Step 2: Run test to verify it fails**
- Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Expected: failures because the manifest metadata and report mode do not exist yet.

### Task 2: Implement manifest metadata and report mode

**Files:**
- Modify: `src/rl_training/assets/zoo/atari/benchmark.yaml`
- Modify: `zoo/atari/benchmark.yaml`
- Modify: `src/rl_training/zoo_cli.py`

**Step 1: Write minimal implementation**
- Add suite-level `protocol` and `score_normalization` metadata.
- Add `--format report` and `--runs-dir`.
- Parse run `metadata.json` files and print a concise benchmark summary.

**Step 2: Run focused tests**
- Run: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`

### Task 3: Document zoo benchmark reporting

**Files:**
- Modify: `README.md`
- Modify: `src/rl_training/assets/zoo/README.md`
- Modify: `zoo/README.md`

**Step 1: Add docs**
- Show the new `axiomrl zoo --format report --runs-dir runs` usage.
- Clarify what fields the report reads from benchmark-aware runs.

### Task 4: Verification

**Run:**
- Focused: `pytest -q tests/test_zoo_presets.py tests/test_cli.py`
- Broader: `pytest -q`

**Notes:**
- This plan intentionally omits commits because the session instructions forbid committing unless explicitly requested.
