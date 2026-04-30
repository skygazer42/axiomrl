# Double DQN And Dueling DQN Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `double_dqn` and `dueling_dqn` as first-class algorithms with configs, examples, registry wiring, public API exports, and tests.

**Architecture:** Reuse the existing DQN runtime and checkpoint flow instead of cloning a new trainer per variant. Add a configurable DQN core that supports Double DQN target selection and a dedicated dueling Q-network implementation, then expose both variants through the existing registry/API layers.

**Tech Stack:** Python 3.10, PyTorch, Gymnasium, pytest, setuptools package layout

---

### Task 1: Add failing coverage for algorithm variants

**Files:**
- Modify: `tests/test_dqn_update.py`
- Modify: `tests/test_dqn_trainer_smoke.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_package_api_exports.py`
- Create: `tests/test_double_dqn_reference_script.py`
- Create: `tests/test_dueling_dqn_reference_script.py`

**Step 1: Write the failing test**

Add tests that expect:
- `double_dqn` update logic to use online argmax + target gather
- `dueling_dqn` network outputs the correct shape
- `double_dqn` and `dueling_dqn` smoke training paths to produce checkpoints
- public API exports `DoubleDQN` and `DuelingDQN`
- example scripts for both variants to run successfully

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dqn_update.py tests/test_dqn_trainer_smoke.py tests/test_cli.py tests/test_public_api.py tests/test_package_api_exports.py tests/test_double_dqn_reference_script.py tests/test_dueling_dqn_reference_script.py -q`

Expected: FAIL because the new algorithms, exports, and scripts do not exist yet.

### Task 2: Implement reusable DQN variant support

**Files:**
- Modify: `src/axiomrl/algorithms/dqn.py`
- Modify: `src/axiomrl/runtime/dqn_trainer.py`
- Create: `src/axiomrl/models/mlp_dueling_q_network.py`
- Modify: `src/axiomrl/models/__init__.py`

**Step 1: Write minimal implementation**

Implement:
- `DQN(double_q: bool = False)` for Double DQN target computation
- `MLPDuelingQNetwork` with shared trunk plus value/advantage heads
- trainer-side network selection based on `config.algo`

**Step 2: Run focused tests**

Run: `pytest tests/test_dqn_update.py tests/test_dqn_trainer_smoke.py -q`

Expected: PASS

### Task 3: Wire new variants into registry and public API

**Files:**
- Modify: `src/axiomrl/experiment/registry.py`
- Modify: `src/axiomrl/api/algorithms.py`
- Modify: `src/axiomrl/api/__init__.py`
- Modify: `src/axiomrl/algorithms/__init__.py`
- Modify: `src/axiomrl/__init__.py`

**Step 1: Write minimal implementation**

Expose:
- registry entries for `double_dqn` and `dueling_dqn`
- managed API classes `DoubleDQN` and `DuelingDQN`
- top-level package exports and compatibility aliases

**Step 2: Run focused tests**

Run: `pytest tests/test_cli.py tests/test_public_api.py tests/test_package_api_exports.py -q`

Expected: PASS

### Task 4: Add configs and runnable examples

**Files:**
- Create: `configs/double_dqn/cartpole.yaml`
- Create: `configs/dueling_dqn/cartpole.yaml`
- Create: `examples/double_dqn_cartpole_reference.py`
- Create: `examples/dueling_dqn_cartpole_reference.py`
- Modify: `README.md`

**Step 1: Write minimal implementation**

Add two CartPole configs and two reference scripts matching the existing project style, then update README algorithm list to mention the added variants.

**Step 2: Run focused tests**

Run: `pytest tests/test_double_dqn_reference_script.py tests/test_dueling_dqn_reference_script.py -q`

Expected: PASS

### Task 5: Verify end-to-end

**Files:**
- Verify only

**Step 1: Run the targeted new coverage**

Run: `pytest tests/test_dqn_update.py tests/test_dqn_trainer_smoke.py tests/test_cli.py tests/test_public_api.py tests/test_package_api_exports.py tests/test_double_dqn_reference_script.py tests/test_dueling_dqn_reference_script.py -q`

Expected: PASS

**Step 2: Run the full suite**

Run: `pytest -q`

Expected: PASS
