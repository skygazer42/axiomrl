# Yearly Gap Batch (NAF + D4PG) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a pragmatic first yearly-gap batch by formalizing a 2014-2026 yearly intake roadmap and implementing `NAF` and `D4PG` as the most architecture-compatible missing algorithms from the current 2014-2018 gap window.

**Architecture:** Keep the package shape stable. Reuse the existing continuous-control replay-buffer runtime lane instead of inventing a new execution model. `NAF` gets its own value-based continuous-control model and trainer on top of the current replay infrastructure. `D4PG` reuses the current `DDPG/TD3` continuous-control actor path plus the existing categorical projection ideas from `C51`, while staying explicit as its own non-distributed distributional actor-critic path. Years that require asynchronous or distributed runtime redesign stay documented but deferred.

**Tech Stack:** Python 3.10, PyTorch, Gymnasium, pytest, existing `rl_training` config/checkpoint/API stack.

---

### Task 1: Publish the yearly-gap roadmap for this batch

**Files:**
- Create: `docs/plans/2026-03-13-yearly-gap-batch-naf-mpo.md`
- Modify: `README.md`
- Modify: `docs/plans/2026-03-12-rl-yearly-sourcebook-design.md`

**Step 1: Document the batch scope**
- Update the yearly sourcebook so `2014-2026` reads as a tracked roadmap with explicit status language: `already implemented`, `in this batch`, `deferred for runtime redesign`, `watchlist`.
- Keep `2014-2018` honest: the package already covers much of `2014`, `2015`, and `2017`; this batch is for the remaining architecture-friendly gaps.

**Step 2: Update README direction text**
- Add `NAF` and `D4PG` to the package-direction lists only after implementation is wired.
- Add CLI examples for the two new configs.

**Step 3: Verify docs are internally consistent**
Run:
```bash
rg -n "NAF|MPO|2014|2018" README.md docs/plans/2026-03-12-rl-yearly-sourcebook-design.md
```
Expected: updated wording with no contradictory year/status statements.

### Task 2: Add failing NAF coverage

**Files:**
- Create: `tests/test_naf_update.py`
- Create: `tests/test_naf_trainer_smoke.py`
- Create: `tests/test_naf_reference_script.py`
- Modify: `tests/test_package_api_exports.py`
- Modify: `tests/test_public_api.py`

**Step 1: Write focused failing tests**
- `MLPNAFModel` produces bounded actions and scalar Q-values.
- `naf_loss()` returns named metrics.
- `NAF.update()` returns `UpdateResult` with value-loss style metrics.
- `train_naf()` writes a checkpoint and final eval metrics on `Pendulum-v1`.
- public API exports `NAF`.
- example script runs with a short smoke configuration.

**Step 2: Run tests to verify they fail**
Run:
```bash
pytest -q tests/test_naf_update.py tests/test_naf_trainer_smoke.py tests/test_naf_reference_script.py tests/test_package_api_exports.py tests/test_public_api.py
```
Expected: import or symbol failures for missing `NAF` implementation.

### Task 3: Implement NAF end-to-end

**Files:**
- Create: `src/rl_training/models/mlp_naf.py`
- Create: `src/rl_training/algorithms/naf.py`
- Create: `src/rl_training/runtime/naf_trainer.py`
- Create: `examples/naf_pendulum_reference.py`
- Create: `configs/naf/pendulum.yaml`
- Create: `src/rl_training/assets/configs/naf/pendulum.yaml`
- Modify: `src/rl_training/models/__init__.py`
- Modify: `src/rl_training/algorithms/__init__.py`
- Modify: `src/rl_training/api/algorithms.py`
- Modify: `src/rl_training/api/__init__.py`
- Modify: `src/rl_training/__init__.py`
- Modify: `src/rl_training/experiment/registry.py`
- Modify: `README.md`

**Step 1: Add the model**
- Build an MLP that outputs:
  - deterministic mean action `mu(s)` in `[-1, 1]`
  - scalar state value `V(s)`
  - lower-triangular entries for a positive-definite advantage precision matrix
- Implement helper methods for:
  - `actor(obs)`
  - `q_values(obs, actions)`
  - `state_values(obs)`

**Step 2: Add the algorithm**
- Use a target network, replay buffer, and Bellman update for `Q(s,a)`.
- Optimize only the value-style NAF loss; policy improvement is implicit through `mu(s)`.
- Return named metrics such as `loss`, `target_q_mean`, `q_mean`.

**Step 3: Add the trainer**
- Mirror the `DDPG` trainer surface for config/eval/checkpoint behavior.
- Reuse continuous action scaling helpers from `td3_trainer` or equivalent local helpers.
- Store normalized replay actions in `[-1, 1]` just like other continuous trainers.

**Step 4: Wire package surfaces**
- Add registry load/eval/predict functions.
- Add managed API class `NAF`.
- Add config, packaged asset config, example script, and README commands.

**Step 5: Run targeted NAF tests**
Run:
```bash
pytest -q tests/test_naf_update.py tests/test_naf_trainer_smoke.py tests/test_naf_reference_script.py tests/test_package_api_exports.py tests/test_public_api.py
```
Expected: PASS.

### Task 4: Add failing D4PG coverage

**Files:**
- Create: `tests/test_d4pg_update.py`
- Create: `tests/test_d4pg_trainer_smoke.py`
- Create: `tests/test_d4pg_reference_script.py`
- Modify: `tests/test_package_api_exports.py`
- Modify: `tests/test_public_api.py`

**Step 1: Write focused failing tests**
- `MLPD4PGModel` returns bounded deterministic actions and stochastic samples with log-prob-like outputs.
- `d4pg_loss()` returns critic and actor metric names.
- `D4PG.update()` returns `UpdateResult`.
- `train_d4pg()` writes a checkpoint and eval metrics on `Pendulum-v1`.
- public API exports `D4PG`.
- example script runs with a short smoke configuration.

**Step 2: Run tests to verify they fail**
Run:
```bash
pytest -q tests/test_d4pg_update.py tests/test_d4pg_trainer_smoke.py tests/test_d4pg_reference_script.py tests/test_package_api_exports.py tests/test_public_api.py
```
Expected: import or symbol failures for missing `D4PG` implementation.

### Task 5: Implement D4PG end-to-end

**Files:**
- Create: `src/rl_training/models/mlp_d4pg.py`
- Create: `src/rl_training/algorithms/d4pg.py`
- Create: `src/rl_training/runtime/d4pg_trainer.py`
- Create: `examples/d4pg_pendulum_reference.py`
- Create: `configs/d4pg/pendulum.yaml`
- Create: `src/rl_training/assets/configs/d4pg/pendulum.yaml`
- Modify: `src/rl_training/models/__init__.py`
- Modify: `src/rl_training/algorithms/__init__.py`
- Modify: `src/rl_training/api/algorithms.py`
- Modify: `src/rl_training/api/__init__.py`
- Modify: `src/rl_training/__init__.py`
- Modify: `src/rl_training/experiment/registry.py`
- Modify: `README.md`

**Step 1: Add the model**
- Use a Deterministic tanh actor plus categorical distributional critic over fixed value atoms.
- Expose helpers for deterministic action selection, distribution logits, expected Q-values, and target support access.

**Step 2: Add the algorithm**
- Keep the implementation narrow and explicit:
  - replay-buffer off-policy continuous control
  - target actor/critic network
  - categorical Bellman projection onto fixed atoms
  - actor update driven by the expected Q under the distributional critic
- Return metrics such as `critic_loss`, `actor_loss`, `target_q_mean`, `q_value_mean`.

**Step 3: Add the trainer**
- Mirror the existing `DDPG` / `TD3` trainer shape.
- Keep evaluation deterministic.
- Reuse shared continuous-action scaling helpers.

**Step 4: Wire package surfaces**
- Add registry load/eval/predict functions.
- Add managed API class `D4PG`.
- Add config, packaged asset config, example script, and README commands.

**Step 5: Run targeted MPO tests**
Run:
```bash
pytest -q tests/test_d4pg_update.py tests/test_d4pg_trainer_smoke.py tests/test_d4pg_reference_script.py tests/test_package_api_exports.py tests/test_public_api.py
```
Expected: PASS.

### Task 6: Final regression and consistency verification

**Files:**
- Modify only if verification reveals regressions.

**Step 1: Run focused regression coverage**
Run:
```bash
pytest -q \
  tests/test_naf_update.py \
  tests/test_naf_trainer_smoke.py \
  tests/test_naf_reference_script.py \
  tests/test_d4pg_update.py \
  tests/test_d4pg_trainer_smoke.py \
  tests/test_d4pg_reference_script.py \
  tests/test_package_api_exports.py \
  tests/test_public_api.py \
  tests/test_cli.py
```
Expected: PASS.

**Step 2: Run broader algorithm regressions**
Run:
```bash
pytest -q tests/test_ddpg_update.py tests/test_ddpg_trainer_smoke.py tests/test_sac_trainer_smoke.py tests/test_td3_trainer_smoke.py
```
Expected: PASS.

**Step 3: Summarize remaining deferred yearly gaps**
- Explicitly note that `A3C`, `ACER`, `ACKTR`, `IMPALA`, and `Ape-X` remain deferred because they need async/distributed runtime work instead of just another trainer file.
