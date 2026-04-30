# Algorithm Expansion (FQF, MBPO, Dreamer, GAIL) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add four new algorithms (`fqf`, `mbpo`, `dreamer`, `gail`) to the RL Training/AxiomRL codebase with minimal-but-real implementations, configs, examples, checkpoint support, and smoke tests.

**Architecture:** Reuse existing trainer/workflow patterns:
- `fqf` extends the existing `dqn_trainer` distributional lane (alongside `qr_dqn` and `iqn`).
- `mbpo` reuses MOPO’s ensemble dynamics model but runs online like `sac_trainer` (real replay + model-rollout replay).
- `gail` layers a discriminator on top of the existing PPO-style rollout + update loop.
- `dreamer` introduces a small world-model runtime (pixels + discrete actions) and a sequence replay buffer integration.

**Tech Stack:** Python 3.10+, PyTorch, Gymnasium, existing `axiomrl.runtime.*` trainers, existing checkpointing/workflows.

---

### Task 1: FQF (Fully-parameterized Quantile Function)

**Files:**
- Create: `src/axiomrl/models/mlp_fqf_network.py`
- Create: `src/axiomrl/algorithms/fqf.py`
- Modify: `src/axiomrl/runtime/dqn_trainer.py`
- Modify: `src/axiomrl/experiment/registry.py`
- Modify: `src/axiomrl/api/algorithms.py`
- Modify: `src/axiomrl/__init__.py`
- Create: `configs/fqf/cartpole.yaml`
- Create: `src/axiomrl/assets/configs/fqf/cartpole.yaml`
- Create: `examples/fqf_cartpole_reference.py`
- Test: `tests/test_fqf_update.py`
- Test: `tests/test_fqf_reference_script.py`
- (Optional) Test: `tests/test_dqn_trainer_smoke.py` (only if needed for coverage)

**Step 1: Write the failing tests**
- Add `tests/test_fqf_update.py`:
  - network forward shapes (`quantiles_hat`, `taus`, `tau_hats`)
  - algorithm `.update()` returns `UpdateResult` with expected metric keys
  - `act()` returns valid discrete actions.
- Add `tests/test_fqf_reference_script.py` to run the reference script with tiny timesteps.

**Step 2: Run tests to verify RED**
- Run: `pytest -q tests/test_fqf_update.py -q`
- Run: `pytest -q tests/test_fqf_reference_script.py -q`
- Expected: failures due to missing modules / missing algorithm wiring.

**Step 3: Minimal implementation**
- Implement `MLPFQFNetwork` (state-only fraction proposal, IQN-style quantile embedding).
- Implement `FQF` update:
  - quantile regression loss with learned `tau_hats`
  - fraction loss + entropy regularization
  - target syncing and `last_td_errors` for prioritized replay support.

**Step 4: GREEN**
- Run the same tests; confirm pass.

**Step 5: Wire into runtime surfaces**
- Update `dqn_trainer` to build `fqf`.
- Update `registry.py`:
  - add loader/eval/predict entries for `fqf`.
- Update API exports so `from axiomrl import FQF` (and `ManagedAlgorithm`) works.

---

### Task 2: MBPO (Model-Based Policy Optimization)

**Files:**
- Create: `src/axiomrl/algorithms/mbpo.py`
- Create: `src/axiomrl/runtime/mbpo_trainer.py`
- Modify: `src/axiomrl/experiment/registry.py`
- Modify: `src/axiomrl/api/algorithms.py`
- Modify: `src/axiomrl/__init__.py`
- Create: `configs/mbpo/pendulum.yaml`
- Create: `src/axiomrl/assets/configs/mbpo/pendulum.yaml`
- Create: `examples/mbpo_pendulum_reference.py`
- Test: `tests/test_mbpo_update.py`
- Test: `tests/test_mbpo_trainer_smoke.py`
- Test: `tests/test_mbpo_reference_script.py`

**Step 1: Write the failing tests**
- `tests/test_mbpo_update.py`: ensure model update + policy update return metrics.
- `tests/test_mbpo_trainer_smoke.py`: run short Pendulum job, verify checkpoint + eval metrics.
- `tests/test_mbpo_reference_script.py`: run example script quickly.

**Step 2: Verify RED**
- Run those tests; expect missing modules and missing registry spec.

**Step 3: Minimal implementation**
- `MBPO` algorithm:
  - reuse `MLPMOPOEnsembleModel` and `MLPSACModel`
  - no reward penalty (unlike MOPO).
- `train_mbpo` runtime:
  - collect real transitions (like SAC)
  - periodically fit dynamics model on real replay
  - periodically refresh synthetic buffer via short rollouts
  - update SAC on mixed real/synthetic batches
  - checkpoint: store both buffers and algorithm state.

**Step 4: GREEN**
- Run targeted tests; confirm pass.

---

### Task 3: GAIL (Generative Adversarial Imitation Learning)

**Files:**
- Create: `src/axiomrl/models/mlp_gail_discriminator.py`
- Create: `src/axiomrl/algorithms/gail.py`
- Create: `src/axiomrl/runtime/gail_trainer.py`
- Modify: `src/axiomrl/experiment/registry.py`
- Modify: `src/axiomrl/api/algorithms.py`
- Modify: `src/axiomrl/__init__.py`
- Create: `configs/gail/cartpole.yaml`
- Create: `src/axiomrl/assets/configs/gail/cartpole.yaml`
- Create: `examples/gail_cartpole_reference.py`
- Test: `tests/test_gail_trainer_smoke.py`
- Test: `tests/test_gail_reference_script.py`

**Step 1: Write failing smoke tests**
- Trainer runs a few updates, writes checkpoint, returns eval metrics.

**Step 2: Verify RED**
- Run the tests; expect missing trainer and registry wiring.

**Step 3: Minimal implementation**
- Discriminator over `(obs, one_hot(action))` for discrete tasks.
- Replace env reward with `softplus(discriminator_logits)` for PPO returns.
- Support expert dataset sources:
  - `expert_dataset_kind: random` (generated from env) for tests/smoke
  - `expert_dataset_kind: npz/minari` via `load_transition_dataset` for real use.

---

### Task 4: Dreamer (Pixels, Atari-style discrete actions)

**Files:**
- Create: `src/axiomrl/models/dreamer.py`
- Create: `src/axiomrl/algorithms/dreamer.py`
- Create: `src/axiomrl/runtime/dreamer_trainer.py`
- Modify: `src/axiomrl/experiment/registry.py`
- Modify: `src/axiomrl/api/algorithms.py`
- Modify: `src/axiomrl/__init__.py`
- Create: `configs/dreamer/breakout_atari.yaml`
- Create: `src/axiomrl/assets/configs/dreamer/breakout_atari.yaml`
- Create: `examples/dreamer_atari_reference.py`
- Test: `tests/test_dreamer_trainer_smoke.py`
- Test: `tests/test_dreamer_reference_script.py`

**Step 1: Write failing tests**
- Smoke trainer on a tiny image env (not full ALE) to keep runtime small.
- Unit-test basic model forward shapes and loss scalars.

**Step 2: Implement MVP**
- Sequence replay buffer: reuse `RecurrentReplayBuffer` with `obs_dtype=torch.uint8`.
- World model: encoder + RSSM + decoder + reward predictor.
- Actor/Critic: categorical policy head over discrete actions.
- Minimal imagination-based actor/critic updates (policy gradient on imagined rollouts).

---

### Task 5: Verification

**Run:**
- Targeted: `pytest -q tests/test_fqf_update.py tests/test_mbpo_trainer_smoke.py tests/test_gail_trainer_smoke.py tests/test_dreamer_trainer_smoke.py`
- Full: `pytest -q`

**Notes:**
- The repo guidelines disallow committing unless explicitly requested; this plan omits commit steps even though the skill recommends them.
