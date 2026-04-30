# Rainbow DQN Implementation Plan

**Goal:** Add `rainbow_dqn` as a first-class DQN-family algorithm combining:
- Double Q-learning target selection
- Dueling network architecture
- NoisyLinear exploration
- Prioritized Experience Replay (PER)

**Architecture:** Reuse the existing `train_dqn` trainer and checkpoint workflow. Add one new model (`MLPDuelingNoisyQNetwork`) and one thin algorithm wrapper (`RainbowDQN`) to express the Double-Q behavior under a distinct algorithm name. Keep evaluation/prediction via the existing registry paths.

**N-step returns (true Rainbow):**
- `algo_kwargs.n_step` controls n-step returns for `rainbow_dqn` (default: `1` for backward compatibility).
- Recommended: `n_step=3` to match the common "true Rainbow" setting.
- When `n_step > 1`, the trainer inserts n-step transitions into replay and uses an effective discount of `gamma ** n_step` for bootstrapping.

---

### Task 1: Add failing coverage

**Files:**
- Modify: `tests/test_dqn_trainer_smoke.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_package_api_exports.py`
- Modify: `tests/test_dqn_update.py`
- Create: `tests/test_rainbow_dqn_reference_script.py`

Expected: FAIL until the algo name, model, exports, and example exist.

---

### Task 2: Implement model + wiring

**Files:**
- Create: `src/axiomrl/models/mlp_dueling_noisy_q_network.py`
- Modify: `src/axiomrl/runtime/dqn_trainer.py`
- Modify: `src/axiomrl/experiment/registry.py`
- Modify: `src/axiomrl/api/algorithms.py`
- Modify: `src/axiomrl/api/__init__.py`
- Modify: `src/axiomrl/algorithms/__init__.py`
- Modify: `src/axiomrl/__init__.py`

Notes:
- Use `PrioritizedReplayBuffer` and priority updates when `config.algo == "rainbow_dqn"`.
- Ensure eval/predict uses `eval()` mode so noisy layers become deterministic.

---

### Task 3: Add runnable example + config

**Files:**
- Create: `examples/rainbow_dqn_cartpole_reference.py`
- Create: `configs/rainbow_dqn/cartpole.yaml`

---

### Verification

Run: `pytest -q`
