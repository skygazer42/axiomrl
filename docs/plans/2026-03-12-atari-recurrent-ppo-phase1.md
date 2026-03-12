# Atari + Recurrent PPO Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn `rl_training` into a mainstream-style RL package for classic control and Atari by adding pixel-observation support, Atari presets, a `RecurrentPPO` contrib algorithm, a first `zoo/` layer, and the packaging/docs polish needed for real adoption.

**Architecture:** Keep the existing `rl_training` core stable while extending it in three deliberate directions. First, teach the current env and trainer stack to support Atari wrappers and CNN encoders so existing DQN/PPO flows can train on image observations. Second, isolate recurrent PPO into `rl_training.contrib` so sequence state and hidden-state checkpoint semantics do not leak into every core policy path. Third, add a `zoo/` layer for benchmark presets, scripts, and docs, plus installable CLI metadata, so the new capability is visible and reproducible instead of being buried in ad hoc examples.

**Tech Stack:** Python 3.10, PyTorch, Gymnasium, ALE-compatible Atari environments, pytest, setuptools

---

### Task 1: Add failing Atari environment and CNN coverage

**Files:**
- Create: `tests/test_atari_envs.py`
- Create: `tests/test_nature_cnn.py`
- Modify: `tests/test_envs.py`
- Modify: `tests/test_module_contracts.py`

**Step 1: Write the failing test**

Add tests that define:
- `make_env()` applies Atari preprocessing only when `env_kwargs` or config tags request the Atari pipeline
- Atari env factories can produce stacked image observations with stable shapes
- `NatureCNN` accepts `(batch, channels, height, width)` observations and returns a fixed-size feature tensor
- non-image classic-control flows remain unchanged

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_atari_envs.py tests/test_nature_cnn.py tests/test_envs.py tests/test_module_contracts.py -q`
Expected: FAIL because Atari preprocessing and CNN feature extraction do not exist yet.

**Step 3: Write minimal implementation**

Create the smallest Atari wrapper helpers and CNN model needed to satisfy the tests without changing trainer logic yet.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_atari_envs.py tests/test_nature_cnn.py tests/test_envs.py tests/test_module_contracts.py -q`
Expected: PASS

### Task 2: Implement Atari wrappers and image-ready env factories

**Files:**
- Create: `src/rl_training/envs/atari.py`
- Modify: `src/rl_training/envs/factory.py`
- Modify: `src/rl_training/envs/__init__.py`
- Create: `src/rl_training/models/cnn/__init__.py`
- Create: `src/rl_training/models/cnn/nature.py`
- Modify: `src/rl_training/models/__init__.py`

**Step 1: Write minimal implementation**

Implement:
- a helper that detects Atari-style configs and builds the wrapper stack
- support for grayscale, resize, frame stack, and channel-first conversion
- `NatureCNN` with configurable output feature size
- exports so CNN encoders are available to trainer/model code

**Step 2: Run focused tests**

Run: `PYTHONPATH=src pytest tests/test_atari_envs.py tests/test_nature_cnn.py -q`
Expected: PASS

### Task 3: Add image-observation support to DQN and PPO training paths

**Files:**
- Create: `tests/test_atari_dqn_trainer_smoke.py`
- Create: `tests/test_atari_ppo_trainer_smoke.py`
- Modify: `src/rl_training/runtime/dqn_trainer.py`
- Modify: `src/rl_training/runtime/ppo_trainer.py`
- Modify: `src/rl_training/experiment/registry.py`
- Modify: `tests/test_checkpoint_workflows.py`
- Modify: `tests/test_experiment_manager.py`

**Step 1: Write the failing tests**

Add smoke tests that expect:
- `train_dqn()` can train on a tiny Atari-configured env and emit a checkpoint
- `train_ppo()` can train on a tiny Atari-configured env and emit a checkpoint
- `evaluate_checkpoint()` works for Atari DQN and PPO checkpoints

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_atari_dqn_trainer_smoke.py tests/test_atari_ppo_trainer_smoke.py tests/test_checkpoint_workflows.py tests/test_experiment_manager.py -q`
Expected: FAIL because the current trainers only support flat 1D observations and MLP-only loading.

**Step 3: Write minimal implementation**

Update trainer and registry logic so:
- image observations can flow through CNN-based models
- vector observations still use the current MLP code paths
- checkpoint load/evaluate/predict infer the correct encoder family from config

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_atari_dqn_trainer_smoke.py tests/test_atari_ppo_trainer_smoke.py tests/test_checkpoint_workflows.py tests/test_experiment_manager.py -q`
Expected: PASS

### Task 4: Add Atari presets, reference scripts, and CLI coverage

**Files:**
- Create: `configs/dqn/breakout_atari.yaml`
- Create: `configs/ppo/breakout_atari.yaml`
- Create: `examples/dqn_breakout_atari_reference.py`
- Create: `examples/ppo_breakout_atari_reference.py`
- Create: `tests/test_atari_reference_scripts.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`

**Step 1: Write the failing tests**

Add coverage for:
- CLI `train --config` with Atari DQN and PPO configs
- reference scripts that run a tiny smoke job with the Atari pipeline enabled
- README usage snippets that mention the Atari configs

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_cli.py tests/test_atari_reference_scripts.py -q`
Expected: FAIL because Atari configs, examples, and docs do not exist yet.

**Step 3: Write minimal implementation**

Add compact Atari configs and examples that use tiny smoke-test settings and document the intended commands.

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_cli.py tests/test_atari_reference_scripts.py -q`
Expected: PASS

### Task 5: Add failing recurrent rollout and recurrent policy coverage

**Files:**
- Create: `tests/test_recurrent_rollout_buffer.py`
- Create: `tests/test_recurrent_ppo_update.py`
- Create: `tests/test_recurrent_models.py`
- Modify: `tests/test_module_contracts.py`

**Step 1: Write the failing tests**

Add tests that define:
- a recurrent rollout buffer can store hidden states, episode starts, and sequence masks
- an LSTM actor-critic returns actions, values, logprobs, and next hidden state
- `recurrent_ppo_loss(...)` or equivalent update logic returns named metrics
- invalid sequence-length or hidden-size settings fail fast

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_recurrent_rollout_buffer.py tests/test_recurrent_ppo_update.py tests/test_recurrent_models.py tests/test_module_contracts.py -q`
Expected: FAIL because recurrent buffer and recurrent policy components do not exist yet.

**Step 3: Write minimal implementation**

Create the buffer and model primitives with the smallest sequence semantics required by the tests.

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_recurrent_rollout_buffer.py tests/test_recurrent_ppo_update.py tests/test_recurrent_models.py tests/test_module_contracts.py -q`
Expected: PASS

### Task 6: Implement `RecurrentPPO` under `rl_training.contrib`

**Files:**
- Create: `src/rl_training/data/recurrent_rollout_buffer.py`
- Create: `src/rl_training/models/recurrent/__init__.py`
- Create: `src/rl_training/models/recurrent/lstm_actor_critic.py`
- Create: `src/rl_training/contrib/__init__.py`
- Create: `src/rl_training/contrib/api.py`
- Create: `src/rl_training/contrib/recurrent_ppo.py`
- Modify: `src/rl_training/models/__init__.py`

**Step 1: Write minimal implementation**

Implement:
- `RecurrentRolloutBuffer`
- `LSTMActorCritic`
- a managed `RecurrentPPO` API class in `rl_training.contrib`
- enough serialization support for hidden-state-aware training checkpoints

**Step 2: Run focused tests**

Run: `PYTHONPATH=src pytest tests/test_recurrent_rollout_buffer.py tests/test_recurrent_models.py tests/test_recurrent_ppo_update.py -q`
Expected: PASS

### Task 7: Add the recurrent PPO trainer, registry wiring, and checkpoint flows

**Files:**
- Create: `src/rl_training/runtime/recurrent_ppo_trainer.py`
- Modify: `src/rl_training/experiment/registry.py`
- Modify: `src/rl_training/cli.py`
- Modify: `src/rl_training/runtime/workflows.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_package_api_exports.py`
- Modify: `tests/test_checkpoint_workflows.py`
- Modify: `tests/test_experiment_manager.py`
- Create: `tests/test_recurrent_ppo_trainer_smoke.py`
- Create: `tests/test_recurrent_ppo_reference_script.py`

**Step 1: Write the failing tests**

Add end-to-end coverage for:
- `algo: recurrent_ppo` train / eval / resume
- `rl_training.contrib.RecurrentPPO(...).learn()`
- checkpoint load, prediction, and evaluation
- example-script smoke execution

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_recurrent_ppo_trainer_smoke.py tests/test_public_api.py tests/test_package_api_exports.py tests/test_checkpoint_workflows.py tests/test_experiment_manager.py tests/test_recurrent_ppo_reference_script.py -q`
Expected: FAIL because recurrent PPO is not yet wired into trainer, registry, and public surfaces.

**Step 3: Write minimal implementation**

Implement:
- `train_recurrent_ppo()`
- registry loading, evaluation, and prediction helpers
- `contrib` exports and example support
- CLI handling that keeps `recurrent_ppo` explicit rather than silently overloading vanilla PPO

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_recurrent_ppo_trainer_smoke.py tests/test_public_api.py tests/test_package_api_exports.py tests/test_checkpoint_workflows.py tests/test_experiment_manager.py tests/test_recurrent_ppo_reference_script.py -q`
Expected: PASS

### Task 8: Add `zoo/` presets, benchmark manifests, and docs

**Files:**
- Create: `zoo/README.md`
- Create: `zoo/atari/benchmark.yaml`
- Create: `zoo/atari/dqn_breakout.yaml`
- Create: `zoo/atari/ppo_breakout.yaml`
- Create: `zoo/atari/recurrent_ppo_breakout.yaml`
- Create: `scripts/benchmark_zoo.py`
- Create: `tests/test_zoo_presets.py`
- Modify: `README.md`

**Step 1: Write the failing tests**

Add tests that assert:
- zoo preset files are present and parse cleanly
- benchmark manifests point to real config paths
- README references the zoo layout and run commands

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_zoo_presets.py -q`
Expected: FAIL because the zoo layer does not exist yet.

**Step 3: Write minimal implementation**

Create a minimal but real zoo layer with named presets, a benchmark manifest, and one script that can enumerate or launch preset runs.

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_zoo_presets.py -q`
Expected: PASS

### Task 9: Add installable CLI metadata and package docs polish

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Create: `tests/test_package_smoke.py`

**Step 1: Write the failing tests**

Add or extend package smoke coverage so it asserts:
- a `project.scripts` entry exposes the CLI
- README explains train / eval / resume for both core and contrib flows
- package metadata still imports cleanly

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_package_smoke.py tests/test_cli.py -q`
Expected: FAIL because installable CLI entrypoints and updated docs are missing.

**Step 3: Write minimal implementation**

Add a console script such as `rl-training` and update the README so new users can discover the stable core path, the `contrib` path, and the zoo presets.

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_package_smoke.py tests/test_cli.py -q`
Expected: PASS

### Task 10: Final verification

**Files:**
- Verify only

**Step 1: Run targeted Phase 1 verification**

Run: `PYTHONPATH=src pytest tests/test_atari_envs.py tests/test_nature_cnn.py tests/test_atari_dqn_trainer_smoke.py tests/test_atari_ppo_trainer_smoke.py tests/test_atari_reference_scripts.py tests/test_recurrent_rollout_buffer.py tests/test_recurrent_models.py tests/test_recurrent_ppo_update.py tests/test_recurrent_ppo_trainer_smoke.py tests/test_recurrent_ppo_reference_script.py tests/test_zoo_presets.py tests/test_cli.py tests/test_public_api.py tests/test_package_api_exports.py tests/test_checkpoint_workflows.py tests/test_experiment_manager.py tests/test_package_smoke.py -q`
Expected: PASS

**Step 2: Run full verification**

Run: `PYTHONPATH=src pytest -q`
Expected: PASS

**Step 3: Commit in logical slices**

Use small commits that match the phase boundaries above, for example:

```bash
git add tests/test_atari_envs.py tests/test_nature_cnn.py src/rl_training/envs/atari.py src/rl_training/envs/factory.py src/rl_training/models/cnn src/rl_training/models/__init__.py
git commit -m "feat: add atari env pipeline and nature cnn"

git add tests/test_atari_dqn_trainer_smoke.py tests/test_atari_ppo_trainer_smoke.py src/rl_training/runtime/dqn_trainer.py src/rl_training/runtime/ppo_trainer.py src/rl_training/experiment/registry.py
git commit -m "feat: enable atari training for dqn and ppo"

git add tests/test_recurrent_rollout_buffer.py tests/test_recurrent_models.py tests/test_recurrent_ppo_update.py src/rl_training/data/recurrent_rollout_buffer.py src/rl_training/models/recurrent src/rl_training/contrib
git commit -m "feat: add recurrent ppo primitives"

git add tests/test_recurrent_ppo_trainer_smoke.py tests/test_recurrent_ppo_reference_script.py src/rl_training/runtime/recurrent_ppo_trainer.py src/rl_training/experiment/registry.py src/rl_training/cli.py src/rl_training/runtime/workflows.py
git commit -m "feat: wire recurrent ppo into contrib workflows"

git add zoo scripts/benchmark_zoo.py tests/test_zoo_presets.py README.md pyproject.toml
git commit -m "feat: add zoo presets and package entrypoints"
```
