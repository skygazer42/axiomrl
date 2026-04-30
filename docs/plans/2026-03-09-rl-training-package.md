# RL Training Package Implementation Plan

Foundation design:

- `docs/plans/2026-03-09-rl-package-foundation-design.md`
- `docs/plans/2026-03-09-rl-package-module-contracts.md`
- `docs/plans/2026-03-09-rl-package-roadmap-design.md`

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python-first RL training package that can grow into a full-featured library with multiple algorithm families, richer runtime modes, and real experiment infrastructure. The first milestone is one complete PPO vertical slice, but the package must be designed to expand cleanly to DQN, SAC, TD3, and additional training workflows later.

**Architecture:** Use a hybrid of SB3, RL Baselines3 Zoo, CleanRL, and Tianshou. Keep the algorithm core stable, keep experiment management outside the core, preserve a readable reference training script per algorithm, and establish explicit collector / buffer / trainer boundaries early so the package can support both on-policy and off-policy training without architectural rewrites.

**Tech Stack:** Python 3.11+, PyTorch, Gymnasium, NumPy, Tyro, PyYAML, TensorBoard, PyTest, Ruff

## Milestone Scope

This document is the execution plan for **Phase 1** of the broader package
roadmap.

Phase summary:

- `Phase 1`: package foundation plus one full PPO training vertical slice
- `Phase 1.1`: off-policy expansion with DQN and replay-driven training
- `Phase 1.2`: continuous-control off-policy support with SAC and related
  infrastructure
- `Phase 2`: runtime and product maturity such as richer callbacks, presets,
  benchmarks, and stronger experiment tooling
- `Phase 3`: scale-oriented extensions such as async runners, distributed
  execution, offline RL, and multi-agent support

This plan deliberately starts with PPO because the repository needs one complete
training path before it adds more algorithms. That starting point must not be
misread as the final product boundary. The package is intended to expand into a
multi-algorithm RL library once the phase-one architecture is proven.

---

### Task 1: Bootstrap the package and test harness

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/axiomrl/__init__.py`
- Create: `src/axiomrl/version.py`
- Create: `tests/test_package_smoke.py`

**Step 1: Write the failing test**

```python
def test_package_exports_version():
    import axiomrl

    assert axiomrl.__version__ == "0.1.0"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_package_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'axiomrl'`

**Step 3: Write minimal implementation**

```python
# src/axiomrl/version.py
__version__ = "0.1.0"

# src/axiomrl/__init__.py
from axiomrl.version import __version__
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_package_smoke.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pyproject.toml README.md src/axiomrl/__init__.py src/axiomrl/version.py tests/test_package_smoke.py
git commit -m "chore: bootstrap rl training package"
```

### Task 2: Add typed runtime config and run directory layout

**Files:**
- Create: `src/axiomrl/config.py`
- Create: `src/axiomrl/runs.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
from pathlib import Path

from axiomrl.config import TrainConfig
from axiomrl.runs import make_run_dir


def test_make_run_dir_uses_algo_env_seed(tmp_path: Path):
    cfg = TrainConfig(algo="ppo", env_id="CartPole-v1", seed=7, output_dir=tmp_path)
    run_dir = make_run_dir(cfg)
    assert run_dir.name.startswith("ppo__CartPole-v1__seed7__")
    assert run_dir.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL because `TrainConfig` and `make_run_dir` do not exist

**Step 3: Write minimal implementation**

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainConfig:
    algo: str
    env_id: str
    seed: int
    output_dir: Path
```

```python
from datetime import datetime


def make_run_dir(cfg: TrainConfig) -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.output_dir / f"{cfg.algo}__{cfg.env_id}__seed{cfg.seed}__{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/axiomrl/config.py src/axiomrl/runs.py tests/test_config.py
git commit -m "feat: add runtime config and run directory layout"
```

### Task 3: Add environment factory and vectorized env support

**Files:**
- Create: `src/axiomrl/envs.py`
- Create: `tests/test_envs.py`

**Step 1: Write the failing test**

```python
import gymnasium as gym

from axiomrl.config import TrainConfig
from axiomrl.envs import make_vector_env


def test_make_vector_env_returns_sync_vector_env(tmp_path):
    cfg = TrainConfig(algo="ppo", env_id="CartPole-v1", seed=11, output_dir=tmp_path)
    envs = make_vector_env(cfg, num_envs=4)
    assert isinstance(envs, gym.vector.SyncVectorEnv)
    obs, info = envs.reset(seed=cfg.seed)
    assert obs.shape[0] == 4
    envs.close()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_envs.py -v`
Expected: FAIL because `make_vector_env` does not exist

**Step 3: Write minimal implementation**

```python
import gymnasium as gym


def make_env(env_id: str):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def make_vector_env(cfg: TrainConfig, num_envs: int) -> gym.vector.SyncVectorEnv:
    return gym.vector.SyncVectorEnv([make_env(cfg.env_id) for _ in range(num_envs)])
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_envs.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/axiomrl/envs.py tests/test_envs.py
git commit -m "feat: add vector environment factory"
```

### Task 4: Add rollout buffer with GAE support

**Files:**
- Create: `src/axiomrl/data/rollout_buffer.py`
- Create: `src/axiomrl/data/__init__.py`
- Create: `tests/test_rollout_buffer.py`

**Step 1: Write the failing test**

```python
import torch

from axiomrl.data.rollout_buffer import RolloutBuffer


def test_rollout_buffer_computes_returns_and_advantages():
    buf = RolloutBuffer(num_steps=2, num_envs=1, obs_shape=(4,), action_shape=())
    buf.rewards[:] = torch.tensor([[1.0], [1.0]])
    buf.values[:] = torch.tensor([[0.5], [0.5]])
    buf.dones[:] = torch.tensor([[0.0], [0.0]])
    buf.compute_returns_and_advantages(last_value=torch.tensor([0.0]), gamma=0.99, gae_lambda=0.95)
    assert buf.advantages.shape == (2, 1)
    assert buf.returns.shape == (2, 1)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rollout_buffer.py -v`
Expected: FAIL because `RolloutBuffer` does not exist

**Step 3: Write minimal implementation**

```python
class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_shape, action_shape):
        ...

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        ...
```

Implement only the fields needed for PPO first:

- `obs`
- `actions`
- `logprobs`
- `rewards`
- `dones`
- `values`
- `advantages`
- `returns`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rollout_buffer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/axiomrl/data/__init__.py src/axiomrl/data/rollout_buffer.py tests/test_rollout_buffer.py
git commit -m "feat: add rollout buffer with gae"
```

### Task 5: Implement PPO policy network and update step

**Files:**
- Create: `src/axiomrl/algorithms/__init__.py`
- Create: `src/axiomrl/algorithms/ppo.py`
- Create: `src/axiomrl/models/__init__.py`
- Create: `src/axiomrl/models/mlp_actor_critic.py`
- Create: `tests/test_ppo_update.py`

**Step 1: Write the failing test**

```python
import torch

from axiomrl.algorithms.ppo import ppo_loss


def test_ppo_loss_returns_named_metrics():
    minibatch = {
        "logprobs": torch.zeros(8),
        "old_logprobs": torch.zeros(8),
        "advantages": torch.ones(8),
        "returns": torch.ones(8),
        "values": torch.zeros(8),
        "new_values": torch.zeros(8),
        "entropy": torch.ones(8),
    }
    metrics = ppo_loss(minibatch, clip_coef=0.2, ent_coef=0.01, vf_coef=0.5)
    assert set(metrics) >= {"loss", "policy_loss", "value_loss", "entropy_loss"}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ppo_update.py -v`
Expected: FAIL because `ppo_loss` does not exist

**Step 3: Write minimal implementation**

Create:

- a small MLP actor-critic model for classic control
- a pure function `ppo_loss(...)`
- a PPO trainer helper that performs one gradient step

Keep v1 scope narrow:

- discrete action spaces only
- MLP policy only
- no recurrent policy
- no image encoder yet

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ppo_update.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/axiomrl/algorithms/__init__.py src/axiomrl/algorithms/ppo.py src/axiomrl/models/__init__.py src/axiomrl/models/mlp_actor_critic.py tests/test_ppo_update.py
git commit -m "feat: implement ppo loss and model"
```

### Task 6: Add trainer loop, logging, checkpoints, and evaluation

**Files:**
- Create: `src/axiomrl/trainer.py`
- Create: `src/axiomrl/logging.py`
- Create: `src/axiomrl/checkpointing.py`
- Create: `src/axiomrl/eval.py`
- Create: `tests/test_trainer_smoke.py`

**Step 1: Write the failing test**

```python
from pathlib import Path

from axiomrl.config import TrainConfig
from axiomrl.trainer import train_ppo


def test_train_ppo_writes_checkpoint(tmp_path: Path):
    cfg = TrainConfig(
        algo="ppo",
        env_id="CartPole-v1",
        seed=3,
        output_dir=tmp_path,
    )
    result = train_ppo(cfg, total_timesteps=1024, num_envs=2, num_steps=64)
    assert result.checkpoint_path.exists()
    assert result.metrics["global_step"] >= 1024
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trainer_smoke.py -v`
Expected: FAIL because `train_ppo` does not exist

**Step 3: Write minimal implementation**

Implement:

- `train_ppo(...)` that owns env creation, rollout collection, update epochs, evaluation, and checkpoint save
- TensorBoard scalar logging
- periodic evaluation on a single env
- checkpoint payload with model state, optimizer state, and config

Use a small result dataclass:

```python
@dataclass(slots=True)
class TrainResult:
    checkpoint_path: Path
    metrics: dict[str, int | float]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trainer_smoke.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/axiomrl/trainer.py src/axiomrl/logging.py src/axiomrl/checkpointing.py src/axiomrl/eval.py tests/test_trainer_smoke.py
git commit -m "feat: add ppo trainer loop and checkpointing"
```

### Task 7: Add CLI entrypoint and config-driven training

**Files:**
- Create: `src/axiomrl/cli.py`
- Create: `configs/ppo/cartpole.yaml`
- Create: `scripts/train.py`
- Create: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
from pathlib import Path

from axiomrl.cli import load_config


def test_load_config_reads_yaml(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("algo: ppo\nenv_id: CartPole-v1\nseed: 1\n")
    cfg = load_config(config_file)
    assert cfg.algo == "ppo"
    assert cfg.env_id == "CartPole-v1"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL because `load_config` does not exist

**Step 3: Write minimal implementation**

Implement:

- YAML config loader
- Tyro CLI for runtime overrides
- `python -m axiomrl.cli train --config configs/ppo/cartpole.yaml`
- `scripts/train.py` wrapper for local development

Keep the command surface small:

- `train`
- `eval`
- `resume`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/axiomrl/cli.py configs/ppo/cartpole.yaml scripts/train.py tests/test_cli.py
git commit -m "feat: add config-driven training cli"
```

### Task 8: Add a CleanRL-style reference script and end-to-end verification

**Files:**
- Create: `examples/ppo_cartpole_reference.py`
- Create: `tests/test_reference_script.py`
- Modify: `README.md`

**Step 1: Write the failing test**

```python
import subprocess
import sys


def test_reference_script_smoke_runs():
    proc = subprocess.run(
        [sys.executable, "examples/ppo_cartpole_reference.py", "--total-timesteps", "256"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_reference_script.py -v`
Expected: FAIL because the example script does not exist

**Step 3: Write minimal implementation**

Create a short, readable PPO reference script that mirrors the package internals but keeps the full training loop visible in one file. This becomes the learning and debugging artifact for future algorithms.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_reference_script.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add examples/ppo_cartpole_reference.py README.md tests/test_reference_script.py
git commit -m "docs: add ppo reference script and usage guide"
```

## Post-v1 Follow-up Plan

After Task 8 is complete and verified:

1. Add `ReplayBuffer` and DQN as the first off-policy algorithm.
2. Generalize the trainer into `OnPolicyTrainer` and `OffPolicyTrainer`.
3. Add environment preset configs for Atari and MuJoCo-style tasks.
4. Add Weights and Biases integration behind an optional flag.
5. Revisit async sampling only after single-node PPO and DQN are stable.

## Guardrails

- Keep the core library importable and testable without a heavyweight runtime.
- Do not build a distributed system before proving a single-process training loop.
- Keep algorithm implementations readable enough that a new contributor can debug PPO without reading the whole package.
- Prefer explicit dataclasses and small modules over magic registries in v1.
- Use the reference repositories only as design input, not as copy-paste sources.
