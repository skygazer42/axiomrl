# RL Package Survey

Date: 2026-03-09

Updated: 2026-03-20 (refresh pinned commits; add TorchRL and d3rlpy)

Foundation follow-up:

- `docs/plans/2026-03-09-rl-package-foundation-design.md`
- `docs/plans/2026-03-09-rl-package-module-contracts.md`

## Goal

Pick a reference architecture for a new Python RL training package by studying mature open-source projects with different strengths:

- stable algorithm API
- experiment management
- readable research implementations
- modular training abstractions
- high-throughput and distributed execution

## Repository Snapshot

All repositories below were cloned locally under `references/`.

| Repository | Local path | Latest local commit | Approx size | Why it matters |
| --- | --- | --- | --- | --- |
| Stable-Baselines3 | `references/stable-baselines3` | `a72be40` on `2026-03-18` | `4.6M` | Stable, well-scoped algorithm core |
| RL Baselines3 Zoo | `references/rl-baselines3-zoo` | `bb4bdf1` on `2026-03-13` | `9.5M` | Training CLI, tuning, experiment layout |
| CleanRL | `references/cleanrl` | `004f8a0` on `2025-07-08` | `179M` | Single-file, research-friendly reference implementations |
| Tianshou | `references/tianshou` | `1bbe05b` on `2025-12-01` | `12M` | Strong modular abstractions for data, algorithms, and trainers |
| Sample Factory | `references/sample-factory` | `8b35494` on `2026-01-29` | `35M` | Throughput-focused runner and async/sync training architecture |
| Ray / RLlib | `references/ray` | `70e34a3` on `2026-03-19` | `243M` | Distributed training platform with learner/module separation |
| TorchRL | `references/torchrl` | `4e2e787` on `2026-03-20` | (varies) | PyTorch-native collectors, replay buffers, transforms |
| d3rlpy | `references/d3rlpy` | `38c34b6` on `2025-09-11` | (varies) | Offline-RL ergonomics, evaluation + CLI tooling |

## Official References

- Stable-Baselines3: <https://github.com/DLR-RM/stable-baselines3> and <https://stable-baselines3.readthedocs.io/en/master/>
- RL Baselines3 Zoo: <https://github.com/DLR-RM/rl-baselines3-zoo>
- CleanRL: <https://github.com/vwxyzjn/cleanrl> and <https://docs.cleanrl.dev/>
- Tianshou: <https://github.com/thu-ml/tianshou> and <https://tianshou.org/en/stable/>
- Sample Factory: <https://github.com/alex-petrenko/sample-factory> and <https://samplefactory.dev>
- RLlib: <https://github.com/ray-project/ray/tree/master/rllib> and <https://docs.ray.io/en/latest/rllib/index.html>
- TorchRL: <https://github.com/pytorch/rl>
- d3rlpy: <https://github.com/takuseno/d3rlpy>

## What Each Repository Teaches

### 1. Stable-Baselines3: keep the algorithm core compact and stable

Key files:

- `references/stable-baselines3/stable_baselines3/common/base_class.py`
- `references/stable-baselines3/stable_baselines3/common/on_policy_algorithm.py`
- `references/stable-baselines3/stable_baselines3/common/off_policy_algorithm.py`
- `references/stable-baselines3/stable_baselines3/common/buffers.py`
- `references/stable-baselines3/stable_baselines3/common/callbacks.py`

Patterns worth borrowing:

- A single, predictable base algorithm API shared by all algorithms.
- Clear split between shared utilities and algorithm-specific modules.
- Practical wrappers around Gymnasium vector environments.
- Callbacks, logging, checkpointing, and save/load are part of the core story.

Patterns to avoid copying literally:

- SB3 is intentionally conservative and maintenance-focused; it is not a good template for rapid algorithm experimentation.
- Some abstractions are tightly shaped around a small set of classic algorithms and may feel rigid if we want novel variants early.

### 2. RL Baselines3 Zoo: keep experiment orchestration out of the algorithm core

Key files:

- `references/rl-baselines3-zoo/rl_zoo3/train.py`
- `references/rl-baselines3-zoo/rl_zoo3/exp_manager.py`
- `references/rl-baselines3-zoo/rl_zoo3/hyperparams_opt.py`
- `references/rl-baselines3-zoo/rl_zoo3/callbacks.py`

Patterns worth borrowing:

- A separate experiment layer that owns CLI parsing, logging directories, evaluation cadence, tuning, and env-specific configuration.
- Hyperparameters live outside the algorithm implementation.
- The `ExperimentManager` pattern keeps `train.py` thin.

Patterns to avoid copying literally:

- The argument surface is large and environment-specific.
- For a new package, a smaller config surface will be easier to maintain.

### 3. CleanRL: every algorithm should have a readable vertical slice

Key files:

- `references/cleanrl/cleanrl/ppo.py`
- `references/cleanrl/cleanrl/dqn.py`
- `references/cleanrl/cleanrl_utils/benchmark.py`
- `references/cleanrl/cleanrl_utils/evals/ppo_eval.py`

Patterns worth borrowing:

- Each algorithm has a simple, readable reference script.
- Config is explicit and colocated with the training loop.
- Logging and reproducibility are first-class, not afterthoughts.
- Great for validating the minimal algorithmic path before creating abstractions.

Patterns to avoid copying literally:

- CleanRL intentionally duplicates code across scripts.
- This is excellent for learning and prototyping, but not ideal as the main architecture of a reusable package.

### 4. Tianshou: separate data collection, algorithm logic, and trainer concerns

Key files:

- `references/tianshou/tianshou/algorithm/algorithm_base.py`
- `references/tianshou/tianshou/data/collector.py`
- `references/tianshou/tianshou/data/buffer/`
- `references/tianshou/tianshou/trainer.py`
- `references/tianshou/tianshou/highlevel/`

Patterns worth borrowing:

- A strong boundary between policy, algorithm update logic, collector, buffer, and trainer.
- Dual-layer API: low-level building blocks plus a higher-level usability layer.
- Good support for vector env workers, replay buffers, logging, and typed interfaces.

Patterns to avoid copying literally:

- Tianshou v2 has a broad abstraction surface and a large algorithm matrix.
- Adopting the full abstraction model in v1 would create too much upfront design work.

### 5. Sample Factory: performance architecture should be isolated, not spread everywhere

Key files:

- `references/sample-factory/sample_factory/algo/runners/runner.py`
- `references/sample-factory/sample_factory/algo/sampling/`
- `references/sample-factory/sample_factory/algo/learning/`
- `references/sample-factory/sample_factory/cfg/`

Patterns worth borrowing:

- Explicit runner orchestration.
- Clear sampler / learner / batcher separation.
- Serial debug mode and restart behavior are good operational ideas.
- Performance-focused code lives in dedicated layers instead of contaminating the whole codebase.

Patterns to avoid copying literally:

- Event-loop orchestration, multiprocessing, and PBT are v2 concerns for this project.
- The complexity is justified for throughput, but not for a clean first version.

### 6. RLlib: module and learner separation is powerful, but expensive

Key files:

- `references/ray/rllib/algorithms/algorithm_config.py`
- `references/ray/rllib/algorithms/ppo/ppo.py`
- `references/ray/rllib/core/rl_module/rl_module.py`
- `references/ray/rllib/core/learner/learner.py`
- `references/ray/rllib/connectors/`

Patterns worth borrowing:

- Config objects that can build a trainable algorithm instance.
- Clear separation between the trainable module and the learner that updates it.
- Connector-style preprocessing pipelines are a useful future direction.

Patterns to avoid copying literally:

- RLlib is a distributed platform, not just an RL package.
- Multi-agent, offline RL, actor orchestration, and legacy compatibility should stay out of v1.

### 7. TorchRL: PyTorch-native collectors and replay buffers are composable building blocks

Key files:

- `references/torchrl/torchrl/collectors/collectors.py`
- `references/torchrl/torchrl/data/replay_buffers/`

Patterns worth borrowing:

- Reusable, composable collector and replay-buffer interfaces are a strong foundation for both performance and testability.
- Distributed or async execution layers can be optional, layered on top of the same interfaces.

Patterns to avoid copying literally:

- Hydra-heavy examples and distributed infrastructure are great for large stacks, but should not be the starting point for a compact single-node library.

### 8. d3rlpy: offline-RL ergonomics and tool-first workflows

Key files:

- `references/d3rlpy/d3rlpy/cli.py`

Patterns worth borrowing:

- CLI tools for evaluation, plotting, and model export reduce “glue code” for users.
- Offline-first assumptions encourage clean dataset contracts and repeatable evaluation.

Patterns to avoid copying literally:

- d3rlpy’s center of gravity is offline RL; adopt patterns where AxiomRL’s workflows align, but don’t force the same abstractions onto online-first training.

## Recommended Direction

Use a hybrid architecture:

1. SB3-style stable algorithm core.
2. Zoo-style experiment layer and run management.
3. CleanRL-style reference scripts for each implemented algorithm.
4. Tianshou-style separation between collector, buffer, trainer, and algorithm update logic.

Defer these to later phases:

- RLlib-style learner/module platform abstractions beyond a single-process learner.
- Sample Factory-style async runners, population-based training, and aggressive multiprocessing.

## Proposed Product Shape

Recommended v1:

- Python-first package
- PyTorch only
- Gymnasium-first environment interface
- single-node training
- vectorized environment support
- one complete on-policy vertical slice first: PPO
- experiment directories, checkpointing, evaluation, and TensorBoard from day one

Recommended v1.1:

- DQN for off-policy structure
- replay buffer generalization
- config presets per environment family

Recommended v2:

- async sampler / learner split
- offline RL data ingestion
- multi-agent support
- distributed training

## Why This Direction Fits a New Repository

The current repository is empty except for `.git`, so the main risk is overbuilding abstractions before proving the first training loop. The best move is to build a compact, readable core that can train PPO on classic control tasks, while keeping enough separation that DQN or SAC can be added without rewriting the package.

That means:

- do not start with a distributed runtime
- do not start with a huge algorithm matrix
- do not hide the training loop behind too many layers
- do build a reusable collector / buffer / trainer boundary early

## Clone Notes

The reference repositories are intentionally ignored via `.gitignore` so they can stay in the workspace for study without polluting this repository's future commits.

## Actionable takeaways for AxiomRL

Follow-up roadmap document (created 2026-03-20):

- `docs/plans/2026-03-20-axiomrl-training-engineering-optimization.md`
