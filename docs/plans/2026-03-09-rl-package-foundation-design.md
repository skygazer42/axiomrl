# RL Package Foundation Design

**Date:** 2026-03-09

## Goal

Define the long-term foundation for a Python RL package that can eventually stand beside mature projects such as Stable-Baselines3, RL Baselines3 Zoo, CleanRL, Tianshou, Sample Factory, and RLlib.

Implementation-facing contracts:

- `docs/plans/2026-03-09-rl-package-module-contracts.md`

The intent of this document is not to optimize the first algorithm implementation. The intent is to decide the architectural ground rules that keep the package coherent as it grows from:

- a first PPO implementation
- to a small algorithm library
- to a robust training product
- to a higher-throughput runtime when needed

## Design Question

What should we align with from mature RL packages, and what should we refuse to copy too early?

## Recommendation

Build the package around four layers:

1. **Core algorithm layer** inspired by Stable-Baselines3.
2. **Training runtime layer** inspired by Tianshou.
3. **Experiment and product layer** inspired by RL Baselines3 Zoo.
4. **Reference implementation layer** inspired by CleanRL.

Keep a clear growth path toward:

- Sample Factory for asynchronous high-throughput execution
- RLlib for learner/module separation and distributed platform concerns

This gives a package that is stable enough for users, readable enough for research, and still capable of evolving into a larger system later.

## Option Space

### Option A: SB3-style monolithic algorithm library

Summary:

- Put most functionality inside algorithm classes and shared utilities.
- Keep a simple public API such as `PPO(...).learn(...)`.
- Treat training scripts and experiment tooling as secondary.

Pros:

- Fastest path to a usable package.
- Easy to explain to users.
- Stable public API.

Cons:

- Runtime orchestration tends to leak into algorithms over time.
- Research iteration becomes harder when everything is optimized for library stability.
- Experiment logic may end up duplicated in scripts.

Verdict:

- Good influence for the public API and algorithm lifecycle.
- Not enough on its own as the package foundation.

### Option B: Tianshou-style modular runtime plus Zoo-style experiment layer

Summary:

- Keep `Policy`, `Algorithm`, `Collector`, `Buffer`, `Trainer`, and `EnvWorker` separate.
- Add a product-facing layer for config, runs, checkpoints, evaluation, and CLI.
- Preserve simple reference scripts outside the reusable core.

Pros:

- Clean separation of responsibilities.
- Better extensibility across on-policy and off-policy algorithms.
- Easier to evolve without rewriting the package.

Cons:

- Requires discipline in interface design.
- More upfront design work than a single-script start.

Verdict:

- Best overall foundation.
- Recommended.

### Option C: Throughput-first runtime based on Sample Factory or RLlib patterns

Summary:

- Optimize early for async sampling, learners, shared memory, distributed actors, and advanced orchestration.

Pros:

- Strong long-term ceiling.
- Useful for large-scale training workloads.

Cons:

- High complexity before the package proves its first training path.
- Easy to overbuild and underdeliver.
- Harder to keep the code readable.

Verdict:

- Correct for later phases.
- Wrong starting point for this repository.

## Architecture Principles

### 1. Separate the package into product layers, not just folders

The package should not grow as a loose collection of utilities. Each layer needs a distinct contract:

- **public API layer**: what users import or call
- **runtime layer**: how experience is collected and training is orchestrated
- **algorithm layer**: how networks are updated
- **experiment layer**: how runs are configured, stored, resumed, and evaluated
- **reference layer**: readable end-to-end scripts for learning and debugging

### 2. Keep the training runtime outside the algorithm math

The biggest mistake to avoid is embedding rollout orchestration, evaluation cadence, checkpoint policy, and run directory logic inside every algorithm. Mature projects do not do this well by accident:

- SB3 centralizes algorithm lifecycle in shared base classes.
- Tianshou centralizes runtime responsibilities in collector, trainer, buffer, and env abstractions.
- RL Zoo centralizes experiment management outside the algorithm core.

This package should follow the same separation from day one.

### 3. Make the first algorithm implementation readable enough to debug line by line

CleanRL is a strong reminder that reusable abstractions are not the only artifact that matters. We need at least one complete reference implementation per algorithm family that a contributor can read without understanding the entire package.

That means:

- the reusable PPO implementation lives in the package
- a short reference PPO script also exists in `examples/`

### 4. Build for single-node correctness before distributed scale

The package should first be excellent at:

- local training
- deterministic seeding
- vectorized environments
- checkpoints
- evaluation
- logging
- configuration

It should not begin with:

- distributed actor management
- multi-agent orchestration
- offline dataset ingestion
- async policy lag correction
- dynamic module topologies

### 5. Favor explicit boundaries over magical flexibility

Tianshou and RLlib both show the power of flexible data structures and generic orchestration. They also show the maintenance cost. For this package:

- use explicit dataclasses or typed protocols where practical
- keep configuration schema explicit
- avoid runtime `eval()` patterns
- avoid hidden mode switches when a narrower context object will do

## Target Capability Alignment

| Capability | Primary reference | Adopt in v1 | Notes |
| --- | --- | --- | --- |
| Stable `Algorithm` lifecycle | Stable-Baselines3 | Yes | `learn`, `predict`, `save`, `load` must be consistent |
| Experiment manager and run layout | RL Baselines3 Zoo | Yes | Keep orchestration out of algorithm code |
| Readable end-to-end reference scripts | CleanRL | Yes | Required for learning and debugging |
| Collector / buffer / trainer separation | Tianshou | Yes | This is the main runtime backbone |
| Vector env and env worker abstractions | SB3 + Tianshou | Yes | Start simple, but define the boundary early |
| Shared-memory async runtime | Sample Factory | No | Keep as v2 runway |
| Learner / module / connector platform abstractions | RLlib | No | Too expensive for v1 |
| Hyperparameter search platform | RL Zoo | No | Add after the first training stack is stable |
| Multi-agent and offline RL | Tianshou + RLlib | No | Future work |

## Recommended Package Layout

```text
src/axiomrl/
  api/
    __init__.py
  algorithms/
    base.py
    on_policy.py
    off_policy.py
    ppo.py
  policies/
    base.py
    actor_critic.py
    distributions.py
  data/
    rollout_buffer.py
    replay_buffer.py
    batch.py
  envs/
    factory.py
    vector_env.py
    workers.py
    wrappers.py
  runtime/
    collector.py
    trainer.py
    evaluator.py
    callbacks.py
  experiment/
    config.py
    runs.py
    checkpointing.py
    logging.py
    registry.py
  cli/
    train.py
    eval.py
    resume.py
  utils/
    seed.py
    torch.py
    schedule.py
  version.py

configs/
  ppo/
    cartpole.yaml
    pendulum.yaml

examples/
  ppo_cartpole_reference.py

tests/
  unit/
  integration/
  smoke/
```

## Core Contracts

### Policy

Responsibilities:

- map observations to actions or action distributions
- expose value estimates when required by the algorithm
- support train and eval behavior explicitly

Should not own:

- rollout collection
- checkpoint layout
- experiment logging cadence

### Algorithm

Responsibilities:

- define the update rule
- preprocess training data for update
- perform gradient steps
- manage optimizer state

Recommended shape:

- `Algorithm.update(batch)`
- `_preprocess_batch(batch, buffer)`
- `_compute_losses(batch)`
- `_postprocess_batch(batch, buffer)`

This follows the strongest idea from Tianshou while staying narrower and easier to reason about.

### Collector

Responsibilities:

- query policy for actions
- step envs
- write transitions into the appropriate buffer
- track episode statistics

Collector must not know algorithm-specific loss math.

### Buffer

Need two buffer families:

- `RolloutBuffer` for on-policy methods
- `ReplayBuffer` for off-policy methods

The key design choice is that buffers should preserve temporal structure, not only random sampling. This is required for:

- GAE
- n-step returns
- episode boundary handling
- prioritized replay later

### Trainer

Responsibilities:

- own the outer training loop
- decide when to collect, update, evaluate, save, and stop
- report metrics

Recommended layering:

- `BaseTrainer`
- `OnPolicyTrainer`
- `OffPolicyTrainer`

This should remain orchestration-only. If algorithm math appears here, the boundary is wrong.

### Experiment Layer

Responsibilities:

- load config
- create run directory
- persist arguments and config
- wire envs, policy, algorithm, trainer, logger, and callbacks
- resume from checkpoints

This layer is where RL Zoo should influence the package most strongly.

## Data Flow

### On-policy flow

1. Experiment layer builds config, run context, envs, policy, algorithm, and trainer.
2. Collector gathers a fixed rollout horizon from vector envs.
3. Rollout buffer computes returns and advantages.
4. Algorithm performs update epochs over the rollout buffer.
5. Trainer logs metrics, evaluates periodically, and checkpoints.

### Off-policy flow

1. Collector continuously writes transitions into replay buffer.
2. Trainer decides when enough data exists to update.
3. Algorithm samples from replay buffer and performs gradient steps.
4. Evaluation and checkpoint logic remain in the trainer and experiment layers.

## Public API Direction

The package should expose a simple user-facing API, even if the internals are modular:

```python
from axiomrl.algorithms import PPO

algo = PPO(config)
algo.learn()
```

But the internal implementation should still be composed from:

- policy
- collector
- buffer
- trainer
- experiment context

This duality is important. SB3 gets the public surface right. Tianshou gets the internal separation right. The package should combine both.

## Configuration Strategy

Use explicit schema-driven configuration.

Preferred structure:

- Python dataclasses for runtime config
- YAML presets for environments and algorithm defaults
- CLI overrides for a small number of runtime values

Do not use:

- arbitrary Python evaluation in config
- implicit imports with broad side effects
- algorithm constructors that silently accept unrelated runtime parameters

## Run Artifact Policy

Every run should persist enough information to be reproducible:

- resolved config
- raw CLI args
- seed
- git commit if available
- checkpoint metadata
- evaluation summaries
- TensorBoard logs

This is non-negotiable for a training package.

## Testing Strategy

Testing should be designed by layers:

### Unit tests

- schedules
- distributions
- buffer indexing and returns
- config parsing
- checkpoint serialization

### Integration tests

- PPO on CartPole for a short run
- vector env creation
- resume training from checkpoint
- evaluation loop

### Reference parity tests

- ensure the reference script and reusable package implementation agree on tensor shapes and major metrics for a small fixed rollout

### Future performance tests

- collector throughput
- learner throughput
- queue saturation or buffer pressure once async runtime exists

## Non-Goals For v1

The following are explicitly out of scope for the first foundation:

- distributed training
- multi-node actor management
- multi-agent training
- offline RL datasets
- population-based training
- advanced connector pipelines
- fully generic module graphs
- every major RL algorithm

Not doing these early is part of the architecture, not a temporary weakness.

## Growth Path

### Phase 0: foundation

- package skeleton
- config and run layout
- vector env factory
- rollout buffer
- PPO training loop
- evaluation and checkpointing
- reference PPO script

### Phase 1: small reusable library

- DQN
- replay buffer family
- shared callback and logging interfaces
- environment presets

### Phase 2: runtime expansion

- separate sampler and learner roles
- async collection
- throughput metrics
- policy lag monitoring

This is where Sample Factory ideas should enter.

### Phase 3: platform expansion

- learner/module separation
- connector pipelines
- offline data readers
- multi-agent support

This is where RLlib ideas become relevant.

## Architectural Invariants

These rules should remain true even as the package grows:

1. Algorithm update logic must stay separate from experiment management.
2. The trainer must orchestrate training, not implement algorithm math.
3. A readable reference script must exist for each major algorithm family.
4. Public API simplicity must not require internal monolith design.
5. Throughput optimizations must be isolated behind runtime boundaries.
6. Reproducibility artifacts are required for every train run.

## Reading List

Use these local files when implementing the foundation:

- `references/stable-baselines3/stable_baselines3/common/base_class.py`
- `references/stable-baselines3/stable_baselines3/common/on_policy_algorithm.py`
- `references/rl-baselines3-zoo/rl_zoo3/exp_manager.py`
- `references/cleanrl/cleanrl/ppo.py`
- `references/cleanrl/cleanrl_utils/evals/ppo_eval.py`
- `references/tianshou/tianshou/algorithm/algorithm_base.py`
- `references/tianshou/tianshou/data/collector.py`
- `references/tianshou/tianshou/trainer.py`
- `references/sample-factory/docs/06-architecture/overview.md`
- `references/ray/rllib/algorithms/algorithm.py`

## Final Decision

The foundation should be:

- **SB3-like on the outside**
- **Tianshou-like in the runtime core**
- **RL Zoo-like in experiment management**
- **CleanRL-like in reference readability**
- **Sample Factory-aware for v2**
- **RLlib-aware for v3**

Anything more ambitious than that at the start is likely to create a framework before it creates a dependable RL package.
