# RL Package Roadmap Design

**Date:** 2026-03-09

Related documents:

- `docs/plans/2026-03-09-rl-package-foundation-design.md`
- `docs/plans/2026-03-09-rl-package-module-contracts.md`
- `docs/plans/2026-03-09-rl-training-package.md`

## Goal

Define the product roadmap for `rl_training` so the repository is understood as
a real reinforcement learning package with phased delivery, not as a one-off
PPO script or a narrow learning exercise.

This document answers a simple question:

> What does "becoming a real RL package" mean for this repository?

## Product Definition

`rl_training` should become a Python-first reinforcement learning package that
supports multiple algorithm families, multiple execution patterns, and the
operational tooling required for real experiments.

The package is not just responsible for implementing update equations. It must
eventually provide:

- reusable algorithm implementations
- policy and model building blocks
- rollout and replay data systems
- environment factories and worker abstractions
- trainer and evaluator orchestration
- checkpointing, resume, logging, and run management
- configuration, presets, and CLI workflows
- readable reference implementations for debugging and learning

The first delivery milestone may be PPO, but PPO is not the product. PPO is the
first end-to-end proof that the package architecture is viable.

## What "Real Package" Means

A real RL package should be able to support the common workflows that users
expect from mature libraries:

1. Train more than one algorithm family with a stable public API.
2. Cover both on-policy and off-policy data paths.
3. Support both discrete and continuous control tasks.
4. Save, load, resume, and evaluate runs without ad hoc scripts.
5. Expose configuration and experiment management as first-class features.
6. Keep enough modularity that new algorithms do not require architecture
   rewrites.

If the repository only trains PPO in one style with no clear path to DQN, SAC,
or richer runtime modes, then it is still a demo. If it can absorb those
capabilities without collapsing into special-case code, it is on the path to
being a real package.

## Capability Areas

### 1. Algorithm Coverage

The package should grow across the main algorithm families instead of staying in
a single narrow lane.

Planned progression:

- `v1`: PPO as the first complete on-policy vertical slice
- `v1.1`: DQN to establish the off-policy discrete path
- `v1.2`: SAC to establish the off-policy continuous path
- later: TD3, A2C, and selected extensions where they reuse the same runtime
  and data boundaries

This sequence is deliberate. PPO proves the trainer, rollout buffer, policy, and
evaluation loop. DQN proves replay-driven training, target-network management,
and epsilon-style exploration. SAC proves continuous-control off-policy support
and actor-critic training with entropy regularization. Together, those three
families demonstrate that the package is general, not accidental.

### 2. Runtime and Data Systems

The runtime must be designed for more than one training style.

Core systems the package should support over time:

- rollout buffers for on-policy methods
- replay buffers for off-policy methods
- minibatch iteration and sampling utilities
- vectorized environments
- collector abstractions that are not tied to one algorithm
- trainer abstractions that can drive both on-policy and off-policy loops
- evaluator paths that can run independently from training

Follow-up runtime capabilities:

- n-step return support
- prioritized replay
- recurrent policy state handling
- asynchronous environment workers
- learner / sampler split for higher-throughput execution

The rule is that high-throughput and distributed concerns should extend the
runtime layer, not leak into every algorithm implementation.

### 3. Product and Experiment Capabilities

A mature package needs product infrastructure, not just math code.

The experiment layer should eventually include:

- strongly typed run configuration
- filesystem-safe run directory creation
- checkpoint save / load / resume
- structured metrics logging
- TensorBoard integration
- evaluation scheduling
- config presets by algorithm and environment family
- CLI entrypoints for train, eval, and resume
- reproducibility helpers such as seeding and metadata capture

After the core training flows are stable, the package can add:

- hyperparameter sweep integration
- benchmark suites
- result export utilities
- experiment registries and preset bundles

### 4. Public API Quality

The public package surface should stay stable even as internals become more
capable.

Desired shape:

```python
from rl_training.algorithms import PPO, DQN, SAC

algo = PPO(config)
algo.learn()
algo.save(path)
metrics = algo.evaluate(num_episodes=10)
```

That does not mean every algorithm must share identical internals. It means the
user-facing lifecycle should be predictable:

- construct from config
- train
- evaluate
- save
- load or resume

The modular contracts in the existing design docs are the mechanism that keeps
the public API stable while allowing internal evolution.

## Phased Roadmap

### Phase 1: Foundation and First Vertical Slice

Primary outcome:

- prove the package architecture with one serious end-to-end PPO implementation

Deliverables:

- package bootstrapping and importable `src` layout
- typed configuration and run context
- environment factory and vectorized environment support
- rollout buffer with GAE
- PPO policy, update logic, and trainer loop
- evaluation, logging, and checkpointing
- a thin experiment manager and CLI path
- unit and smoke tests for package contracts and PPO flow

Success criteria:

- a user can run PPO training, checkpoint it, resume it, and evaluate it
- the code already has explicit boundaries for `Policy`, `Algorithm`,
  `Collector`, `Buffer`, `Trainer`, and `Experiment`

### Phase 1.1: Off-Policy Expansion

Primary outcome:

- prove that the package can support a second training family without
  architectural rewrites

Deliverables:

- generalized replay buffer implementation
- off-policy algorithm base utilities
- DQN implementation for discrete control
- target-network update helpers
- exploration scheduling utilities
- config presets for classic control and Atari-like discrete tasks where
  practical

Success criteria:

- on-policy and off-policy algorithms coexist behind the same package-level
  experiment flow
- replay-driven training does not require bypassing the core runtime design

### Phase 1.2: Continuous Off-Policy Maturity

Primary outcome:

- support a mainstream continuous-control off-policy algorithm family

Deliverables:

- SAC implementation
- continuous-action policy distribution utilities
- actor / critic model presets
- replay sampling improvements needed by SAC
- stronger evaluation and checkpoint coverage for off-policy runs

Optional additions if the design stays clean:

- TD3
- observation normalization
- reward scaling helpers

Success criteria:

- the package covers discrete on-policy, discrete off-policy, and continuous
  off-policy training with shared infrastructure

### Phase 2: Runtime and Product Maturity

Primary outcome:

- move from "usable package" to "credible day-to-day training library"

Deliverables:

- async environment workers where justified
- richer callback and logging integrations
- algorithm and environment preset registries
- benchmark commands and reference result baselines
- stronger integration tests
- clearer examples and reference training scripts
- better failure handling for resume, checkpoint compatibility, and partial runs

Potential additions:

- recurrent policies
- prioritized replay
- mixed precision support
- multi-step returns

### Phase 3: Scale-Oriented Extensions

Primary outcome:

- expand the package toward higher-throughput and broader RL workloads without
  polluting the core design

Deliverables:

- sampler / learner separation
- distributed or multi-process execution modes
- offline RL data ingestion
- multi-agent extensions
- experiment orchestration for larger training fleets

These features are important, but they should be layered on top of a proven
single-node core rather than used to justify premature complexity in the first
milestones.

## Non-Goals for Early Phases

The package should not pretend to be complete by scattering incomplete support
for too many advanced topics.

Avoid in the early phases:

- half-implemented distributed training
- fragile multi-agent abstractions
- broad plugin systems without real need
- too many algorithm stubs with no tested training path
- hyper-flexible config systems that obscure the training loop

The package becomes more credible by shipping a few complete algorithm families
on a solid runtime than by advertising many unfinished features.

## Definition of Success

The repository is on the right trajectory when:

- users can train, evaluate, save, and resume multiple algorithm families
- on-policy and off-policy flows share a coherent package architecture
- the experiment layer is useful enough that external scripts stay thin
- adding a new algorithm mostly means implementing algorithm-specific math and
  model code, not inventing a new runtime
- the codebase remains readable enough that contributors can debug a full
  training path end to end

That is the bar for treating `rl_training` as a real package rather than a toy
repository.
