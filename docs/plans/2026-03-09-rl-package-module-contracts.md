# RL Package Module Contracts

**Date:** 2026-03-09

## Goal

Freeze the minimum module contracts for `v1` so the package can grow into a mature RL library without rewriting its foundation.

This document sits between:

- high-level architecture in `docs/plans/2026-03-09-rl-package-foundation-design.md`
- implementation sequencing in `docs/plans/2026-03-09-rl-training-package.md`

It answers one practical question:

> What interfaces should exist before we write the first real training code?

## Scope

These contracts cover `v1` only.

Included:

- `Algorithm`
- `Policy`
- `Collector`
- `RolloutBuffer`
- `ReplayBuffer`
- `Trainer`
- `ExperimentConfig`
- `RunContext`
- `Checkpoint`
- `Callback`
- `Logger`

Explicitly excluded from these contracts:

- distributed actors
- multi-agent abstractions
- offline dataset readers
- connector pipelines
- plugin marketplaces
- dynamic module graphs

## Ground Rules

### Rule 1: Contracts are smaller than implementations

The contract should define the minimum surface area that other modules may depend on.

Do not expose:

- every helper method
- internal counters
- implementation-specific caches
- algorithm-specific temporary tensors

### Rule 2: Each layer owns one kind of complexity

- `Policy` owns inference behavior.
- `Algorithm` owns update math.
- `Collector` owns rollout collection.
- `Buffer` owns temporal storage and sampling.
- `Trainer` owns the outer loop.
- `Experiment` owns configuration, directories, resume, and wiring.

If a module starts doing a neighbor's job, that is a design bug.

### Rule 3: Public API may be simple even when internal contracts are modular

The user should eventually be able to do this:

```python
from axiomrl.algorithms import PPO

algo = PPO(config)
algo.learn()
```

But the implementation must still be composed from smaller runtime pieces.

## Dependency Direction

The dependency graph should stay one-way:

```text
api -> experiment -> runtime -> algorithms -> policies
                     runtime -> data
                     runtime -> envs
algorithms -> data
algorithms -> policies
experiment -> runtime
experiment -> envs
experiment -> algorithms
```

Forbidden dependency directions:

- `algorithms -> experiment`
- `policies -> trainer`
- `buffers -> trainer`
- `collector -> algorithm-specific loss helpers`

## Shared Types

These are the shared concepts that multiple modules can depend on.

### TrainConfig

Purpose:

- runtime and experiment configuration for one train run

Recommended location:

- `src/axiomrl/experiment/config.py`

Suggested shape:

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(slots=True)
class TrainConfig:
    algo: str
    env_id: str
    seed: int
    total_timesteps: int
    output_dir: Path
    device: str = "auto"
    num_envs: int = 1
    eval_episodes: int = 5
    log_interval: int = 1
    checkpoint_interval: int = 1
    tags: tuple[str, ...] = ()
    algo_kwargs: dict[str, Any] = field(default_factory=dict)
    env_kwargs: dict[str, Any] = field(default_factory=dict)
```

Rules:

- top-level fields are stable and generic
- algorithm-specific values go under `algo_kwargs`
- environment-specific constructor values go under `env_kwargs`

### RunContext

Purpose:

- resolved run metadata and filesystem layout

Recommended location:

- `src/axiomrl/experiment/runs.py`

Suggested shape:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunContext:
    run_id: str
    run_dir: Path
    checkpoints_dir: Path
    tensorboard_dir: Path
    config_path: Path
    metadata_path: Path
```

### TrainMetrics

Purpose:

- normalized metric payload shared across callbacks, logging, and the trainer

Recommended location:

- `src/axiomrl/runtime/types.py`

Suggested shape:

```python
MetricDict = dict[str, int | float]
```

For `v1`, keep this simple. Do not add a heavyweight metrics object unless there is a proven need.

## Policy Contract

### Responsibilities

- compute actions from observations
- optionally compute values for actor-critic methods
- support deterministic and stochastic action selection
- expose train and eval behavior

### Must Not Own

- rollout collection
- gradient scheduling policy across epochs
- experiment logging cadence
- checkpoint directory policy

### Minimum Interface

Recommended location:

- `src/axiomrl/policies/base.py`

```python
from typing import Protocol

import torch


class PolicyOutput(Protocol):
    actions: torch.Tensor
    logprobs: torch.Tensor | None
    values: torch.Tensor | None
    entropy: torch.Tensor | None
    state: object | None


class Policy(Protocol):
    def train(self, mode: bool = True) -> "Policy": ...
    def eval(self) -> "Policy": ...

    def act(
        self,
        obs: torch.Tensor,
        *,
        state: object | None = None,
        deterministic: bool = False,
    ) -> PolicyOutput: ...

    def parameters(self): ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict: dict) -> None: ...
```

### Notes

- `act(...)` is the primary contract, not `forward(...)`. The actual implementation can still use `forward`.
- `state` exists in the contract now so recurrent policies can be added later without breaking the interface.
- `PolicyOutput` should be a small dataclass in implementation, not a raw dict.

## Algorithm Contract

### Responsibilities

- preprocess sampled data if required
- compute losses
- apply optimizer steps
- own optimizer and scheduler state

### Must Not Own

- environment stepping
- evaluation schedule
- run directory creation
- checkpoint path naming

### Minimum Interface

Recommended location:

- `src/axiomrl/algorithms/base.py`

```python
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(slots=True)
class UpdateResult:
    metrics: dict[str, int | float]
    num_gradient_steps: int


class Algorithm(Protocol):
    policy: Policy

    def update(self, batch, *, global_step: int) -> UpdateResult: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict: dict) -> None: ...
    def set_train_mode(self) -> None: ...
    def set_eval_mode(self) -> None: ...
```

### Recommended Internal Shape

Algorithms should be implemented in this structure:

```python
class BaseAlgorithm:
    def update(self, batch, *, global_step: int) -> UpdateResult:
        processed = self._preprocess_batch(batch)
        losses = self._compute_losses(processed)
        self._apply_gradients(losses)
        self._postprocess_batch(processed)
        return UpdateResult(...)
```

This is the narrowest useful version of the Tianshou idea.

### On-Policy / Off-Policy Split

Recommended locations:

- `src/axiomrl/algorithms/on_policy.py`
- `src/axiomrl/algorithms/off_policy.py`

Contract distinction:

- on-policy algorithms expect batch data from the latest rollout horizon
- off-policy algorithms expect samples from replay storage

Do not create separate trainer classes per algorithm. Keep the contract split at the algorithm family level.

## RolloutBuffer Contract

### Responsibilities

- store one rollout horizon for on-policy methods
- compute returns and advantages
- expose flat minibatch views for PPO-style updates

### Must Not Own

- random replay sampling across old trajectories
- environment resets
- evaluation statistics

### Minimum Interface

Recommended location:

- `src/axiomrl/data/rollout_buffer.py`

```python
from typing import Protocol

import torch


class RolloutBuffer(Protocol):
    num_steps: int
    num_envs: int

    def reset(self) -> None: ...

    def add(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        logprobs: torch.Tensor,
    ) -> None: ...

    def compute_returns_and_advantages(
        self,
        *,
        last_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None: ...

    def iter_minibatches(
        self,
        *,
        minibatch_size: int,
        shuffle: bool,
    ): ...
```

### Design Notes

- `add(...)` must use named parameters. Positional payloads are too brittle here.
- buffer layout should stay tensor-first and contiguous.
- `iter_minibatches(...)` belongs on the buffer because flattening and indexing are storage concerns.

## ReplayBuffer Contract

### Responsibilities

- append transitions over time
- preserve episode boundary information
- support random batch sampling

### Must Not Own

- policy inference
- checkpoint policy
- trainer stop conditions

### Minimum Interface

Recommended location:

- `src/axiomrl/data/replay_buffer.py`

```python
from typing import Protocol


class ReplayBuffer(Protocol):
    def reset(self) -> None: ...

    def add(
        self,
        *,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
    ) -> None: ...

    def sample(self, batch_size: int): ...
    def __len__(self) -> int: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict: dict) -> None: ...
```

### Design Notes

- for `v1`, start with single-agent replay only
- prioritized replay can be added later as a sibling implementation, not by bloating the base interface

## Collector Contract

### Responsibilities

- run the policy against envs
- step vector envs
- write collected transitions to the appropriate buffer
- return collection statistics

### Must Not Own

- algorithm update rules
- checkpoint writing
- run directory logic

### Minimum Interface

Recommended location:

- `src/axiomrl/runtime/collector.py`

```python
from dataclasses import dataclass


@dataclass(slots=True)
class CollectResult:
    num_env_steps: int
    num_episodes: int
    metrics: dict[str, int | float]
    last_obs: object | None = None


class Collector(Protocol):
    def reset(self) -> None: ...

    def collect_steps(
        self,
        *,
        num_steps: int,
        deterministic: bool = False,
    ) -> CollectResult: ...

    def collect_episodes(
        self,
        *,
        num_episodes: int,
        deterministic: bool = False,
    ) -> CollectResult: ...
```

### Design Notes

- keep separate methods for steps and episodes; do not overload one method with too many mutually exclusive arguments
- `CollectResult.metrics` should include at least episodic return mean, episodic length mean, and collection fps when available
- `Collector` should accept either a `RolloutBuffer` or `ReplayBuffer` at construction, but that union does not need to appear in the public protocol

## Trainer Contract

### Responsibilities

- own the outer loop
- decide when to collect, update, evaluate, log, and checkpoint
- maintain global counters

### Must Not Own

- algorithm loss computation
- low-level buffer indexing
- policy network architecture

### Minimum Interface

Recommended location:

- `src/axiomrl/runtime/trainer.py`

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainResult:
    run_dir: Path
    checkpoint_path: Path | None
    metrics: dict[str, int | float]


class Trainer(Protocol):
    def train(self) -> TrainResult: ...
```

### Recommended Implementation Shape

```python
class BaseTrainer:
    def train(self) -> TrainResult:
        self._setup()
        while not self._should_stop():
            self._run_training_iteration()
            self._maybe_evaluate()
            self._maybe_checkpoint()
            self._maybe_log()
        return self._build_result()
```

Subclasses:

- `OnPolicyTrainer`
- `OffPolicyTrainer`

### State Owned By Trainer

The trainer should own:

- `global_step`
- `gradient_step`
- `epoch`
- best evaluation score
- stop state

This state should not be split across callback objects or hidden inside algorithms.

## Callback Contract

### Responsibilities

- observe lifecycle events
- add optional behavior such as evaluation, checkpointing, or early stopping

### Must Not Own

- the core train loop
- required algorithm state

### Minimum Interface

Recommended location:

- `src/axiomrl/runtime/callbacks.py`

```python
class Callback(Protocol):
    def on_train_start(self, trainer) -> None: ...
    def on_collect_end(self, trainer, result: CollectResult) -> None: ...
    def on_update_end(self, trainer, result: UpdateResult) -> None: ...
    def on_eval_end(self, trainer, metrics: dict[str, int | float]) -> None: ...
    def on_train_end(self, trainer, result: TrainResult) -> None: ...
```

### Design Notes

- callbacks are optional extension points, not a replacement for trainer structure
- `v1` should ship with at least:
  - evaluation callback
  - checkpoint callback
  - progress logging callback

## Logger Contract

### Responsibilities

- consume scalar metrics and artifacts
- write them to one or more sinks

### Minimum Interface

Recommended location:

- `src/axiomrl/experiment/logging.py`

```python
class Logger(Protocol):
    def log_metrics(self, metrics: dict[str, int | float], *, step: int) -> None: ...
    def log_config(self, config: dict) -> None: ...
    def close(self) -> None: ...
```

### Design Notes

- use TensorBoard first
- keep the interface generic enough that W&B can be added later
- callbacks should depend on the logger contract, not on TensorBoard directly

## Evaluator Contract

### Responsibilities

- run deterministic or configured evaluation episodes
- report normalized evaluation metrics

### Minimum Interface

Recommended location:

- `src/axiomrl/runtime/evaluator.py`

```python
from dataclasses import dataclass


@dataclass(slots=True)
class EvalResult:
    num_episodes: int
    metrics: dict[str, int | float]


class Evaluator(Protocol):
    def evaluate(self, *, num_episodes: int) -> EvalResult: ...
```

### Design Notes

- keep evaluator separate from collector because evaluation policy and logging rules usually diverge from train collection
- sharing env factory and policy contracts is enough

## Experiment Manager Contract

### Responsibilities

- turn config into runnable objects
- create run directories
- persist metadata
- restore from checkpoint
- return a ready trainer

### Must Not Own

- policy math
- algorithm losses
- environment stepping

### Minimum Interface

Recommended location:

- `src/axiomrl/experiment/manager.py`

```python
class ExperimentManager(Protocol):
    def setup(self, config: TrainConfig) -> Trainer: ...
    def resume(self, checkpoint_path: str | Path) -> Trainer: ...
```

### Concrete Responsibilities

`setup(...)` should:

1. resolve run metadata
2. create envs
3. build policy
4. build algorithm
5. build buffers
6. build collector and evaluator
7. build trainer
8. persist resolved config

This is the narrowest useful version of the RL Zoo `ExperimentManager` pattern.

## Checkpoint Contract

### Responsibilities

- persist enough state to resume training
- stay independent from the exact filesystem layout used by callers

### Minimum Interface

Recommended location:

- `src/axiomrl/experiment/checkpointing.py`

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CheckpointState:
    algorithm_state: dict
    buffer_state: dict | None
    trainer_state: dict
    config: dict
    metadata: dict


def save_checkpoint(path: Path, state: CheckpointState) -> None: ...
def load_checkpoint(path: Path) -> CheckpointState: ...
```

### Required Contents

Every training checkpoint should include:

- policy and optimizer state
- algorithm-specific internal state
- buffer state when needed
- trainer counters
- resolved config
- metadata such as package version and seed

## API Freeze For v1

These surfaces should be considered stable for `v1`:

- `Policy.act`
- `Algorithm.update`
- `RolloutBuffer.add`
- `RolloutBuffer.compute_returns_and_advantages`
- `ReplayBuffer.add`
- `ReplayBuffer.sample`
- `Collector.collect_steps`
- `Trainer.train`
- `ExperimentManager.setup`
- checkpoint payload schema at the top level

Everything else may stay implementation-private.

## Intentional Omissions

These features are deliberately not present in the contracts:

- `Connector`
- `LearnerGroup`
- `EnvRunnerGroup`
- `ModuleSpec`
- `MultiAgentBatch`
- `OfflineDatasetReader`
- `RemoteWorkerManager`

If one of these becomes necessary, that is a signal that the package has entered a new phase and needs a new design pass.

## Suggested Skeleton Mapping

When implementation starts, use these file mappings:

- `src/axiomrl/algorithms/base.py`
- `src/axiomrl/algorithms/on_policy.py`
- `src/axiomrl/algorithms/off_policy.py`
- `src/axiomrl/algorithms/ppo.py`
- `src/axiomrl/policies/base.py`
- `src/axiomrl/policies/actor_critic.py`
- `src/axiomrl/data/rollout_buffer.py`
- `src/axiomrl/data/replay_buffer.py`
- `src/axiomrl/runtime/collector.py`
- `src/axiomrl/runtime/trainer.py`
- `src/axiomrl/runtime/evaluator.py`
- `src/axiomrl/runtime/callbacks.py`
- `src/axiomrl/experiment/config.py`
- `src/axiomrl/experiment/runs.py`
- `src/axiomrl/experiment/manager.py`
- `src/axiomrl/experiment/checkpointing.py`
- `src/axiomrl/experiment/logging.py`

## Final Recommendation

The package should freeze these ideas now:

1. `Algorithm` is update math, not training orchestration.
2. `Trainer` is orchestration, not loss math.
3. `Collector` is rollout IO, not learning.
4. `Buffer` is temporal storage, not environment control.
5. `ExperimentManager` is wiring and reproducibility, not model logic.
6. Every algorithm family gets both a reusable implementation and a readable reference script.

If these contracts stay intact, the package can grow from a small PPO project into a serious RL training library without architectural rework.
