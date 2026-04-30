# HER Goal Replay Design

**Date:** 2026-03-12

## Goal

Add a first-class `HER` package surface that unlocks sparse-reward,
goal-conditioned training without forcing users to bring their own replay
relabeling stack.

## Recommended First Slice

The first implementation should be deliberately narrow:

- expose `algo: her` as a goal-conditioned baseline
- back it with the existing `DDPG` continuous-control stack
- add future-goal relabeling through a dedicated HER replay buffer
- ship one built-in goal-conditioned reference environment so the package has a
  stable preset and smoke path without optional robotics dependencies

This is the lowest-risk path because it reuses:

- the existing deterministic actor-critic model family
- online off-policy training controls
- checkpoint / registry / public API surfaces

while adding the one missing capability that makes `HER` meaningful:
goal relabeling with reward recomputation.

## Observation And Reward Strategy

The trainer should operate on raw goal-conditioned dict observations with the
standard keys:

- `observation`
- `achieved_goal`
- `desired_goal`

The policy itself should still consume flat vectors. The runtime therefore
flattens `observation + desired_goal` for the actor-critic, while the replay
buffer stores achieved / desired goals separately so virtual transitions can be
relabelled later.

Relabeling should use the `future` strategy first:

- sample a completed episode
- sample a transition index
- with configurable HER ratio, replace `desired_goal` with a future
  `achieved_goal` from the same episode
- recompute reward through `env.unwrapped.compute_reward(...)`
- when available, recompute termination through
  `compute_terminated(...)` / `compute_truncated(...)`

## Built-In Example Environment

The package should include a tiny continuous-control goal environment such as a
1D point-mass reaching task:

- Box action space
- Dict observation space
- sparse reward through `compute_reward`
- optional `compute_terminated` and `compute_truncated`

This gives the package:

- a packaged `HER` config that does not depend on external robotics packages
- a deterministic test target for replay relabeling
- a minimal reference task for docs and future benchmark presets

## Package Shape

The first package-facing `HER` surface should include:

- `axiomrl.algorithms.HER` as a thin goal-conditioned wrapper over `DDPG`
- `axiomrl.runtime.her_trainer.train_her(...)`
- `configs/her/point_goal.yaml`
- registry / public API / packaged asset integration

This intentionally prioritizes product usability over implementing every HER
backend combination on day one.
