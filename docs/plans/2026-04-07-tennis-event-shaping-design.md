# Tennis Event Shaping Design

## Goal

Increase the chance of learning a genuinely strong Tennis policy by adding a Tennis-specific training reward wrapper that provides denser event-level feedback, while keeping evaluation on the original game reward.

This design assumes the current failure mode is not a broken reward pipeline, but an overly sparse and weak task signal for value-based learning.

## Why This Exists

Current evidence from the Tennis runs shows:

- many runs can touch `0.0`
- no run has produced `eval_return_mean > 0.0`
- several lines bounce around near the floor even after long budgets
- light reward shaping changes learning behavior materially

That suggests the main issue is task signal design, not simply more runtime.

## Scope

This round should:

- preserve the current two control lines
  - `apex_dqn_tennis_stable_lr`
  - `rainbow_dqn_tennis_no_early_stop`
- add two new training lines
  - `apex_dqn_tennis_event_shaped`
  - `rainbow_dqn_tennis_event_shaped`
- implement the shaping at the environment layer, not inside algorithm trainers
- keep evaluation reward completely unchanged

This round should not:

- rewrite the DQN trainers
- add external dependencies or new external repos
- hard-code Tennis policy logic inside the learner

## Core Design

### 1. Add a Tennis event wrapper

Introduce a new environment wrapper that observes the current Atari frame stream and emits a small auxiliary reward during training.

This wrapper should be configured through YAML and activated through `env_kwargs.training.wrappers`.

### 2. Keep evaluation raw

Evaluation must remain on the original Tennis reward.

That means:

- no event shaping in evaluation mode
- no special score normalization tricks
- no hidden reward modifications in evaluation wrappers

### 3. Reuse the wrapper across algorithms

The wrapper should work for both:

- `apex_dqn`
- `rainbow_dqn`

This avoids algorithm-specific forks and makes future reuse easy.

## Event Signals

The first version should only use a small, interpretable set of signals.

### `rally_survival_bonus`

Add a tiny positive reward at each step while the rally continues.

Purpose:
- encourage longer exchanges
- discourage immediate collapse after serve/return

### `net_cross_bonus`

Add a small reward when the ball crosses from the agent side to the opponent side.

Purpose:
- reward the agent for producing a valid return
- distinguish “hit the ball back into play” from random motion

### `successful_return_bonus`

Add a slightly larger reward when the agent appears to have returned the ball and kept the rally alive.

Purpose:
- reinforce the skill of tracking and returning the ball

### `failure_penalty`

Add a small negative shaping term when the rally appears to terminate after the agent failed to return.

Purpose:
- make failure more attributable than waiting only for sparse game-level scoring

## Detection Philosophy

This wrapper should not try to become a full Tennis simulator.

The detector should be heuristic and lightweight:

- track the moving bright object that likely corresponds to the ball
- infer ball side of court from horizontal position
- detect side transitions that imply crossing the net
- infer rally continuation or collapse from repeated ball visibility / termination transitions

The goal is not perfect semantic parsing. The goal is to produce a useful learning signal that correlates with “I hit it back” and “the rally stayed alive.”

## Reward Magnitudes

The shaping terms should be deliberately small compared with match-level outcomes.

Recommended ordering:

- `rally_survival_bonus`: smallest
- `net_cross_bonus`: small
- `successful_return_bonus`: moderate
- `failure_penalty`: moderate negative

The exact values should be tuned conservatively in the first pass. We want the auxiliary reward to guide exploration, not replace the task.

## Experiment Structure

### Control Lines

- `apex_dqn_tennis_stable_lr`
- `rainbow_dqn_tennis_no_early_stop`

### Event-Shaped Lines

- `apex_dqn_tennis_event_shaped`
- `rainbow_dqn_tennis_event_shaped`

This creates clean A/B comparisons:

- same family
- same base config
- only the reward wrapper changes

## Success Criteria

This design is successful if at least one event-shaped line:

- improves faster than its control
- exceeds the control’s local plateau
- shows a more sustained positive trend instead of isolated `0.0` spikes

The strongest signal would be the first run to break above `0.0`, but even before that, more stable movement toward positive scores would count as evidence the shaping is useful.

## Risks

### 1. Bad ball detection

If the frame heuristic is too noisy, the auxiliary reward may become random.

Mitigation:
- keep the first detector simple
- test it on sampled rollouts
- make reward magnitudes small enough that noise does not dominate

### 2. Over-shaping

If the shaping reward is too strong, the agent could optimize the auxiliary objective instead of Tennis.

Mitigation:
- keep evaluation raw
- compare only against the control line
- keep shaping magnitudes bounded

### 3. Wrapper complexity

If the wrapper becomes a large vision system, iteration slows down.

Mitigation:
- first version should be heuristic and compact
- no object detector training
- no external CV stack

## Recommendation

Proceed with this design.

This is the most direct next step if the real bottleneck is sparse Tennis reward, and it preserves the strongest remaining experimental discipline:

- two control lines
- two event-shaped lines
- original evaluation signal untouched
