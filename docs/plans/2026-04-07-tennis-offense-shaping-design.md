# Tennis Offense Shaping Design

## Goal

Build on the current Tennis `event_shaped` reward signal and add a second layer of offensive shaping so the agent does not only learn to rally and avoid losing, but starts learning to create harder-to-return shots that can translate into positive match scores.

The purpose of this design is to push beyond the current `0.0` plateau.

## Current Evidence

- `event_shaped` is clearly better than the plain control lines.
- `apex_dqn_tennis_event_shaped` has stabilized at repeated `0.0` evaluations.
- `rainbow_dqn_tennis_event_shaped` has recovered from strongly negative values to `0.0`.
- No line has produced `eval_return_mean > 0.0`.

This strongly suggests the current shaping is sufficient for learning "how to rally", but not yet sufficient for learning "how to win".

## Scope

This round should:

- preserve the two best current lines as controls:
  - `apex_dqn_tennis_event_shaped`
  - `rainbow_dqn_tennis_event_shaped`
- add two new lines:
  - `apex_dqn_tennis_event_offense`
  - `rainbow_dqn_tennis_event_offense`
- extend the Tennis event wrapper rather than replacing it
- keep evaluation reward raw and unchanged

This round should not:

- rewrite the DQN trainers
- replace the event-shaping wrapper with trainer-specific logic
- use large, benchmark-breaking reward hacks

## Core Idea

The current event wrapper mainly rewards:

- keeping the rally alive
- getting the ball back over the net
- avoiding immediate failure

That is enough to move toward draw-like behavior, but not enough to encourage offensive pressure.

The next shaping layer should reward not only "successful return", but "strong return placement".

## Offensive Signals

### `deep_landing_bonus`

When the ball crosses onto the opponent side, add a bonus if the detected landing / trajectory location is deep into the opponent half.

Purpose:
- encourage shots that push the opponent back
- reward more aggressive court control than merely clearing the net

### `wide_landing_bonus`

When the ball crosses onto the opponent side, add a bonus if the detected landing / trajectory is closer to the sideline than to the court center.

Purpose:
- encourage angular returns
- create harder recovery positions for the opponent

## Design Constraints

- Keep these bonuses smaller than the true game point reward.
- Apply them only during training.
- Preserve the existing event-shaping base:
  - `rally_survival_bonus`
  - `net_cross_bonus`
  - `successful_return_bonus`
  - `failure_penalty`
- Layer offense bonuses on top of the base event signal instead of replacing it.

## Detection Philosophy

This still uses heuristic frame analysis, not perfect semantic parsing.

The wrapper only needs a useful approximation of:

- ball side of court
- whether the ball crossed the net
- whether the ball entered a deeper or wider region on the opponent side

The objective is to introduce directional pressure, not to build a full Tennis rules engine.

## Experiment Structure

### Control

- `apex_dqn_tennis_event_shaped`
- `rainbow_dqn_tennis_event_shaped`

### New Offensive Variants

- `apex_dqn_tennis_event_offense`
- `rainbow_dqn_tennis_event_offense`

This gives a clean comparison:

- same algorithm family
- same event-shaping base
- only offensive shaping differs

## Success Criteria

This design is successful if at least one offensive variant:

- exceeds the corresponding `event_shaped` control
- shows more frequent positive-pressure behavior before full point rewards appear
- breaks the `0.0` ceiling sooner than the control

## Risks

### 1. Over-rewarding placement

If the bonuses are too large, the agent may chase wide/deep trajectories even when they are strategically poor.

Mitigation:
- keep magnitudes small
- compare directly against event-shaped controls

### 2. Noisy placement detection

Court-side and depth heuristics may be imperfect.

Mitigation:
- use coarse court regions rather than precise coordinates
- reward only strong, clear placement differences

### 3. False optimism

Even offense shaping may still plateau at `0.0`.

Mitigation:
- run offense variants as an additive comparison, not as a replacement for the current best controls

## Recommendation

Proceed with offense shaping as the next iteration.

The current evidence says the model can already learn "keep the rally alive." The missing step is "turn safe returns into scoring pressure." This design is the smallest plausible change aimed directly at that gap.
