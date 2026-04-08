# Tennis Resource Focus Design

## Goal

Stop wasting compute on Tennis lines that have not shown a path beyond `0.0`, keep only the two most credible control lines, and prepare one stronger Tennis-specific variant for each surviving algorithm family.

The immediate goal is not to add more algorithms. The goal is to reduce the search space and put nearly all available budget behind the two lines most likely to produce a real breakthrough:

- `apex_dqn_tennis_stable_lr`
- `rainbow_dqn_tennis_no_early_stop`

## Current Situation

The current Tennis experiment matrix has shown a hard ceiling:

- many lines can occasionally reach `0.0`
- no line has produced `eval_return_mean > 0.0`
- several families have consumed substantial budget without showing believable upward momentum

That means the bottleneck is no longer "more time on the same matrix". The bottleneck is experiment selection and task-specific signal design.

## Decision

### Keep

Keep only these two control lines running:

- `apex_dqn_tennis_stable_lr`
- `rainbow_dqn_tennis_no_early_stop`

These two lines represent the strongest remaining bets:

- `apex_dqn_tennis_stable_lr` is the most promising Ape-X direction after stabilizing optimization.
- `rainbow_dqn_tennis_no_early_stop` is the cleanest Rainbow direction after removing the clearly-bad early stopping behavior.

### Stop

Stop all other Tennis lines, including:

- `agent57`
- `ppo`
- `impala`
- `r2d2 20M`
- old `apex_main`
- `apex_explore_tuned`
- `apex_reward_lite`
- old `rainbow_20m`
- `rainbow_stable_lr`
- `rainbow_reward_lite`

The reason is not that all of them are irredeemable in theory. The reason is that none of them deserve primary budget anymore.

## Next Iteration

The next Tennis-specific iteration should not be another large matrix. It should be a focused `4`-line comparison:

- control: `apex_dqn_tennis_stable_lr`
- control: `rainbow_dqn_tennis_no_early_stop`
- candidate: `apex_dqn_tennis_shaped`
- candidate: `rainbow_dqn_tennis_shaped`

This creates a clean experimental structure:

- one stable control per algorithm family
- one stronger Tennis-specialized variant per family

## Tennis-Specific Changes

The next specialized variants should move beyond tiny generic Atari tweaks.

Allowed changes:

- keep Atari preprocessing and evaluation protocol intact
- disable training-time Atari reward clipping
- use wrapper-level reward shaping during training only
- preserve raw evaluation reward

Preferred first shaping round:

- small but stronger step penalty than the previous lightweight test
- no giant success bonus
- no handcrafted win detector inside the trainer
- no change to evaluation reward semantics

This keeps the experiment interpretable while still giving Tennis a clearer optimization signal than the generic Atari baseline.

## Why This Is Better

This design has three advantages:

1. It stops compute fragmentation.
2. It preserves a clean control-vs-specialized comparison for each family.
3. It treats the current evidence honestly: the issue is the setup, not just runtime.

## Selection Rule After This Reduction

Once the focused `4`-line set has run long enough:

- compare latest score
- compare best score
- compare the last few evaluations for trend
- keep only the strongest `1` or `2` lines for long-budget continuation

The intent is to aggressively concentrate budget instead of continuing broad shallow search.
