# Tennis Specialized Tuning Design

## Goal

Focus Tennis training effort on the two most promising value-based lines, `apex_dqn` and `rainbow_dqn`, and stop spending primary compute on weak lines that have not shown useful progress. The immediate objective is not to claim paper-level performance yet, but to create a Tennis-specific tuning pipeline that can distinguish promising experiment variants quickly and then scale the best ones to longer budgets.

## Current Evidence

- No Tennis run has produced `eval_return_mean > 0.0`.
- `agent57` has consumed substantial budget and repeatedly regressed back to the floor, so it is not a good primary use of compute.
- `ppo` and `impala` can reach `0.0`, but they look more like flat plateaus than improving curves.
- `apex_dqn` has shown the strongest "still learning" behavior: noisy, but with a longer-term upward envelope from the `-20` range toward `0`.
- `rainbow_dqn` reached `0.0` early and was then prematurely cut off by early stopping despite later recovery toward `0`.

## Scope

This tuning round is allowed to change:

- algorithm configuration
- replay/update schedules
- exploration schedules
- early-stopping behavior
- reward wrappers / shaping
- Atari wrapper settings that remain compatible with the current training stack

This round will not add a new external RL implementation. Work will stay within the existing package and use the current training CLI and trainer stack.

## Recommended Strategy

Use a two-stage tuning pipeline.

### Stage 1: Six parallel candidate runs

Run three `apex_dqn` variants and three `rainbow_dqn` variants in parallel.

The purpose of Stage 1 is fast discrimination, not final performance. Each variant should run long enough to reveal whether it is still improving, but short enough to avoid burning a full multi-day budget on bad ideas.

### Stage 2: Promote only the winners

Select the strongest `apex_dqn` variant and the strongest `rainbow_dqn` variant from Stage 1, then continue only those lines to longer budgets such as `20M` to `30M` timesteps.

## Variant Families

### `apex_dqn`

#### 1. `stable-lr`

Purpose:
- reduce training oscillation
- keep the strong `apex_dqn` learning signal while making value updates less jumpy

Changes:
- lower learning rate
- slow target network updates
- keep reward semantics unchanged

#### 2. `explore-tuned`

Purpose:
- improve state-space coverage and avoid the "touch `0.0`, then stall" pattern

Changes:
- adjust actor epsilon distribution
- revisit PER beta schedule / replay emphasis
- keep the same overall task definition

#### 3. `reward-lite`

Purpose:
- provide a slightly denser learning signal for Tennis without turning the task into a different benchmark

Changes:
- keep Atari pipeline
- add only light reward shaping, such as a very small step penalty or mild terminal shaping
- avoid large success bonuses

### `rainbow_dqn`

#### 1. `stable-lr`

Purpose:
- reduce training instability after the early `0.0` plateau

Changes:
- lower learning rate
- consider calmer update cadence / target behavior

#### 2. `no-early-stop`

Purpose:
- stop killing a line that has noisy recovery behavior but does not strictly exceed its early best value

Changes:
- disable or materially relax early stopping
- keep core reward/task semantics unchanged

#### 3. `reward-lite`

Purpose:
- test whether light Tennis shaping helps Rainbow escape the `0.0` plateau

Changes:
- same light shaping philosophy as `apex_dqn`
- no aggressive reward hacks

## Reward Shaping Rules

To keep experiments interpretable, the first shaping round should obey these rules:

- no large one-time success bonus
- no hard-coded Tennis-specific score detector added in trainer logic
- use only wrapper-level transformations already supported by the environment stack where possible
- keep shaping magnitudes small enough that the original game reward still dominates

This keeps Stage 1 exploratory, but not yet benchmark-breaking or irreproducible.

## Selection Criteria

Stage 1 promotion should not use only a single best point. Use a simple hierarchy:

1. latest and near-latest evaluation quality
2. short-horizon trend over the last 3 to 5 evaluation points
3. stability, not just one lucky `0.0`
4. absolute best point as a secondary signal

This is important because Tennis curves here are noisy and have already shown false positives from single-point spikes.

## Compute Allocation

- Primary compute goes to `apex_dqn`
- Secondary compute goes to `rainbow_dqn`
- `agent57`, `ppo`, and `impala` receive no new primary tuning budget
- `r2d2 20M` may continue as a background observer, but does not drive the main tuning loop

## Risks

### 1. Over-shaping the task

If reward shaping is too strong, the agent may optimize the shaping signal instead of real Tennis competence.

Mitigation:
- keep shaping minimal
- compare shaped lines against non-shaped controls

### 2. Mistaking noise for progress

Tennis results here are noisy, so one good checkpoint can be misleading.

Mitigation:
- compare recent windows, not just single maxima

### 3. Resource fragmentation

Too many variants at once makes every line too shallow.

Mitigation:
- cap Stage 1 at six runs
- aggressively promote only two winners

## Deliverables

Implementation should produce:

- dedicated Stage 1 config presets for all six candidate lines
- a lightweight experiment naming scheme that makes runs easy to compare
- updated docs/plans describing the experiment matrix
- launch commands or managed runs for the six Stage 1 candidates
- a simple comparison rule for deciding which lines advance

## Recommendation

Proceed with this design as the next Tennis optimization round.

The main bet is:

- `apex_dqn` is the strongest current line to optimize
- `rainbow_dqn` deserves a second chance once the early-stopping failure mode is removed
- Tennis-specific tuning should begin with stable optimization and light shaping, not aggressive task surgery
