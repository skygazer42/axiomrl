# Tennis Event Experiment Log

Date: 2026-04-15

Keep this table current before launching another Tennis event-shaping run. The goal is to avoid rerunning lines that already regressed or were superseded.

## Current Recommendation

- Recommended baseline: `apex_dqn_tennis_event_offense_v5`
- Next candidate to run: `apex_dqn_tennis_event_offense_v5_1`
- Do not rerun: `apex_dqn_tennis_event_v5`

## Experiment Table

| Variant | Launch Date (UTC) | Run ID | Key Training-Time Shaping | Best Eval | Latest Eval | Decision | Notes |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| `apex_dqn_tennis_event_v2` | 2026-04-13 | `apex_dqn__ALE-Tennis-v5__seed53__20260413-071036-153291` | Cross-aware shaping with larger landing bonuses and `max_step_shaping_abs=0.25` | `0.0 @ 1.25M` | `0.0 @ 1.25M` | Keep as historical reference | Reached `0.0` quickly, but shaping was intentionally aggressive. |
| `apex_dqn_tennis_event_v3` | 2026-04-13 | `apex_dqn__ALE-Tennis-v5__seed53__20260413-091706-929328` | Reduced dense shaping, lower clip `0.12`, stronger failure penalty than v2 | `0.0 @ 10.75M` | `0.0 @ 10.75M` | Keep as slow-stable reference | Stable, but much slower than the best line. |
| `apex_dqn_tennis_event_v4` | 2026-04-14 | `apex_dqn__ALE-Tennis-v5__seed53__20260414-011608-658384` | Dense return shaping only: `successful_return_bonus=0.06`, `failure_penalty=-0.4`, light landing bonuses | `0.0 @ 3.0M` | `0.0 @ 3.0M` | Keep as stable reference | Fastest stable line before v5.1. |
| `apex_dqn_tennis_event_v5` | 2026-04-14 | `apex_dqn__ALE-Tennis-v5__seed53__20260414-081229-998182` | Reduced dense shaping, removed failure penalty, added outcome anchors `point_win_bonus=0.5`, `point_loss_penalty=0.5` | `2.0 @ 10.4M` | `-5.3 @ 10.5M` | Drop, do not rerun | Unstable versus v4, despite a brief spike above `0.0`. Run stopped and artifacts deleted on 2026-04-15. |
| `apex_dqn_tennis_event_v5_1` | 2026-04-15 | `apex_dqn__ALE-Tennis-v5__seed53__20260415-021902-154049` | Restores v4 dense shaping, keeps a mild win-only anchor `point_win_bonus=0.04`, leaves `point_loss_penalty=0.0` | `0.0 @ 2.5M` | `0.0 @ 2.5M` | Recommended stage-1 control | Stable at `0.0` from `1.2M` through `2.5M`. This is a rally / draw control, not yet a winning line. |
| `apex_dqn_tennis_event_offense_v2` | 2026-04-15 | `apex_dqn__ALE-Tennis-v5__seed53__20260415-121056-270984` | Preserves v5.1 stability terms, adds modest `net_cross_bonus`, stronger `deep_landing_bonus` and `wide_landing_bonus`, slightly higher `point_win_bonus` | `0.0 @ 900k` | `0.0 @ 900k` | Keep as resume base | Stable at `0.0` from `500k` through `900k`. First launch `20260415-090552-871405` failed at `200k` because periodic checkpoint export cloned the replay buffer on GPU. |
| `apex_dqn_tennis_event_offense_v3` | 2026-04-15 | `apex_dqn__ALE-Tennis-v5__seed53__20260415-143434-771920` | Resume from `offense_v2 step_750000`, lower generic return bonus, increase deep/wide landing pressure and `point_win_bonus`, raise clip to `0.10` | `0.0 @ 4.6M` | `0.0 @ 4.6M` | Stop, plateau | Stayed exactly at `0.0` for every eval from `800k` through `4.6M`. Stable, but no winning breakout. |
| `apex_dqn_tennis_event_offense_v4` | 2026-04-16 | `apex_dqn__ALE-Tennis-v5__seed53__20260416-000059-092168` | Resume from `offense_v3 step_4500000`, reduce generic return bonus, lower loss-side shaping so it no longer saturates negative clip, raise `point_win_bonus` and clip so point wins dominate routine returns | `0.0 @ 5.3M` | `0.0 @ 5.3M` | Stop, plateau | Eight evals from `4.6M` through `5.3M` all stayed at `0.0`. This reward ordering still converged to the same draw policy. |
| `apex_dqn_tennis_event_offense_v5` | 2026-04-16 | `apex_dqn__ALE-Tennis-v5__seed53__20260416-020145-352291` | Resume from `offense_v4 step_5250000`, remove generic return reward, keep only light immediate attack cues, and add an attack conversion window that rewards winning the point shortly after a deep / wide offensive shot | `0.0 @ 5.5M` | `-9.5 @ 7.6M` | Stop, regressed | First line with outcome-linked attack shaping, but the initial weighting was too aggressive. Best checkpoint is `best.pt` backed by `step_5500000.pt`, before the policy fell into sustained negative returns. |
| `apex_dqn_tennis_event_offense_v5_1` | 2026-04-16 | `apex_dqn__ALE-Tennis-v5__seed53__20260416-061740-639488` | Resume from `offense_v5 best.pt`, keep the attack conversion window but restore a small return signal and cut outcome penalties / bonuses to avoid collapsing from the draw basin into large negative scores | pending | pending | Running | Conservative follow-up to v5. Designed to keep the conversion credit path while reducing instability pressure. |

## Deleted Artifacts

- Deleted run directory: `/data/rl/axiomrl/runs/tennis-apex-event-v5`
- Deleted launch log: `/data/rl/axiomrl/runs/tennis-apex-event-v5.launch.log`

## Engineering Notes

- The original `apex_dqn` trainer path only saved a checkpoint at training end. This was patched on 2026-04-15 to save periodic checkpoints so future Tennis phase changes can resume from intermediate control policies.
- `resume --config` was added on 2026-04-15. When the new config changes `env_kwargs`, the resume path now keeps model state but resets replay / collector state instead of restoring transitions generated under the old shaping regime.
- The Tennis event wrapper was extended on 2026-04-16 with an attack conversion window: deep / wide offensive shots can open a short TTL, and the wrapper now adds a separate bonus or penalty depending on whether that attack is converted into a point win or a point loss.

## Run Gate For Future Tennis Event Variants

1. Check this file before creating a new Tennis event preset.
2. If a proposed variant only repeats a dropped shaping strategy, do not launch it.
3. Prefer modifying the latest recommended baseline instead of branching from a failed variant.
4. Record absolute dates, run IDs, best eval, latest eval, and a keep/drop decision before deleting any artifacts.
