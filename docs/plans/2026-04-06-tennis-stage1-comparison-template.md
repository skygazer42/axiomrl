# Tennis Stage 1 Comparison Template

Use this worksheet after the first common comparison checkpoint for the focused four-line Tennis comparison set.

## Runs

| Preset | Latest Eval | Best Eval | Last 3 Eval Points | Stability Notes | Keep / Drop |
|---|---:|---:|---|---|---|
| `apex_dqn_tennis_stable_lr` |  |  |  |  |  |
| `apex_dqn_tennis_event_shaped` |  |  |  |  |  |
| `rainbow_dqn_tennis_no_early_stop` |  |  |  |  |  |
| `rainbow_dqn_tennis_event_shaped` |  |  |  |  |  |

## Selection Rules

1. Prefer the strongest latest score among runs that still look like they are improving.
2. Use the last 3 to 5 evaluation points to reject one-off spikes.
3. Prefer stable progress over a single lucky `0.0`.
4. Promote the strongest `apex` line and the strongest `rainbow` line to the next longer budget.
