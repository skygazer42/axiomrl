# RL Training Zoo

The zoo layer collects benchmark-oriented presets and launch manifests without
introducing a second training runtime.

Current focus:

- Atari DQN presets
- Atari PPO presets
- Atari recurrent PPO presets

Use the CLI directly with the config files, or enumerate them with
`rl-training zoo` or `rl-training-zoo`.

Each zoo preset points at a full training config, and the CLI can resolve that
link directly.

Examples:

```bash
rl-training train --config zoo/atari/dqn_breakout.yaml
rl-training train --config zoo/atari/ppo_breakout.yaml
rl-training train --config zoo/atari/recurrent_ppo_breakout.yaml
rl-training zoo --format commands
rl-training-zoo --format commands
```
