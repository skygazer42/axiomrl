# Compatibility Policy

AxiomRL exposes three compatibility tiers:

- `rl_training` and `rl_training.core`: stable public surface for application engineers.
- `rl_training.experimental`: advanced and research-oriented managed algorithms that may evolve faster.
- `rl_training.contrib` and zoo workflows: explicitly experimental workflow layers.

## Semantic Versioning

AxiomRL follows Semantic Versioning for the stable root and `rl_training.core` APIs.

- Patch releases fix bugs without changing stable import paths or stable config field meaning.
- Minor releases may add stable algorithms, new optional parameters, and new workflow capabilities without breaking existing stable imports.
- Major releases may remove or redesign stable APIs.

## Deprecation Policy

Deprecated stable APIs emit warnings before removal.

- New deprecations are introduced in a minor release.
- Deprecated stable APIs remain available for at least one subsequent minor release before removal in the next major release.
- Experimental APIs may change faster, but release notes still document visible breaking changes.

## Default Stable Core

The default stable core currently includes:

- `A2C`
- `BC`
- `CQL`
- `DQN`
- `DiscreteSAC`
- `IQL`
- `PPO`
- `SAC`
- `TD3`
- `TRPO`
