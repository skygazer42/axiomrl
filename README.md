# RL Training

RL Training is being built as a real reinforcement learning training package,
not a single-algorithm demo or a lightweight toy scaffold. The target is a
Python-first library that can grow toward the capability level of mature RL
packages while staying modular, readable, and practical to extend.

The package is intended to provide a stable core API for multiple algorithm
families, including on-policy and off-policy methods. The long-term direction
includes support for algorithms such as PPO, DQN, SAC, TD3, and related
training patterns, together with reusable abstractions for policies, rollout
buffers, replay buffers, collectors, trainers, evaluators, and experiment
management.

From the product side, RL Training is meant to cover the full workflow needed
for serious experimentation and training operations:

- vectorized environment support
- on-policy and off-policy data handling
- checkpointing, resume, and run directory management
- evaluation, logging, and TensorBoard integration
- structured configuration and CLI-oriented experiment control
- clean extension points for future async, distributed, and offline RL modes

The first milestone is still a complete PPO vertical slice built with PyTorch
and Gymnasium, but that milestone is only the foundation. The architecture is
being shaped so the package can expand into a broader RL library without
rewriting its core when more algorithms, more execution modes, and more product
capabilities are added.

The design draws from Stable-Baselines3, RL Baselines3 Zoo, CleanRL, and
Tianshou: a stable algorithm core, modular runtime boundaries, a dedicated
experiment layer, and readable reference implementations. The goal is to become
a serious RL package with a credible growth path, rather than stopping at a
minimal training example.
