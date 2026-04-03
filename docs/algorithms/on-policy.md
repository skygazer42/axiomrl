---
title: 策略梯度算法
icon: material/trending-up
---

# 策略梯度算法（On-Policy）

策略梯度方法直接对策略进行参数化并通过梯度上升来优化期望累积回报。这类算法通常需要使用当前策略采集的新鲜数据来进行更新，因此也被称为"在线策略"（On-Policy）方法。

---

## PPO

**Proximal Policy Optimization（近端策略优化）**

通过裁剪概率比或 KL 散度约束来限制策略更新幅度，在稳定性和采样效率之间取得出色平衡。PPO 是目前应用最广泛的策略梯度算法之一。

> Schulman et al., "Proximal Policy Optimization Algorithms", 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

**稳定性：** <span class="badge badge-stable">Core</span> &nbsp; **动作空间：** Discrete + Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 策略和值函数的学习率 |
| `n_steps` | `2048` | 每次更新前采集的步数 |
| `batch_size` | `64` | 小批量大小 |
| `n_epochs` | `10` | 每次更新的训练轮数 |
| `gamma` | `0.99` | 折扣因子 |
| `gae_lambda` | `0.95` | GAE 的 lambda 参数 |
| `clip_range` | `0.2` | 裁剪概率比的范围 |
| `ent_coef` | `0.0` | 熵正则化系数 |
| `vf_coef` | `0.5` | 值函数损失系数 |
| `max_grad_norm` | `0.5` | 梯度裁剪阈值 |

### YAML 配置示例

```yaml
algorithm: ppo
algo_kwargs:
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
```

### Python API 示例

```python
from rl_training.core import PPO

model = PPO(
    env_id="CartPole-v1",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    clip_range=0.2,
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - 对于连续控制任务，使用独立的策略和值函数网络通常效果更好。
    - `clip_range` 设为 0.1~0.3 在大多数任务上表现稳定。
    - 增大 `n_steps` 可以降低方差，但会增加每次更新的计算量。
    - 若训练不稳定，尝试降低学习率或增大 `batch_size`。

---

## A2C

**Advantage Actor-Critic（优势演员-评论家）**

A2C 是 A3C 的同步版本，使用优势函数来降低策略梯度估计的方差，同时保持无偏性。实现简单、训练稳定，是策略梯度方法的基础算法。

> Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning", 2016. [arXiv:1602.01783](https://arxiv.org/abs/1602.01783)

**稳定性：** <span class="badge badge-stable">Core</span> &nbsp; **动作空间：** Discrete + Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `7e-4` | 学习率 |
| `n_steps` | `5` | 每次更新前采集的步数 |
| `gamma` | `0.99` | 折扣因子 |
| `gae_lambda` | `1.0` | GAE lambda（1.0 表示不使用 GAE） |
| `ent_coef` | `0.01` | 熵正则化系数 |
| `vf_coef` | `0.5` | 值函数损失系数 |
| `max_grad_norm` | `0.5` | 梯度裁剪阈值 |
| `normalize_advantage` | `false` | 是否对优势值做标准化 |

### YAML 配置示例

```yaml
algorithm: a2c
algo_kwargs:
  learning_rate: 7e-4
  n_steps: 5
  gamma: 0.99
  gae_lambda: 1.0
  ent_coef: 0.01
  vf_coef: 0.5
```

### Python API 示例

```python
from rl_training.core import A2C

model = A2C(
    env_id="CartPole-v1",
    learning_rate=7e-4,
    n_steps=5,
    ent_coef=0.01,
)
model.train(total_timesteps=500_000)
```

!!! tip "最佳实践"
    - A2C 在多环境并行采样时效率显著提升，推荐设置 `n_envs >= 8`。
    - 熵系数 `ent_coef` 对于鼓励探索至关重要，离散任务中可适当增大。
    - 对比 PPO，A2C 实现更简单但通常需要更多调参。

---

## TRPO

**Trust Region Policy Optimization（信赖域策略优化）**

通过在 KL 散度约束下求解优化问题来保证每次策略更新的单调改进。使用共轭梯度法高效求解，是 PPO 的理论前身。

> Schulman et al., "Trust Region Policy Optimization", 2015. [arXiv:1502.05477](https://arxiv.org/abs/1502.05477)

**稳定性：** <span class="badge badge-stable">Core</span> &nbsp; **动作空间：** Discrete + Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-3` | 值函数的学习率 |
| `n_steps` | `2048` | 每次更新前采集的步数 |
| `batch_size` | `128` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `gae_lambda` | `0.95` | GAE lambda |
| `target_kl` | `0.01` | KL 散度约束上界 |
| `cg_max_steps` | `10` | 共轭梯度最大迭代步数 |
| `cg_damping` | `0.1` | 共轭梯度阻尼系数 |
| `line_search_max_iter` | `10` | 线搜索最大迭代次数 |

### YAML 配置示例

```yaml
algorithm: trpo
algo_kwargs:
  n_steps: 2048
  batch_size: 128
  gamma: 0.99
  gae_lambda: 0.95
  target_kl: 0.01
  cg_max_steps: 10
  cg_damping: 0.1
```

### Python API 示例

```python
from rl_training.core import TRPO

model = TRPO(
    env_id="HalfCheetah-v4",
    n_steps=2048,
    batch_size=128,
    target_kl=0.01,
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - TRPO 的理论保证使其在训练稳定性方面优于朴素策略梯度方法。
    - `target_kl` 过小会导致更新步长过短，过大则可能失去稳定性保证。
    - 计算成本高于 PPO，通常推荐优先考虑 PPO。

---

## IMPALA

**Importance Weighted Actor-Learner Architecture（重要性加权演员-学习者架构）**

面向大规模分布式训练设计的架构，使用 V-trace 离策略校正来处理多个 Actor 与集中式 Learner 之间的策略延迟问题。

> Espeholt et al., "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures", 2018. [arXiv:1802.01561](https://arxiv.org/abs/1802.01561)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete + Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `5e-4` | 学习率 |
| `n_steps` | `20` | 每个 Actor 每次发送的展开步数 |
| `gamma` | `0.99` | 折扣因子 |
| `vtrace_clip_rho` | `1.0` | V-trace rho 裁剪阈值 |
| `vtrace_clip_c` | `1.0` | V-trace c 裁剪阈值 |
| `ent_coef` | `0.01` | 熵正则化系数 |
| `vf_coef` | `0.5` | 值函数损失系数 |
| `num_actors` | `32` | 并行 Actor 数量 |

### YAML 配置示例

```yaml
algorithm: impala
algo_kwargs:
  learning_rate: 5e-4
  n_steps: 20
  gamma: 0.99
  vtrace_clip_rho: 1.0
  vtrace_clip_c: 1.0
  ent_coef: 0.01
  num_actors: 32
```

### Python API 示例

```python
from rl_training.experimental import IMPALA

model = IMPALA(
    env_id="Pong-v5",
    learning_rate=5e-4,
    num_actors=32,
    vtrace_clip_rho=1.0,
)
model.train(total_timesteps=10_000_000)
```

!!! tip "最佳实践"
    - IMPALA 的优势在于规模化，Actor 数量越多，吞吐量提升越明显。
    - V-trace 校正使得即使 Actor 策略滞后也能保证学习稳定性。
    - 适合 Atari 等需要大量采样的基准任务。

---

## APPO

**Asynchronous Proximal Policy Optimization（异步近端策略优化）**

将 PPO 的裁剪目标与 IMPALA 的分布式架构相结合，在保持 PPO 训练稳定性的同时实现高吞吐量分布式训练。

> 基于 IMPALA 架构与 PPO 裁剪目标的结合实现。

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete + Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 学习率 |
| `n_steps` | `128` | 展开步数 |
| `batch_size` | `256` | 小批量大小 |
| `n_epochs` | `4` | PPO 更新轮数 |
| `clip_range` | `0.2` | PPO 裁剪范围 |
| `gamma` | `0.99` | 折扣因子 |
| `num_actors` | `16` | 并行 Actor 数量 |
| `ent_coef` | `0.01` | 熵正则化系数 |

### YAML 配置示例

```yaml
algorithm: appo
algo_kwargs:
  learning_rate: 3e-4
  n_steps: 128
  batch_size: 256
  n_epochs: 4
  clip_range: 0.2
  num_actors: 16
```

### Python API 示例

```python
from rl_training.experimental import APPO

model = APPO(
    env_id="Breakout-v5",
    learning_rate=3e-4,
    clip_range=0.2,
    num_actors=16,
)
model.train(total_timesteps=10_000_000)
```

!!! tip "最佳实践"
    - 相较于 IMPALA，APPO 的裁剪机制提供了更稳定的训练过程。
    - 在 GPU 资源充足时，增加 `num_actors` 可以显著提高训练速度。
    - 可以作为 PPO 的分布式替代方案使用。

---

## PPG

**Phasic Policy Gradient（分阶段策略梯度）**

将策略优化和辅助值函数训练分为两个阶段交替进行，在策略阶段保持策略优化的稳定性，在辅助阶段利用共享表示进行值函数蒸馏。

> Karl Cobbe et al., "Phasic Policy Gradient", 2021. [arXiv:2009.04416](https://arxiv.org/abs/2009.04416)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `5e-4` | 学习率 |
| `n_steps` | `256` | 采集步数 |
| `batch_size` | `64` | 小批量大小 |
| `n_policy_epochs` | `1` | 策略阶段训练轮数 |
| `n_aux_epochs` | `6` | 辅助阶段训练轮数 |
| `n_policy_iters` | `32` | 两次辅助阶段之间的策略迭代次数 |
| `beta_clone` | `1.0` | 行为克隆损失权重 |
| `clip_range` | `0.2` | PPO 裁剪范围 |

### YAML 配置示例

```yaml
algorithm: ppg
algo_kwargs:
  learning_rate: 5e-4
  n_steps: 256
  batch_size: 64
  n_policy_epochs: 1
  n_aux_epochs: 6
  n_policy_iters: 32
  beta_clone: 1.0
```

### Python API 示例

```python
from rl_training.experimental import PPG

model = PPG(
    env_id="ProcgenEnv-coinrun",
    learning_rate=5e-4,
    n_policy_iters=32,
    n_aux_epochs=6,
)
model.train(total_timesteps=25_000_000)
```

!!! tip "最佳实践"
    - PPG 在 Procgen 等需要泛化能力的基准测试中表现突出。
    - `n_policy_iters` 控制策略和辅助阶段的交替频率，过低会增加计算开销。
    - 仅支持离散动作空间。

---

## GAIL

**Generative Adversarial Imitation Learning（生成对抗模仿学习）**

通过训练一个判别器来区分专家轨迹和智能体轨迹，将模仿学习问题转化为对抗训练过程，使智能体学会生成类似专家的行为。

> Ho & Ermon, "Generative Adversarial Imitation Learning", 2016. [arXiv:1606.03476](https://arxiv.org/abs/1606.03476)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 策略学习率 |
| `disc_learning_rate` | `3e-4` | 判别器学习率 |
| `n_steps` | `2048` | 每次更新采集的步数 |
| `batch_size` | `64` | 小批量大小 |
| `n_disc_updates` | `4` | 每次策略更新对应的判别器更新次数 |
| `expert_data_path` | 必填 | 专家演示数据路径 |
| `reward_type` | `"airl"` | 奖励计算方式（`airl` 或 `gail`） |
| `gradient_penalty_coef` | `10.0` | 判别器梯度惩罚系数 |

### YAML 配置示例

```yaml
algorithm: gail
algo_kwargs:
  learning_rate: 3e-4
  disc_learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  n_disc_updates: 4
  expert_data_path: "data/expert_demos.npz"
  reward_type: "airl"
  gradient_penalty_coef: 10.0
```

### Python API 示例

```python
from rl_training.experimental import GAIL

model = GAIL(
    env_id="Hopper-v4",
    expert_data_path="data/expert_demos.npz",
    learning_rate=3e-4,
    n_disc_updates=4,
    reward_type="airl",
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - 专家演示数据的质量和数量对最终性能有决定性影响。
    - 使用梯度惩罚可以稳定判别器训练，推荐保持 `gradient_penalty_coef` 在 5~20 之间。
    - `reward_type="airl"` 在实践中通常比纯 GAIL 奖励更稳定。

---

## MARWIL

**Monotonic Advantage Re-Weighted Imitation Learning（单调优势重加权模仿学习）**

通过优势函数对演示数据中的动作进行加权，在模仿学习中实现单调策略改进。既可以使用在线数据也可以使用离线演示数据。

> Wang et al., "Beyond Imitation: Generative and Variational Choreography via Machine Learning", 2020.

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete + Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` | 学习率 |
| `beta` | `1.0` | 优势加权的温度参数 |
| `batch_size` | `2048` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `vf_coef` | `1.0` | 值函数损失系数 |
| `expert_data_path` | 必填 | 演示数据路径 |
| `bc_logstd_coef` | `0.0` | BC 标准差正则化系数 |

### YAML 配置示例

```yaml
algorithm: marwil
algo_kwargs:
  learning_rate: 1e-4
  beta: 1.0
  batch_size: 2048
  gamma: 0.99
  vf_coef: 1.0
  expert_data_path: "data/demos.npz"
```

### Python API 示例

```python
from rl_training.experimental import MARWIL

model = MARWIL(
    env_id="Hopper-v4",
    expert_data_path="data/demos.npz",
    beta=1.0,
    learning_rate=1e-4,
)
model.train(total_timesteps=500_000)
```

!!! tip "最佳实践"
    - `beta` 越大，对高优势动作的偏好越强；`beta=0` 退化为行为克隆。
    - MARWIL 可以在混合质量的演示数据上工作，优于纯 BC。
    - 同时支持离散和连续动作空间，适用范围广。

---

## AWR

**Advantage Weighted Regression（优势加权回归）**

将策略优化问题转化为加权回归问题，使用指数化的优势函数作为权重，以简单的监督学习方式进行策略更新。

> Peng et al., "Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning", 2019. [arXiv:1910.00177](https://arxiv.org/abs/1910.00177)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `5e-4` | 学习率 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `beta` | `0.05` | 优势加权的温度参数 |
| `max_weight` | `20.0` | 权重裁剪上限 |
| `n_steps` | `2048` | 每次更新采集步数 |
| `gae_lambda` | `0.95` | GAE lambda |

### YAML 配置示例

```yaml
algorithm: awr
algo_kwargs:
  learning_rate: 5e-4
  batch_size: 256
  gamma: 0.99
  beta: 0.05
  max_weight: 20.0
  n_steps: 2048
```

### Python API 示例

```python
from rl_training.experimental import AWR

model = AWR(
    env_id="Walker2d-v4",
    learning_rate=5e-4,
    beta=0.05,
    max_weight=20.0,
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - `beta` 控制了对高优势动作的偏好程度，较小的值使分布更尖锐。
    - `max_weight` 可以防止极端优势值导致的训练不稳定。
    - AWR 方法简单，适合作为离策略策略优化的基线。

---

## OpenAI-ES

**Evolution Strategy（进化策略）**

OpenAI 提出的可扩展进化策略方法，通过在参数空间添加高斯噪声并根据适应度对噪声方向进行加权平均来更新参数，天然支持大规模并行化。

> Salimans et al., "Evolution Strategies as a Scalable Alternative to Reinforcement Learning", 2017. [arXiv:1703.03864](https://arxiv.org/abs/1703.03864)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `0.01` | 步长 |
| `noise_std` | `0.02` | 高斯噪声标准差 |
| `pop_size` | `256` | 种群大小（并行扰动数） |
| `num_workers` | `16` | 并行工作进程数 |
| `l2_coef` | `0.005` | L2 正则化系数 |
| `reward_shaping` | `"centered_rank"` | 适应度变换方式 |
| `antithetic` | `true` | 是否使用对称采样 |

### YAML 配置示例

```yaml
algorithm: openai_es
algo_kwargs:
  learning_rate: 0.01
  noise_std: 0.02
  pop_size: 256
  num_workers: 16
  l2_coef: 0.005
  reward_shaping: "centered_rank"
  antithetic: true
```

### Python API 示例

```python
from rl_training.experimental import OpenAIES

model = OpenAIES(
    env_id="Humanoid-v4",
    noise_std=0.02,
    pop_size=256,
    num_workers=16,
)
model.train(total_timesteps=5_000_000)
```

!!! tip "最佳实践"
    - 进化策略不需要计算梯度，适合非可微奖励函数或高度非平滑的优化问题。
    - 增大 `pop_size` 和 `num_workers` 可充分利用多核计算资源。
    - `antithetic=true`（镜像采样）可以有效降低方差。

---

## ARS

**Augmented Random Search（增强随机搜索）**

一种极其简单的无梯度优化方法，通过在参数空间施加随机扰动并根据回报差异来更新参数。在许多连续控制任务上表现出色。

> Mania et al., "Simple random search provides a competitive approach to reinforcement learning", 2018. [arXiv:1803.07055](https://arxiv.org/abs/1803.07055)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `step_size` | `0.02` | 参数更新步长 |
| `noise_std` | `0.03` | 扰动噪声标准差 |
| `n_directions` | `60` | 每次采样的随机方向数 |
| `n_top_directions` | `20` | 选取最优方向的数量 |
| `num_workers` | `8` | 并行工作进程数 |
| `normalize_obs` | `true` | 是否对观测做标准化 |

### YAML 配置示例

```yaml
algorithm: ars
algo_kwargs:
  step_size: 0.02
  noise_std: 0.03
  n_directions: 60
  n_top_directions: 20
  num_workers: 8
  normalize_obs: true
```

### Python API 示例

```python
from rl_training.experimental import ARS

model = ARS(
    env_id="HalfCheetah-v4",
    step_size=0.02,
    noise_std=0.03,
    n_directions=60,
    n_top_directions=20,
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - ARS 使用线性策略即可在部分 MuJoCo 任务上达到不错性能。
    - 观测标准化 (`normalize_obs=true`) 对性能提升至关重要。
    - 适合低维状态空间，对高维像素观测效果有限。

---

## RecurrentPPO

**Recurrent PPO（循环 PPO）**

在 PPO 框架中集成 LSTM 循环网络，使策略能够处理部分可观测环境（POMDP）。通过维护隐藏状态来捕捉时序依赖关系。

> 基于 PPO + LSTM 的集成实现，社区贡献。

**稳定性：** <span class="badge badge-contrib">Contrib</span> &nbsp; **动作空间：** Discrete + Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 学习率 |
| `n_steps` | `128` | 采集步数 |
| `batch_size` | `128` | 小批量大小 |
| `n_epochs` | `10` | PPO 训练轮数 |
| `clip_range` | `0.2` | PPO 裁剪范围 |
| `lstm_hidden_size` | `256` | LSTM 隐藏层大小 |
| `n_lstm_layers` | `1` | LSTM 层数 |
| `sequence_length` | `16` | BPTT 截断序列长度 |

### YAML 配置示例

```yaml
algorithm: recurrent_ppo
algo_kwargs:
  learning_rate: 3e-4
  n_steps: 128
  batch_size: 128
  n_epochs: 10
  clip_range: 0.2
  lstm_hidden_size: 256
  n_lstm_layers: 1
  sequence_length: 16
```

### Python API 示例

```python
from rl_training.contrib import RecurrentPPO

model = RecurrentPPO(
    env_id="MemoryMaze-9x9-v0",
    lstm_hidden_size=256,
    n_lstm_layers=1,
    sequence_length=16,
    clip_range=0.2,
)
model.train(total_timesteps=2_000_000)
```

!!! tip "最佳实践"
    - 仅在环境为部分可观测（POMDP）时使用，全观测环境用标准 PPO 即可。
    - `sequence_length` 应根据环境的时序依赖长度设置，过短会丢失信息。
    - 训练速度慢于标准 PPO，建议使用 GPU 加速。
    - 作为 Contrib 层级算法，API 可能随社区更新而变更。
