---
title: 离策略算法
icon: material/database-arrow-left
---

# 离策略算法（Off-Policy）

离策略方法能够利用由不同策略收集的历史经验数据进行学习，通过经验回放缓冲区（Replay Buffer）极大地提升了采样效率。本页涵盖连续动作空间的 Actor-Critic 方法和离散动作空间的基于值的方法。

---

## 连续动作空间

以下算法适用于连续动作空间的控制任务，如机器人操控、运动控制等。

---

### SAC

**Soft Actor-Critic（软演员-评论家）**

基于最大熵框架的离策略 Actor-Critic 算法，通过自动调节温度参数来平衡回报最大化和策略熵最大化，在连续控制任务中表现卓越。

> Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor", 2018. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)

**稳定性：** <span class="badge badge-stable">Core</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | Actor 和 Critic 的学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `tau` | `0.005` | 目标网络软更新系数 |
| `ent_coef` | `"auto"` | 熵系数（`auto` 表示自动调节） |
| `target_entropy` | `"auto"` | 目标熵（`auto` 为 `-dim(A)`） |
| `learning_starts` | `10_000` | 开始训练前的随机采样步数 |
| `train_freq` | `1` | 每采集几步训练一次 |

#### YAML 配置示例

```yaml
algorithm: sac
algo_kwargs:
  learning_rate: 3e-4
  buffer_size: 1_000_000
  batch_size: 256
  gamma: 0.99
  tau: 0.005
  ent_coef: "auto"
  learning_starts: 10_000
```

#### Python API 示例

```python
from axiomrl.core import SAC

model = SAC(
    env_id="Ant-v4",
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    ent_coef="auto",
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - 使用 `ent_coef="auto"` 可自动调节探索与利用的平衡，是最推荐的设置。
    - SAC 是连续控制任务的首选算法，采样效率高、训练稳定。
    - 增大 `buffer_size` 可以存储更多历史经验，有助于提升稳定性。

---

### TD3

**Twin Delayed DDPG（双延迟深度确定性策略梯度）**

通过双 Q 网络、延迟策略更新和目标策略平滑三项改进来解决 DDPG 中的值函数过估计问题。

> Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods", 2018. [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)

**稳定性：** <span class="badge badge-stable">Core</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `tau` | `0.005` | 目标网络软更新系数 |
| `policy_delay` | `2` | 策略更新延迟步数 |
| `target_policy_noise` | `0.2` | 目标策略平滑噪声 |
| `target_noise_clip` | `0.5` | 目标噪声裁剪范围 |
| `exploration_noise` | `0.1` | 探索噪声标准差 |

#### YAML 配置示例

```yaml
algorithm: td3
algo_kwargs:
  learning_rate: 3e-4
  buffer_size: 1_000_000
  batch_size: 256
  gamma: 0.99
  tau: 0.005
  policy_delay: 2
  target_policy_noise: 0.2
  target_noise_clip: 0.5
```

#### Python API 示例

```python
from axiomrl.core import TD3

model = TD3(
    env_id="HalfCheetah-v4",
    learning_rate=3e-4,
    policy_delay=2,
    target_policy_noise=0.2,
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - TD3 使用确定性策略，适合低维连续控制但不天然支持探索。
    - `exploration_noise` 是性能的关键，过大过小都不行，需要针对任务调整。
    - 与 SAC 相比，TD3 不自动调节探索，但在某些任务上更稳定。

---

### DDPG

**Deep Deterministic Policy Gradient（深度确定性策略梯度）**

将 DQN 的思想扩展到连续动作空间，通过确定性策略和经验回放来学习连续控制策略。是 TD3 和 SAC 的前身。

> Lillicrap et al., "Continuous control with deep reinforcement learning", 2015. [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-3` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `128` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `tau` | `0.001` | 目标网络软更新系数 |
| `exploration_noise` | `0.1` | OU 噪声或高斯噪声标准差 |
| `noise_type` | `"ou"` | 噪声类型（`ou` 或 `gaussian`） |

#### YAML 配置示例

```yaml
algorithm: ddpg
algo_kwargs:
  learning_rate: 1e-3
  buffer_size: 1_000_000
  batch_size: 128
  gamma: 0.99
  tau: 0.001
  exploration_noise: 0.1
  noise_type: "ou"
```

#### Python API 示例

```python
from axiomrl.experimental import DDPG

model = DDPG(
    env_id="Pendulum-v1",
    learning_rate=1e-3,
    buffer_size=1_000_000,
    exploration_noise=0.1,
)
model.train(total_timesteps=100_000)
```

!!! tip "最佳实践"
    - 推荐优先使用 TD3 或 SAC，它们解决了 DDPG 的多种已知问题。
    - 如果使用 DDPG，批归一化（Batch Normalization）可以帮助稳定训练。
    - OU 噪声在某些任务上优于高斯噪声，但差异通常不大。

---

### D4PG

**Distributed Distributional DDPG（分布式分布型 DDPG）**

将分布型值函数、N 步回报、优先经验回放和分布式采样集成到 DDPG 框架中，显著提升性能和采样效率。

> Barth-Maron et al., "Distributed Distributional Deterministic Policy Gradients", 2018. [arXiv:1804.08617](https://arxiv.org/abs/1804.08617)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `n_step` | `5` | N 步回报步数 |
| `num_atoms` | `51` | 分布型值函数的原子数 |
| `v_min` | `-10.0` | 值分布下界 |
| `v_max` | `10.0` | 值分布上界 |
| `num_actors` | `8` | 并行 Actor 数量 |

#### YAML 配置示例

```yaml
algorithm: d4pg
algo_kwargs:
  learning_rate: 1e-4
  buffer_size: 1_000_000
  batch_size: 256
  n_step: 5
  num_atoms: 51
  v_min: -10.0
  v_max: 10.0
  num_actors: 8
```

#### Python API 示例

```python
from axiomrl.experimental import D4PG

model = D4PG(
    env_id="Humanoid-v4",
    learning_rate=1e-4,
    n_step=5,
    num_atoms=51,
    num_actors=8,
)
model.train(total_timesteps=5_000_000)
```

!!! tip "最佳实践"
    - D4PG 在分布式设置下效果最佳，单 Actor 时优势不明显。
    - `v_min` 和 `v_max` 应根据任务的回报范围合理设置。
    - N 步回报 (`n_step=5`) 可以加速信用分配。

---

### TQC

**Truncated Quantile Critics（截断分位数评论家）**

在 SAC 框架中引入分布型评论家，通过截断最高分位数来缓解过估计问题，在连续控制基准上取得 SOTA 表现。

> Kuznetsov et al., "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics", 2020. [arXiv:2005.04269](https://arxiv.org/abs/2005.04269)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `tau` | `0.005` | 软更新系数 |
| `n_critics` | `5` | 评论家网络数量 |
| `n_quantiles` | `25` | 每个评论家的分位数数量 |
| `top_quantiles_to_drop` | `2` | 截断的最高分位数数量 |

#### YAML 配置示例

```yaml
algorithm: tqc
algo_kwargs:
  learning_rate: 3e-4
  buffer_size: 1_000_000
  batch_size: 256
  n_critics: 5
  n_quantiles: 25
  top_quantiles_to_drop: 2
```

#### Python API 示例

```python
from axiomrl.experimental import TQC

model = TQC(
    env_id="Ant-v4",
    n_critics=5,
    n_quantiles=25,
    top_quantiles_to_drop=2,
)
model.train(total_timesteps=3_000_000)
```

!!! tip "最佳实践"
    - `top_quantiles_to_drop` 是最关键的超参数，截断过多会导致低估。
    - 多个评论家增加了计算开销但提升了值估计质量。
    - 在 MuJoCo 基准上通常优于标准 SAC。

---

### CrossQ

**CrossQ**

通过跨批归一化（Cross-Batch Normalization）简化 SAC 的训练流程，无需目标网络，减少超参数数量并保持竞争性能。

> Bhatt et al., "CrossQ: Batch Normalization in Deep Reinforcement Learning", 2024. [arXiv:2302.01855](https://arxiv.org/abs/2302.01855)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `ent_coef` | `"auto"` | 熵系数 |
| `critic_bn` | `true` | 是否使用批归一化 |
| `n_critics` | `2` | 评论家数量 |

#### YAML 配置示例

```yaml
algorithm: crossq
algo_kwargs:
  learning_rate: 1e-4
  buffer_size: 1_000_000
  batch_size: 256
  gamma: 0.99
  ent_coef: "auto"
  critic_bn: true
```

#### Python API 示例

```python
from axiomrl.experimental import CrossQ

model = CrossQ(
    env_id="HalfCheetah-v4",
    learning_rate=1e-4,
    critic_bn=True,
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - CrossQ 不需要目标网络，减少了内存开销和超参数调优工作。
    - 超参数鲁棒性是其主要优势，适合快速原型开发。
    - 在某些任务上性能与 SAC 持平甚至更优。

---

### REDQ

**Randomized Ensemble Double Q-learning（随机集成双 Q 学习）**

通过维护一个较大的 Q 网络集成并在每次更新时随机选取子集来计算目标值，提高更新频率而不引入显著过估计。

> Chen et al., "Randomized Ensembled Double Q-Learning: Learning Fast Without a Model", 2021. [arXiv:2101.05982](https://arxiv.org/abs/2101.05982)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `n_critics` | `10` | 集成 Q 网络总数 |
| `subset_size` | `2` | 每次更新随机选取的 Q 网络数 |
| `utd_ratio` | `20` | 更新-数据比率 |

#### YAML 配置示例

```yaml
algorithm: redq
algo_kwargs:
  learning_rate: 3e-4
  buffer_size: 1_000_000
  batch_size: 256
  n_critics: 10
  subset_size: 2
  utd_ratio: 20
```

#### Python API 示例

```python
from axiomrl.experimental import REDQ

model = REDQ(
    env_id="Walker2d-v4",
    n_critics=10,
    subset_size=2,
    utd_ratio=20,
)
model.train(total_timesteps=300_000)
```

!!! tip "最佳实践"
    - 高 `utd_ratio` 是 REDQ 采样高效的关键，但增加了计算成本。
    - `subset_size=2` 在大多数任务上是最优选择。
    - 适合采样成本高但计算资源充足的场景。

---

### RLPD

**Reinforcement Learning with Prior Data（带先验数据的强化学习）**

在在线强化学习中高效融入离线先验数据，通过简单的回放缓冲区混合策略在在线学习中充分利用离线数据。

> Ball et al., "Efficient Online Reinforcement Learning with Offline Data", 2023. [arXiv:2302.02948](https://arxiv.org/abs/2302.02948)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 学习率 |
| `buffer_size` | `1_000_000` | 在线回放缓冲区大小 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `offline_ratio` | `0.5` | 离线数据在批中的占比 |
| `prior_data_path` | 必填 | 先验数据路径 |
| `utd_ratio` | `20` | 更新-数据比率 |

#### YAML 配置示例

```yaml
algorithm: rlpd
algo_kwargs:
  learning_rate: 3e-4
  buffer_size: 1_000_000
  batch_size: 256
  offline_ratio: 0.5
  prior_data_path: "data/prior_transitions.npz"
  utd_ratio: 20
```

#### Python API 示例

```python
from axiomrl.experimental import RLPD

model = RLPD(
    env_id="Ant-v4",
    prior_data_path="data/prior_transitions.npz",
    offline_ratio=0.5,
    utd_ratio=20,
)
model.train(total_timesteps=300_000)
```

!!! tip "最佳实践"
    - `offline_ratio` 控制在线和离线数据的混合比例，0.5 是较好的起点。
    - 即使先验数据质量一般，RLPD 也能从中获益。
    - 结合 REDQ 的高 UTD 比率使用效果最佳。

---

### NAF

**Normalized Advantage Functions（归一化优势函数）**

将 Q 函数参数化为状态值加上二次型优势项，使得连续动作空间的最优动作可以解析求解，避免了策略网络的训练。

> Gu et al., "Continuous Deep Q-Learning with Model-based Acceleration", 2016. [arXiv:1603.00748](https://arxiv.org/abs/1603.00748)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-3` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `128` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `tau` | `0.001` | 目标网络软更新系数 |
| `learning_starts` | `1000` | 开始训练前的随机步数 |

#### YAML 配置示例

```yaml
algorithm: naf
algo_kwargs:
  learning_rate: 1e-3
  buffer_size: 1_000_000
  batch_size: 128
  gamma: 0.99
  tau: 0.001
```

#### Python API 示例

```python
from axiomrl.experimental import NAF

model = NAF(
    env_id="Pendulum-v1",
    learning_rate=1e-3,
    buffer_size=1_000_000,
)
model.train(total_timesteps=100_000)
```

!!! tip "最佳实践"
    - NAF 的二次型假设限制了其在复杂任务上的表现。
    - 适用于低维连续控制的简单任务。
    - 推荐优先考虑 SAC 或 TD3。

---

### CURL

**Contrastive Unsupervised RL（对比无监督强化学习）**

通过对比学习从原始像素输入中提取高效的状态表示，显著提升基于像素的强化学习性能。

> Srinivas et al., "CURL: Contrastive Unsupervised Representations for Reinforcement Learning", 2020. [arXiv:2004.04136](https://arxiv.org/abs/2004.04136)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-3` | 学习率 |
| `encoder_learning_rate` | `1e-3` | 编码器学习率 |
| `buffer_size` | `100_000` | 回放缓冲区大小 |
| `batch_size` | `128` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `curl_latent_dim` | `128` | 对比表示维度 |
| `encoder_tau` | `0.05` | 编码器动量更新系数 |

#### YAML 配置示例

```yaml
algorithm: curl
algo_kwargs:
  learning_rate: 1e-3
  encoder_learning_rate: 1e-3
  buffer_size: 100_000
  batch_size: 128
  curl_latent_dim: 128
  encoder_tau: 0.05
```

#### Python API 示例

```python
from axiomrl.experimental import CURL

model = CURL(
    env_id="dm_control/cheetah-run",
    learning_rate=1e-3,
    curl_latent_dim=128,
    encoder_tau=0.05,
)
model.train(total_timesteps=500_000)
```

!!! tip "最佳实践"
    - CURL 专为像素输入设计，非像素环境无需使用。
    - 对比学习的增强方式对性能有显著影响，默认使用随机裁剪。
    - 在 DMControl 100k 基准上显著优于朴素 SAC。

---

### DrQ

**Data-regularized Q（数据正则化 Q）**

通过对观测图像施加简单的随机增强（如移位）来正则化 Q 函数学习，无需修改算法结构即可大幅提升像素观测下的性能。

> Kostrikov et al., "Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels", 2020. [arXiv:2004.13649](https://arxiv.org/abs/2004.13649)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-3` | 学习率 |
| `buffer_size` | `100_000` | 回放缓冲区大小 |
| `batch_size` | `128` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `aug_type` | `"shift"` | 增强类型 |
| `aug_K` | `2` | Q 目标中的增强数 |
| `aug_M` | `2` | Q 值中的增强数 |

#### YAML 配置示例

```yaml
algorithm: drq
algo_kwargs:
  learning_rate: 1e-3
  buffer_size: 100_000
  batch_size: 128
  aug_type: "shift"
  aug_K: 2
  aug_M: 2
```

#### Python API 示例

```python
from axiomrl.experimental import DrQ

model = DrQ(
    env_id="dm_control/walker-walk",
    learning_rate=1e-3,
    aug_type="shift",
    aug_K=2,
)
model.train(total_timesteps=500_000)
```

!!! tip "最佳实践"
    - DrQ 的核心思想非常简单：图像增强即可显著提升性能。
    - 随机移位（shift）是最有效且计算成本最低的增强方式。
    - 推荐将 DrQ-v2 作为 DrQ 的升级版本优先使用。

---

### DrQ-v2

**DrQ-v2**

DrQ 的改进版本，将底层算法从 SAC 切换到 DDPG 并引入多项工程改进，在 DMControl 基准上取得 SOTA 表现。

> Yarats et al., "Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning", 2021. [arXiv:2107.09645](https://arxiv.org/abs/2107.09645)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `n_step` | `3` | N 步回报 |
| `exploration_noise` | `0.2` | 探索噪声标准差 |
| `feature_dim` | `50` | 编码器输出特征维度 |
| `encoder_tau` | `0.01` | 编码器动量更新系数 |

#### YAML 配置示例

```yaml
algorithm: drq_v2
algo_kwargs:
  learning_rate: 1e-4
  buffer_size: 1_000_000
  batch_size: 256
  n_step: 3
  exploration_noise: 0.2
  feature_dim: 50
  encoder_tau: 0.01
```

#### Python API 示例

```python
from axiomrl.experimental import DrQV2

model = DrQV2(
    env_id="dm_control/humanoid-walk",
    learning_rate=1e-4,
    n_step=3,
    feature_dim=50,
)
model.train(total_timesteps=2_000_000)
```

!!! tip "最佳实践"
    - DrQ-v2 是当前像素观测连续控制的首选算法之一。
    - 使用 N 步回报 (`n_step=3`) 和噪声调度可以进一步提升性能。
    - 基于 DDPG 而非 SAC，训练速度更快。

---

## 离散动作空间（基于值）

以下算法适用于离散动作空间的任务，如 Atari 游戏、棋盘游戏等。

---

### DQN

**Deep Q-Network（深度 Q 网络）**

使用深度神经网络来逼近 Q 函数，通过经验回放和目标网络来稳定训练。是深度强化学习的奠基之作。

> Mnih et al., "Human-level control through deep reinforcement learning", Nature, 2015.

**稳定性：** <span class="badge badge-stable">Core</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `32` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `target_update_interval` | `10_000` | 目标网络更新频率（步数） |
| `exploration_fraction` | `0.1` | 探索率衰减占总步数的比例 |
| `exploration_initial_eps` | `1.0` | 初始探索率 |
| `exploration_final_eps` | `0.05` | 最终探索率 |
| `learning_starts` | `50_000` | 开始训练前的随机步数 |

#### YAML 配置示例

```yaml
algorithm: dqn
algo_kwargs:
  learning_rate: 1e-4
  buffer_size: 1_000_000
  batch_size: 32
  gamma: 0.99
  target_update_interval: 10_000
  exploration_fraction: 0.1
  exploration_final_eps: 0.05
  learning_starts: 50_000
```

#### Python API 示例

```python
from axiomrl.core import DQN

model = DQN(
    env_id="Breakout-v5",
    learning_rate=1e-4,
    buffer_size=1_000_000,
    batch_size=32,
    target_update_interval=10_000,
)
model.train(total_timesteps=10_000_000)
```

!!! tip "最佳实践"
    - DQN 是离散动作空间最基础的算法，推荐作为基线使用。
    - 足够长的 `learning_starts` 对于稳定训练初期至关重要。
    - 如果需要更好的性能，考虑使用 Rainbow DQN 的各种改进。

---

### DiscreteSAC

**Discrete SAC（离散 SAC）**

将 SAC 的最大熵框架扩展到离散动作空间，通过对离散动作分布计算精确熵来避免连续 SAC 中的重参数化技巧。

> Christodoulou, "Soft Actor-Critic for Discrete Action Settings", 2019. [arXiv:1910.07207](https://arxiv.org/abs/1910.07207)

**稳定性：** <span class="badge badge-stable">Core</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `64` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `tau` | `0.005` | 目标网络软更新系数 |
| `ent_coef` | `"auto"` | 熵系数 |
| `target_entropy_ratio` | `0.98` | 目标熵占最大熵的比例 |

#### YAML 配置示例

```yaml
algorithm: discrete_sac
algo_kwargs:
  learning_rate: 3e-4
  buffer_size: 1_000_000
  batch_size: 64
  gamma: 0.99
  tau: 0.005
  ent_coef: "auto"
  target_entropy_ratio: 0.98
```

#### Python API 示例

```python
from axiomrl.core import DiscreteSAC

model = DiscreteSAC(
    env_id="CartPole-v1",
    learning_rate=3e-4,
    ent_coef="auto",
    target_entropy_ratio=0.98,
)
model.train(total_timesteps=500_000)
```

!!! tip "最佳实践"
    - 对于离散动作空间，DiscreteSAC 是比 DQN 更好的开箱即用选择。
    - 自动熵调节 (`ent_coef="auto"`) 简化了超参数调优。
    - `target_entropy_ratio` 控制探索强度，接近 1.0 时探索更积极。

---

### Double DQN

**Double DQN（双 DQN）**

通过解耦动作选择和动作评估来解决标准 DQN 中的 Q 值过估计问题。使用在线网络选择动作，目标网络评估值。

> van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", 2015. [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `32` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `target_update_interval` | `10_000` | 目标网络更新频率 |
| `exploration_final_eps` | `0.05` | 最终探索率 |

#### YAML 配置示例

```yaml
algorithm: double_dqn
algo_kwargs:
  learning_rate: 1e-4
  buffer_size: 1_000_000
  batch_size: 32
  target_update_interval: 10_000
  exploration_final_eps: 0.05
```

#### Python API 示例

```python
from axiomrl.experimental import DoubleDQN

model = DoubleDQN(
    env_id="Pong-v5",
    learning_rate=1e-4,
    target_update_interval=10_000,
)
model.train(total_timesteps=10_000_000)
```

!!! tip "最佳实践"
    - Double DQN 几乎在所有场景下都优于标准 DQN，是最基本的改进。
    - 可以与其他 DQN 改进（Dueling、PER 等）组合使用。

---

### Dueling DQN

**Dueling DQN（决斗 DQN）**

将 Q 网络的输出分解为状态值 V(s) 和优势函数 A(s,a) 两个分支，使网络能够更好地学习哪些状态更有价值。

> Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", 2015. [arXiv:1511.06581](https://arxiv.org/abs/1511.06581)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `32` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `target_update_interval` | `10_000` | 目标网络更新频率 |
| `dueling_type` | `"avg"` | 优势聚合方式（`avg` 或 `max`） |

#### YAML 配置示例

```yaml
algorithm: dueling_dqn
algo_kwargs:
  learning_rate: 1e-4
  buffer_size: 1_000_000
  batch_size: 32
  target_update_interval: 10_000
  dueling_type: "avg"
```

#### Python API 示例

```python
from axiomrl.experimental import DuelingDQN

model = DuelingDQN(
    env_id="SpaceInvaders-v5",
    learning_rate=1e-4,
    dueling_type="avg",
)
model.train(total_timesteps=10_000_000)
```

!!! tip "最佳实践"
    - Dueling 架构在动作数量多的环境中效果尤为明显。
    - 使用 `avg` 聚合方式在实践中更稳定。

---

### Noisy DQN

**Noisy DQN（噪声 DQN）**

用参数化的噪声层替代 epsilon-greedy 探索，使网络能够学习到状态相关的探索策略。

> Fortunato et al., "Noisy Networks for Exploration", 2017. [arXiv:1706.10295](https://arxiv.org/abs/1706.10295)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `32` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `sigma_init` | `0.5` | 噪声初始标准差 |
| `noise_type` | `"factorized"` | 噪声类型（`factorized` 或 `independent`） |

#### YAML 配置示例

```yaml
algorithm: noisy_dqn
algo_kwargs:
  learning_rate: 1e-4
  buffer_size: 1_000_000
  batch_size: 32
  sigma_init: 0.5
  noise_type: "factorized"
```

#### Python API 示例

```python
from axiomrl.experimental import NoisyDQN

model = NoisyDQN(
    env_id="MontezumaRevenge-v5",
    sigma_init=0.5,
    noise_type="factorized",
)
model.train(total_timesteps=50_000_000)
```

!!! tip "最佳实践"
    - Noisy 网络消除了对 epsilon 调度的需要。
    - `factorized` 噪声计算效率更高，通常推荐使用。
    - 在需要深度探索的任务（如 Montezuma's Revenge）中尤为有效。

---

### N-step DQN

**N-step DQN（N 步 DQN）**

使用 N 步回报替代单步 TD 目标，加速信用分配并减小偏差，但会引入一定的方差。

> 基于 Sutton & Barto 的 N 步时序差分方法。

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `32` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `n_step` | `3` | N 步回报步数 |
| `target_update_interval` | `10_000` | 目标网络更新频率 |

#### YAML 配置示例

```yaml
algorithm: nstep_dqn
algo_kwargs:
  learning_rate: 1e-4
  buffer_size: 1_000_000
  batch_size: 32
  n_step: 3
  target_update_interval: 10_000
```

#### Python API 示例

```python
from axiomrl.experimental import NStepDQN

model = NStepDQN(
    env_id="Qbert-v5",
    n_step=3,
    learning_rate=1e-4,
)
model.train(total_timesteps=10_000_000)
```

!!! tip "最佳实践"
    - `n_step=3~5` 在大多数 Atari 游戏中效果较好。
    - N 值过大会引入较大方差，在随机性高的环境中需谨慎。

---

### Prioritized DQN

**Prioritized DQN / PER（优先经验回放 DQN）**

根据 TD 误差为回放缓冲区中的转换赋予优先级，使高 TD 误差的样本更频繁地被采样，加速学习。

> Schaul et al., "Prioritized Experience Replay", 2015. [arXiv:1511.05952](https://arxiv.org/abs/1511.05952)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `32` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `alpha` | `0.6` | 优先级指数 |
| `beta_start` | `0.4` | 重要性采样权重初始值 |
| `beta_end` | `1.0` | 重要性采样权重终值 |
| `prior_eps` | `1e-6` | 防止零优先级的最小值 |

#### YAML 配置示例

```yaml
algorithm: prioritized_dqn
algo_kwargs:
  learning_rate: 1e-4
  buffer_size: 1_000_000
  batch_size: 32
  alpha: 0.6
  beta_start: 0.4
  beta_end: 1.0
```

#### Python API 示例

```python
from axiomrl.experimental import PrioritizedDQN

model = PrioritizedDQN(
    env_id="Breakout-v5",
    alpha=0.6,
    beta_start=0.4,
)
model.train(total_timesteps=10_000_000)
```

!!! tip "最佳实践"
    - `alpha=0` 退化为均匀采样，`alpha=1` 为完全优先级采样。
    - `beta` 应从较低值逐步退火到 1.0 以补偿采样偏差。
    - PER 在稀疏奖励环境中效果尤为显著。

---

### Rainbow DQN

**Rainbow DQN**

将六项 DQN 改进集成到一个算法中：Double DQN、Dueling、PER、N-step、Noisy Networks 和 C51 分布型值函数。

> Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning", 2017. [arXiv:1710.02298](https://arxiv.org/abs/1710.02298)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `6.25e-5` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `32` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `n_step` | `3` | N 步回报 |
| `num_atoms` | `51` | C51 原子数 |
| `v_min` | `-10.0` | 值分布下界 |
| `v_max` | `10.0` | 值分布上界 |
| `alpha` | `0.5` | PER 优先级指数 |
| `sigma_init` | `0.5` | Noisy 网络噪声初始值 |

#### YAML 配置示例

```yaml
algorithm: rainbow_dqn
algo_kwargs:
  learning_rate: 6.25e-5
  buffer_size: 1_000_000
  batch_size: 32
  n_step: 3
  num_atoms: 51
  v_min: -10.0
  v_max: 10.0
  alpha: 0.5
  sigma_init: 0.5
```

#### Python API 示例

```python
from axiomrl.experimental import RainbowDQN

model = RainbowDQN(
    env_id="Breakout-v5",
    n_step=3,
    num_atoms=51,
    sigma_init=0.5,
)
model.train(total_timesteps=10_000_000)
```

!!! tip "最佳实践"
    - Rainbow 是 Atari 基准上最强的 DQN 变体之一。
    - 学习率建议设为较低值（如 `6.25e-5`），以适应多项改进的组合。
    - `v_min` 和 `v_max` 需根据任务回报范围调整。

---

### C51 DQN

**C51 / Categorical DQN（分类 DQN）**

不预测 Q 值的期望，而是学习完整的值分布。使用固定数量的原子来参数化离散概率分布。

> Bellemare et al., "A Distributional Perspective on Reinforcement Learning", 2017. [arXiv:1707.06887](https://arxiv.org/abs/1707.06887)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `32` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `num_atoms` | `51` | 分布原子数 |
| `v_min` | `-10.0` | 值分布下界 |
| `v_max` | `10.0` | 值分布上界 |

#### YAML 配置示例

```yaml
algorithm: c51_dqn
algo_kwargs:
  learning_rate: 1e-4
  buffer_size: 1_000_000
  batch_size: 32
  num_atoms: 51
  v_min: -10.0
  v_max: 10.0
```

#### Python API 示例

```python
from axiomrl.experimental import C51DQN

model = C51DQN(
    env_id="Pong-v5",
    num_atoms=51,
    v_min=-10.0,
    v_max=10.0,
)
model.train(total_timesteps=10_000_000)
```

!!! tip "最佳实践"
    - `num_atoms=51` 是论文推荐值，增加原子数可以提高精度但增加计算量。
    - `v_min` 和 `v_max` 的设置对性能有重要影响，需要覆盖实际回报范围。

---

### QR-DQN

**Quantile Regression DQN（分位数回归 DQN）**

使用分位数回归来学习值分布，相比 C51 无需设定值分布的上下界，参数化更灵活。

> Dabney et al., "Distributional Reinforcement Learning with Quantile Regression", 2017. [arXiv:1710.10044](https://arxiv.org/abs/1710.10044)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `5e-5` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `32` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `n_quantiles` | `200` | 分位数数量 |
| `target_update_interval` | `10_000` | 目标网络更新频率 |

#### YAML 配置示例

```yaml
algorithm: qr_dqn
algo_kwargs:
  learning_rate: 5e-5
  buffer_size: 1_000_000
  batch_size: 32
  n_quantiles: 200
  target_update_interval: 10_000
```

#### Python API 示例

```python
from axiomrl.experimental import QRDQN

model = QRDQN(
    env_id="Asterix-v5",
    n_quantiles=200,
    learning_rate=5e-5,
)
model.train(total_timesteps=10_000_000)
```

!!! tip "最佳实践"
    - QR-DQN 无需手动设置 `v_min`/`v_max`，使用更方便。
    - 增加 `n_quantiles` 可以更精确地逼近值分布。
    - 通常优于 C51 DQN，推荐作为分布型 DQN 的首选。

---

### IQN

**Implicit Quantile Network（隐式分位数网络）**

将分位数视为连续变量进行采样，使网络能够学习完整的分位数函数而非固定数量的分位数。

> Dabney et al., "Implicit Quantile Networks for Distributional Reinforcement Learning", 2018. [arXiv:1806.06923](https://arxiv.org/abs/1806.06923)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `5e-5` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `32` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `n_quantile_samples` | `64` | 训练时采样的分位数个数 |
| `n_target_quantile_samples` | `64` | 目标分位数采样个数 |
| `embedding_dim` | `64` | 分位数嵌入维度 |

#### YAML 配置示例

```yaml
algorithm: iqn
algo_kwargs:
  learning_rate: 5e-5
  buffer_size: 1_000_000
  batch_size: 32
  n_quantile_samples: 64
  n_target_quantile_samples: 64
  embedding_dim: 64
```

#### Python API 示例

```python
from axiomrl.experimental import IQN

model = IQN(
    env_id="Seaquest-v5",
    n_quantile_samples=64,
    embedding_dim=64,
)
model.train(total_timesteps=10_000_000)
```

!!! tip "最佳实践"
    - IQN 在 Atari 上通常优于 QR-DQN 和 C51。
    - 训练时的分位数采样数影响值估计精度和计算成本的平衡。
    - 支持风险敏感决策，可以通过调整采样分位数范围来实现。

---

### FQF

**Fully Parameterized Quantile Function（全参数化分位数函数）**

在 IQN 的基础上进一步学习最优的分位数位置（概率），使分位数分布自适应地集中在值分布的关键区域。

> Yang et al., "Fully Parameterized Quantile Function for Distributional Reinforcement Learning", 2019. [arXiv:1911.02140](https://arxiv.org/abs/1911.02140)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `5e-5` | 值网络学习率 |
| `fraction_lr` | `1e-5` | 分位数位置网络学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `32` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `n_quantiles` | `32` | 分位数数量 |
| `embedding_dim` | `64` | 嵌入维度 |

#### YAML 配置示例

```yaml
algorithm: fqf
algo_kwargs:
  learning_rate: 5e-5
  fraction_lr: 1e-5
  buffer_size: 1_000_000
  batch_size: 32
  n_quantiles: 32
  embedding_dim: 64
```

#### Python API 示例

```python
from axiomrl.experimental import FQF

model = FQF(
    env_id="BeamRider-v5",
    n_quantiles=32,
    fraction_lr=1e-5,
)
model.train(total_timesteps=10_000_000)
```

!!! tip "最佳实践"
    - FQF 是分布型 RL 方法的最新发展，理论上优于 IQN。
    - `fraction_lr` 通常设为主学习率的 1/5 ~ 1/10。
    - 较少的分位数（32）加上自适应位置，通常比 IQN 的固定 64 个分位数更高效。

---

### R2D2

**Recurrent Replay Distributed DQN（循环回放分布式 DQN）**

将 LSTM 循环网络集成到分布式 DQN 框架中，通过存储和回放序列来处理部分可观测环境。

> Kapturowski et al., "Recurrent Experience Replay in Distributed Reinforcement Learning", 2018. [arXiv:1807.04742](https://arxiv.org/abs/1807.04742)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `64` | 小批量大小（序列数） |
| `gamma` | `0.997` | 折扣因子 |
| `sequence_length` | `80` | 训练序列长度 |
| `burn_in_length` | `40` | 隐藏状态预热长度 |
| `lstm_hidden_size` | `512` | LSTM 隐藏层大小 |
| `num_actors` | `256` | 并行 Actor 数量 |

#### YAML 配置示例

```yaml
algorithm: r2d2
algo_kwargs:
  learning_rate: 1e-4
  buffer_size: 1_000_000
  batch_size: 64
  gamma: 0.997
  sequence_length: 80
  burn_in_length: 40
  lstm_hidden_size: 512
  num_actors: 256
```

#### Python API 示例

```python
from axiomrl.experimental import R2D2

model = R2D2(
    env_id="MontezumaRevenge-v5",
    sequence_length=80,
    burn_in_length=40,
    lstm_hidden_size=512,
    num_actors=256,
)
model.train(total_timesteps=50_000_000)
```

!!! tip "最佳实践"
    - R2D2 面向大规模分布式训练设计，单机性能有限。
    - `burn_in_length` 用于在序列开始时恢复 LSTM 隐藏状态。
    - 适合需要长期记忆的部分可观测环境。

---

### DRQN

**Deep Recurrent Q-Network（深度循环 Q 网络）**

在 DQN 的全连接层之前加入 LSTM 层，使得智能体能够处理部分可观测的马尔可夫决策过程。

> Hausknecht & Stone, "Deep Recurrent Q-Learning for Partially Observable MDPs", 2015. [arXiv:1507.06527](https://arxiv.org/abs/1507.06527)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` | 学习率 |
| `buffer_size` | `500_000` | 回放缓冲区大小 |
| `batch_size` | `32` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `sequence_length` | `10` | 训练序列长度 |
| `lstm_hidden_size` | `256` | LSTM 隐藏层大小 |

#### YAML 配置示例

```yaml
algorithm: drqn
algo_kwargs:
  learning_rate: 1e-4
  buffer_size: 500_000
  batch_size: 32
  sequence_length: 10
  lstm_hidden_size: 256
```

#### Python API 示例

```python
from axiomrl.experimental import DRQN

model = DRQN(
    env_id="FlickeringPong-v0",
    sequence_length=10,
    lstm_hidden_size=256,
)
model.train(total_timesteps=5_000_000)
```

!!! tip "最佳实践"
    - DRQN 是处理 POMDP 的最简单 DQN 扩展。
    - 对于大规模 POMDP 任务，推荐使用更先进的 R2D2。
    - `sequence_length` 应覆盖环境中信息隐藏的时间跨度。

---

### Agent57

**Agent57**

首个在所有 57 款 Atari 游戏上超越人类平均水平的智能体。通过自适应的元控制器在探索和利用之间切换。

> Badia et al., "Agent57: Outperforming the Atari Human Benchmark", 2020. [arXiv:2003.13350](https://arxiv.org/abs/2003.13350)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `64` | 小批量大小 |
| `gamma` | `0.997` | 折扣因子 |
| `num_policies` | `32` | 混合策略数量 |
| `beta_range` | `[0.0, 0.3]` | 内在奖励权重范围 |
| `discount_range` | `[0.99, 0.9999]` | 折扣因子范围 |
| `ucb_window` | `90` | UCB 滑窗大小 |

#### YAML 配置示例

```yaml
algorithm: agent57
algo_kwargs:
  learning_rate: 1e-4
  buffer_size: 1_000_000
  batch_size: 64
  gamma: 0.997
  num_policies: 32
  beta_range: [0.0, 0.3]
  discount_range: [0.99, 0.9999]
```

#### Python API 示例

```python
from axiomrl.experimental import Agent57

model = Agent57(
    env_id="Pitfall-v5",
    num_policies=32,
    beta_range=[0.0, 0.3],
)
model.train(total_timesteps=100_000_000)
```

!!! tip "最佳实践"
    - Agent57 需要非常大的计算量，适合充足计算资源下的研究使用。
    - 自适应策略混合是其核心，`num_policies` 越多探索和利用的粒度越细。
    - 在困难探索任务（如 Pitfall、Montezuma's Revenge）上效果显著。

---

### SPR

**Self-Predictive Representations（自预测表示）**

通过自监督时序预测学习高质量状态表示，在 Atari 100k 基准上取得 SOTA 表现，显著提升采样效率。

> Schwarzer et al., "Data-Efficient Reinforcement Learning with Self-Predictive Representations", 2020. [arXiv:2007.05929](https://arxiv.org/abs/2007.05929)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` | 学习率 |
| `buffer_size` | `100_000` | 回放缓冲区大小 |
| `batch_size` | `32` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `projection_dim` | `256` | 投影头维度 |
| `prediction_depth` | `5` | 预测未来步数 |
| `spr_coef` | `2.0` | SPR 损失权重 |
| `encoder_tau` | `0.01` | 目标编码器动量系数 |

#### YAML 配置示例

```yaml
algorithm: spr
algo_kwargs:
  learning_rate: 1e-4
  buffer_size: 100_000
  batch_size: 32
  projection_dim: 256
  prediction_depth: 5
  spr_coef: 2.0
  encoder_tau: 0.01
```

#### Python API 示例

```python
from axiomrl.experimental import SPR

model = SPR(
    env_id="Pong-v5",
    projection_dim=256,
    prediction_depth=5,
    spr_coef=2.0,
)
model.train(total_timesteps=100_000)
```

!!! tip "最佳实践"
    - SPR 在 Atari 100k（低数据量）基准上表现最佳。
    - `prediction_depth` 控制时序预测的步数，通常 3~5 步即可。
    - 自预测表示学习可以与其他 DQN 改进组合使用。
