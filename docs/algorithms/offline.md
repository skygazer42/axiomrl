---
title: 离线 RL 算法
icon: material/harddisk
---

# 离线强化学习算法（Offline RL）

## 离线强化学习范式

离线强化学习（也称为批量强化学习）是一类从**固定的历史数据集**中学习策略的方法，在训练过程中**不与环境进行任何在线交互**。这一范式在以下场景中至关重要：

- **在线交互代价高昂或存在风险**：如医疗决策、自动驾驶、金融交易等。
- **已有大量历史数据**：如工业控制系统的运行日志、用户行为数据等。
- **安全约束**：探索可能导致不可接受的后果。

离线 RL 的核心挑战在于**分布偏移**（Distribution Shift）：策略可能会选择训练数据中未覆盖的动作，导致值函数对这些未知动作产生不可靠的高估。本页中的算法通过不同的技术手段来应对这一挑战。

### 数据集配置

AxiomRL 支持多种离线数据集格式：

```yaml
# 使用本地 NPZ/PT 文件
algo_kwargs:
  dataset_path: "data/d4rl_hopper_medium.npz"

# 使用 PyTorch 格式
algo_kwargs:
  dataset_path: "data/offline_dataset.pt"

# 使用 Minari 数据集
algo_kwargs:
  minari_dataset: "hopper-medium-v2"
```

!!! info "数据格式说明"
    - **NPZ 格式**：包含 `observations`, `actions`, `rewards`, `next_observations`, `terminals` 数组。
    - **PT 格式**：PyTorch 保存的字典，键与 NPZ 格式相同。
    - **Minari**：标准化的离线 RL 数据集管理器，自动下载和加载。

---

## IQL

**Implicit Q-Learning（隐式 Q 学习）**

通过使用期望分位数回归来避免对未见动作的查询，完全不需要评估行为策略之外的动作值。方法简洁优雅，性能出色。

> Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning", 2021. [arXiv:2110.06169](https://arxiv.org/abs/2110.06169)

**稳定性：** <span class="badge badge-stable">Core</span> &nbsp; **动作空间：** Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 学习率 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `tau` | `0.005` | 目标网络软更新系数 |
| `expectile` | `0.7` | 期望分位数（核心参数） |
| `temperature` | `3.0` | AWR 抽取策略的温度 |
| `dataset_path` | 必填 | 离线数据集路径 |

### YAML 配置示例

```yaml
algorithm: iql
algo_kwargs:
  learning_rate: 3e-4
  batch_size: 256
  gamma: 0.99
  tau: 0.005
  expectile: 0.7
  temperature: 3.0
  dataset_path: "data/hopper_medium_expert.npz"
```

### Python API 示例

```python
from axiomrl.core import IQL

model = IQL(
    env_id="Hopper-v4",
    dataset_path="data/hopper_medium_expert.npz",
    expectile=0.7,
    temperature=3.0,
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - `expectile` 是最关键的超参数。值越高（如 0.9），策略越激进；值越低（如 0.5），越保守。
    - IQL 不需要评估离策略动作，因此对分布偏移的鲁棒性最强。
    - 在 D4RL 基准上，`expectile=0.7` 是多数任务的最优起点。

---

## CQL

**Conservative Q-Learning（保守 Q 学习）**

通过在 Q 函数的训练目标中添加一个正则化项来惩罚对分布外动作的高 Q 值估计，从而学习到 Q 函数的下界。

> Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning", 2020. [arXiv:2006.04779](https://arxiv.org/abs/2006.04779)

**稳定性：** <span class="badge badge-stable">Core</span> &nbsp; **动作空间：** Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 学习率 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `tau` | `0.005` | 目标网络软更新系数 |
| `cql_alpha` | `5.0` | CQL 正则化系数 |
| `cql_temp` | `1.0` | CQL logsumexp 温度 |
| `num_random_actions` | `10` | 随机动作采样数 |
| `with_lagrange` | `false` | 是否使用拉格朗日自动调节 alpha |
| `lagrange_threshold` | `10.0` | 拉格朗日约束阈值 |

### YAML 配置示例

```yaml
algorithm: cql
algo_kwargs:
  learning_rate: 3e-4
  batch_size: 256
  gamma: 0.99
  cql_alpha: 5.0
  cql_temp: 1.0
  num_random_actions: 10
  with_lagrange: false
  dataset_path: "data/walker_medium.npz"
```

### Python API 示例

```python
from axiomrl.core import CQL

model = CQL(
    env_id="Walker2d-v4",
    dataset_path="data/walker_medium.npz",
    cql_alpha=5.0,
    with_lagrange=False,
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - `cql_alpha` 控制保守程度，值越大越保守。通常在 1.0~10.0 之间调整。
    - 使用 `with_lagrange=true` 可以自动调节 `cql_alpha`，减少超参数调优负担。
    - 数据集质量越差（如 random 数据集），需要越大的 `cql_alpha`。

---

## BC

**Behavioral Cloning（行为克隆）**

最简单的模仿学习方法，将策略学习问题转化为有监督的学习问题，直接在专家演示数据上最小化动作预测误差。

> Pomerleau, "ALVINN: An Autonomous Land Vehicle in a Neural Network", 1988.

**稳定性：** <span class="badge badge-stable">Core</span> &nbsp; **动作空间：** Discrete + Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-3` | 学习率 |
| `batch_size` | `256` | 小批量大小 |
| `n_epochs` | `100` | 训练轮数 |
| `weight_decay` | `1e-4` | L2 正则化系数 |
| `dataset_path` | 必填 | 专家演示数据路径 |
| `loss_type` | `"mse"` | 损失类型（`mse` / `nll` / `cross_entropy`） |

### YAML 配置示例

```yaml
algorithm: bc
algo_kwargs:
  learning_rate: 1e-3
  batch_size: 256
  n_epochs: 100
  weight_decay: 1e-4
  dataset_path: "data/expert_demos.npz"
  loss_type: "mse"
```

### Python API 示例

```python
from axiomrl.core import BC

model = BC(
    env_id="Hopper-v4",
    dataset_path="data/expert_demos.npz",
    learning_rate=1e-3,
    n_epochs=100,
)
model.train()
```

!!! tip "最佳实践"
    - BC 是最强的基线方法之一，尤其在专家数据质量高时。
    - 连续动作空间用 `loss_type="mse"` 或 `"nll"`，离散动作空间用 `"cross_entropy"`。
    - 容易产生复合误差（compounding error），可以通过 DAgger 或配合其他方法缓解。
    - 适合作为离线 RL 其他方法的初始化策略。

---

## BCQ

**Batch-Constrained Q-learning（批量约束 Q 学习）**

通过训练一个 VAE 来建模行为策略的动作分布，将策略优化限制在行为策略支撑集内，避免对分布外动作的查询。

> Fujimoto et al., "Off-Policy Deep Reinforcement Learning without Exploration", 2019. [arXiv:1812.02900](https://arxiv.org/abs/1812.02900)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-3` | 学习率 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `tau` | `0.005` | 目标网络软更新系数 |
| `latent_dim` | `2` | VAE 潜在空间维度倍数（`action_dim * latent_dim`） |
| `phi` | `0.05` | 扰动范围 |
| `num_sampled_actions` | `100` | 候选动作采样数 |

### YAML 配置示例

```yaml
algorithm: bcq
algo_kwargs:
  learning_rate: 1e-3
  batch_size: 256
  gamma: 0.99
  latent_dim: 2
  phi: 0.05
  num_sampled_actions: 100
  dataset_path: "data/halfcheetah_medium.npz"
```

### Python API 示例

```python
from axiomrl.experimental import BCQ

model = BCQ(
    env_id="HalfCheetah-v4",
    dataset_path="data/halfcheetah_medium.npz",
    phi=0.05,
    num_sampled_actions=100,
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - `phi` 控制允许的动作偏移范围，越小越保守。
    - `num_sampled_actions` 影响动作选择质量和计算成本的平衡。
    - BCQ 是首个广泛认可的离线 RL 算法，但在较新基准上已被 IQL/CQL 等超越。

---

## BEAR

**Bootstrapping Error Accumulation Reduction（自举误差累积抑制）**

通过 MMD（最大均值差异）约束将学习策略的动作分布限制在行为策略支撑集附近，使用集成 Q 网络来估计不确定性。

> Kumar et al., "Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction", 2019. [arXiv:1906.00949](https://arxiv.org/abs/1906.00949)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 学习率 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `mmd_kernel` | `"laplacian"` | MMD 核函数类型 |
| `mmd_epsilon` | `0.05` | MMD 约束阈值 |
| `num_samples` | `10` | 策略采样数 |
| `lagrange_lr` | `1e-3` | 拉格朗日乘子学习率 |
| `n_critics` | `4` | 集成评论家数量 |

### YAML 配置示例

```yaml
algorithm: bear
algo_kwargs:
  learning_rate: 3e-4
  batch_size: 256
  gamma: 0.99
  mmd_kernel: "laplacian"
  mmd_epsilon: 0.05
  num_samples: 10
  n_critics: 4
  dataset_path: "data/ant_medium_expert.npz"
```

### Python API 示例

```python
from axiomrl.experimental import BEAR

model = BEAR(
    env_id="Ant-v4",
    dataset_path="data/ant_medium_expert.npz",
    mmd_kernel="laplacian",
    mmd_epsilon=0.05,
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - `mmd_epsilon` 是约束的松紧程度，较小值更保守。
    - `laplacian` 核在实践中通常优于 `gaussian` 核。
    - 集成评论家增加计算成本但提升值估计可靠性。

---

## CRR

**Critic Regularized Regression（评论家正则化回归）**

利用学到的值函数来过滤或加权行为数据中的动作，以简单的加权回归方式改进策略。

> Wang et al., "Critic Regularized Regression", 2020. [arXiv:2006.15134](https://arxiv.org/abs/2006.15134)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 学习率 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `weight_type` | `"exp"` | 加权方式（`exp` / `binary`） |
| `temperature` | `1.0` | 指数加权的温度参数 |
| `advantage_type` | `"mean"` | 优势计算方式（`mean` / `max`） |
| `n_action_samples` | `4` | 优势估计的动作采样数 |

### YAML 配置示例

```yaml
algorithm: crr
algo_kwargs:
  learning_rate: 3e-4
  batch_size: 256
  gamma: 0.99
  weight_type: "exp"
  temperature: 1.0
  advantage_type: "mean"
  n_action_samples: 4
  dataset_path: "data/walker_medium_expert.npz"
```

### Python API 示例

```python
from axiomrl.experimental import CRR

model = CRR(
    env_id="Walker2d-v4",
    dataset_path="data/walker_medium_expert.npz",
    weight_type="exp",
    temperature=1.0,
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - `weight_type="binary"` 只保留正优势的动作，`"exp"` 则按优势大小加权。
    - 方法简单，但在数据质量较高的数据集上非常有效。
    - 可以作为离线 RL 的简单基线方法。

---

## TD3+BC

**TD3+BC**

在 TD3 的策略更新中简单地添加一个行为克隆正则化项，以极简的方式实现离线 RL。代码修改量极少但效果出色。

> Fujimoto & Gu, "A Minimalist Approach to Offline Reinforcement Learning", 2021. [arXiv:2106.06860](https://arxiv.org/abs/2106.06860)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 学习率 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `tau` | `0.005` | 目标网络软更新系数 |
| `alpha` | `2.5` | BC 正则化系数 |
| `normalize` | `true` | 是否对状态做标准化 |
| `policy_delay` | `2` | TD3 策略延迟 |

### YAML 配置示例

```yaml
algorithm: td3_bc
algo_kwargs:
  learning_rate: 3e-4
  batch_size: 256
  gamma: 0.99
  alpha: 2.5
  normalize: true
  policy_delay: 2
  dataset_path: "data/hopper_medium_replay.npz"
```

### Python API 示例

```python
from axiomrl.experimental import TD3BC

model = TD3BC(
    env_id="Hopper-v4",
    dataset_path="data/hopper_medium_replay.npz",
    alpha=2.5,
    normalize=True,
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - TD3+BC 的极简设计使其成为离线 RL 的绝佳起点和基线。
    - `alpha` 控制 RL 目标和 BC 正则化的平衡，`alpha=2.5` 在多数任务上工作良好。
    - 状态标准化 (`normalize=true`) 对性能至关重要。

---

## AWAC

**Advantage Weighted Actor-Critic（优势加权演员-评论家）**

在 AWR 的基础上融合离线数据预训练和在线微调，通过约束策略更新到数据分布附近来实现平滑的离线到在线过渡。

> Nair et al., "AWAC: Accelerating Online Reinforcement Learning with Offline Datasets", 2020. [arXiv:2006.09359](https://arxiv.org/abs/2006.09359)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 学习率 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `tau` | `0.005` | 目标网络软更新系数 |
| `lambda_awac` | `1.0` | 优势加权温度 |
| `n_action_samples` | `1` | 优势估计采样数 |
| `offline_steps` | `100_000` | 离线预训练步数 |

### YAML 配置示例

```yaml
algorithm: awac
algo_kwargs:
  learning_rate: 3e-4
  batch_size: 256
  gamma: 0.99
  lambda_awac: 1.0
  offline_steps: 100_000
  dataset_path: "data/pen_demos.npz"
```

### Python API 示例

```python
from axiomrl.experimental import AWAC

model = AWAC(
    env_id="AdroitHandPen-v1",
    dataset_path="data/pen_demos.npz",
    lambda_awac=1.0,
    offline_steps=100_000,
)
model.train(total_timesteps=500_000)
```

!!! tip "最佳实践"
    - AWAC 特别适合"先离线后在线"的场景。
    - `lambda_awac` 控制对高优势动作的偏好，较小值更激进。
    - 在灵巧手操控等任务上表现突出。

---

## ReBRAC

**ReBRAC**

通过在 Actor 和 Critic 上同时施加 BC 正则化来简化离线 RL，结合仔细的超参数选择在 D4RL 基准上取得 SOTA 表现。

> Tarasov et al., "Revisiting the Minimalist Approach to Offline Reinforcement Learning", 2023. [arXiv:2305.09836](https://arxiv.org/abs/2305.09836)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` | 学习率 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `tau` | `0.005` | 目标网络软更新系数 |
| `actor_bc_coef` | `0.2` | Actor BC 正则化系数 |
| `critic_bc_coef` | `0.2` | Critic BC 正则化系数 |
| `normalize` | `true` | 状态标准化 |

### YAML 配置示例

```yaml
algorithm: rebrac
algo_kwargs:
  learning_rate: 1e-4
  batch_size: 256
  gamma: 0.99
  actor_bc_coef: 0.2
  critic_bc_coef: 0.2
  normalize: true
  dataset_path: "data/halfcheetah_medium_expert.npz"
```

### Python API 示例

```python
from axiomrl.experimental import ReBRAC

model = ReBRAC(
    env_id="HalfCheetah-v4",
    dataset_path="data/halfcheetah_medium_expert.npz",
    actor_bc_coef=0.2,
    critic_bc_coef=0.2,
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - ReBRAC 是 TD3+BC 的进化版本，在 Actor 和 Critic 两侧都加了正则化。
    - 在 D4RL 基准上与更复杂的方法竞争甚至超越。
    - `actor_bc_coef` 和 `critic_bc_coef` 需要针对不同任务调整。

---

## XQL

**Extreme Q-Learning（极端 Q 学习）**

利用极端值理论来替代传统的 logsumexp 操作，避免对分布外动作的显式查询，以更稳定的方式估计最优值函数。

> Garg et al., "Extreme Q-Learning: MaxEnt RL without Entropy", 2023. [arXiv:2301.02328](https://arxiv.org/abs/2301.02328)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 学习率 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `tau` | `0.005` | 目标网络软更新系数 |
| `temperature` | `10.0` | Gumbel 分布温度参数 |
| `noise_std` | `1.0` | 值噪声标准差 |

### YAML 配置示例

```yaml
algorithm: xql
algo_kwargs:
  learning_rate: 3e-4
  batch_size: 256
  gamma: 0.99
  temperature: 10.0
  noise_std: 1.0
  dataset_path: "data/ant_medium.npz"
```

### Python API 示例

```python
from axiomrl.experimental import XQL

model = XQL(
    env_id="Ant-v4",
    dataset_path="data/ant_medium.npz",
    temperature=10.0,
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - XQL 的理论基础独特，基于极端值分布而非常规的最大熵框架。
    - `temperature` 控制近似精度，需要针对任务调优。
    - 与 IQL 类似，不需要在训练时查询分布外动作。

---

## EDAC

**Error-Diversified Ensemble Actor-Critic（误差多样化集成演员-评论家）**

通过鼓励集成 Q 网络之间的梯度多样性来产生更可靠的不确定性估计，用于惩罚分布外动作。

> An et al., "Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble", 2021. [arXiv:2110.01548](https://arxiv.org/abs/2110.01548)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 学习率 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `tau` | `0.005` | 目标网络软更新系数 |
| `n_critics` | `10` | 集成评论家数量 |
| `eta` | `1.0` | 不确定性惩罚系数 |
| `diversity_coef` | `1.0` | 梯度多样性正则系数 |

### YAML 配置示例

```yaml
algorithm: edac
algo_kwargs:
  learning_rate: 3e-4
  batch_size: 256
  gamma: 0.99
  n_critics: 10
  eta: 1.0
  diversity_coef: 1.0
  dataset_path: "data/walker_medium_replay.npz"
```

### Python API 示例

```python
from axiomrl.experimental import EDAC

model = EDAC(
    env_id="Walker2d-v4",
    dataset_path="data/walker_medium_replay.npz",
    n_critics=10,
    eta=1.0,
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - 集成评论家数量 (`n_critics`) 越多，不确定性估计越可靠，但计算成本也越高。
    - `eta` 控制保守程度，较大值更保守。
    - 在多数 D4RL 任务上优于 CQL，尤其在 medium-replay 数据集上。

---

## Cal-QL

**Calibrated Conservative Q-Learning（校准保守 Q 学习）**

在 CQL 的基础上校准保守正则化的强度，使其不过度保守，通过自适应地调整下界来逼近真实值函数。

> Nakamoto et al., "Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning", 2023. [arXiv:2303.05479](https://arxiv.org/abs/2303.05479)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 学习率 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `cql_alpha` | `5.0` | CQL 正则化系数 |
| `calibration_mode` | `"mcret"` | 校准方式 (`mcret` / `value`) |
| `with_lagrange` | `true` | 使用拉格朗日自动调节 |
| `lagrange_threshold` | `5.0` | 拉格朗日阈值 |

### YAML 配置示例

```yaml
algorithm: cal_ql
algo_kwargs:
  learning_rate: 3e-4
  batch_size: 256
  gamma: 0.99
  cql_alpha: 5.0
  calibration_mode: "mcret"
  with_lagrange: true
  lagrange_threshold: 5.0
  dataset_path: "data/antmaze_large.npz"
```

### Python API 示例

```python
from axiomrl.experimental import CalQL

model = CalQL(
    env_id="AntMaze-large-diverse-v2",
    dataset_path="data/antmaze_large.npz",
    cql_alpha=5.0,
    calibration_mode="mcret",
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - Cal-QL 在"离线预训练 + 在线微调"场景中表现最佳。
    - 校准机制使得 Q 值的下界不会过度保守，有利于在线微调时的快速适应。
    - 在 AntMaze 等难度较高的任务上显著优于标准 CQL。

---

## Decision Transformer

**Decision Transformer（决策 Transformer）**

将离线 RL 重新定义为序列建模问题，利用 Transformer 架构根据期望回报、历史状态和动作来预测下一个动作。

> Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling", 2021. [arXiv:2106.01345](https://arxiv.org/abs/2106.01345)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` | 学习率 |
| `batch_size` | `64` | 小批量大小 |
| `context_length` | `20` | 上下文序列长度 |
| `n_layers` | `3` | Transformer 层数 |
| `n_heads` | `1` | 注意力头数 |
| `embed_dim` | `128` | 嵌入维度 |
| `dropout` | `0.1` | Dropout 率 |
| `target_return` | 必填 | 目标累积回报（推理时使用） |

### YAML 配置示例

```yaml
algorithm: decision_transformer
algo_kwargs:
  learning_rate: 1e-4
  batch_size: 64
  context_length: 20
  n_layers: 3
  n_heads: 1
  embed_dim: 128
  dropout: 0.1
  target_return: 3600.0
  dataset_path: "data/hopper_medium_expert.npz"
```

### Python API 示例

```python
from axiomrl.experimental import DecisionTransformer

model = DecisionTransformer(
    env_id="Hopper-v4",
    dataset_path="data/hopper_medium_expert.npz",
    context_length=20,
    n_layers=3,
    target_return=3600.0,
)
model.train(n_epochs=20)
```

!!! tip "最佳实践"
    - `target_return` 在推理时控制期望性能水平，可以设为数据集中最优轨迹的回报。
    - Decision Transformer 不需要 Bellman 方程或时序差分学习。
    - `context_length` 影响决策的时序范围和计算成本。
    - 在需要长程规划的任务上优势明显。
