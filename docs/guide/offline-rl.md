---
title: 离线 RL 指南
icon: material/database
---

# 离线 RL 指南

本章节介绍如何使用 AxiomRL 进行离线强化学习（Offline RL），从固定数据集中学习策略而无需与环境交互。

## 离线 RL 概述

离线强化学习（又称批量强化学习）从预先收集的数据集中学习策略，不再需要与环境进行在线交互。这在以下场景中特别有用：

```mermaid
graph LR
    A[预收集数据集] --> B[离线 RL 算法]
    B --> C[学习策略]
    C --> D[部署/评估]

    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#e8f5e9
    style D fill:#fce4ec
```

- **安全敏感场景**：医疗决策、自动驾驶等无法随意试错的领域
- **数据已有场景**：从历史日志或专家演示中学习
- **成本高昂场景**：真实环境交互成本过高

!!! tip "离线 RL vs 模仿学习"

    模仿学习（如 BC）仅从专家数据中学习，而离线 RL 算法（如 IQL、CQL）能够从混合质量的数据集中学习，甚至可能学到超越数据收集策略的行为。

## 支持的数据格式

AxiomRL 支持三种主流离线数据格式：

| 格式 | 扩展名 | 说明 | 适用场景 |
|------|--------|------|----------|
| **NPZ** | `.npz` | NumPy 压缩数组格式 | 自定义数据集、轻量级数据 |
| **PT** | `.pt` | PyTorch 序列化格式 | 与 PyTorch 生态集成 |
| **Minari** | - | Farama 基金会标准格式 | 标准化基准数据集（D4RL 继任者） |

### NPZ 数据格式

NPZ 文件需要包含以下键：

```python
{
    "observations": np.ndarray,     # 形状: (N, obs_dim)
    "actions": np.ndarray,          # 形状: (N, act_dim)
    "rewards": np.ndarray,          # 形状: (N,)
    "next_observations": np.ndarray,  # 形状: (N, obs_dim)
    "terminals": np.ndarray,        # 形状: (N,)  布尔值
}
```

### PT 数据格式

PT 文件存储一个包含相同键的 Python 字典，值为 PyTorch 张量。

### Minari 数据集

通过 Minari 数据集名称直接引用：

```bash
# 安装 Minari
pip install minari

# 下载数据集
minari download pointmaze-medium-v2
```

## 配置方法

### 使用本地数据集（NPZ / PT）

通过 `algo_kwargs.dataset_path` 指定数据集文件路径：

```yaml title="offline_npz.yaml"
algo: IQL
env_id: Hopper-v4
seed: 42
total_timesteps: 1_000_000
output_dir: runs/

algo_kwargs:
  dataset_path: data/hopper_medium.npz
  learning_rate: 3.0e-4
  batch_size: 256
  gamma: 0.99
```

### 使用 Minari 数据集

通过 `algo_kwargs.minari_dataset` 指定 Minari 数据集名称：

```yaml title="offline_minari.yaml"
algo: CQL
env_id: PointMaze_Medium-v3
seed: 42
total_timesteps: 1_000_000
output_dir: runs/

algo_kwargs:
  minari_dataset: pointmaze-medium-v2
  learning_rate: 3.0e-4
  batch_size: 256
  gamma: 0.99
```

!!! warning "环境 ID 匹配"

    确保 `env_id` 与数据集收集时使用的环境一致，否则观测空间和动作空间可能不匹配。

## 支持的离线 RL 算法

AxiomRL 提供了丰富的离线 RL 算法实现：

| 算法 | 类别 | 说明 |
|------|------|------|
| **BC** | 模仿学习 | 行为克隆，直接从数据中监督学习策略 |
| **IQL** | 离线值函数 | Implicit Q-Learning，避免查询分布外动作 |
| **CQL** | 保守估计 | Conservative Q-Learning，惩罚分布外动作的 Q 值 |
| **BCQ** | 批量约束 | Batch-Constrained Q-Learning |
| **BEAR** | 支持约束 | Bootstrapping Error Accumulation Reduction |
| **CRR** | 约束回归 | Critic Regularized Regression |
| **TD3+BC** | 正则化 | TD3 with Behavior Cloning 正则化 |
| **AWAC** | 优势加权 | Advantage Weighted Actor-Critic |
| **ReBRAC** | 正则化 | Revised BRAC，改进的行为正则化 |
| **XQL** | Gumbel 回归 | Extreme Q-Learning |
| **EDAC** | 集成方法 | Ensemble-Diversified Actor Critic |
| **Cal-QL** | 校准保守 | Calibrated Conservative Q-Learning |
| **MOPO** | 模型基础 | Model-based Offline Policy Optimization |
| **Decision Transformer** | 序列建模 | 将 RL 视为序列建模问题 |

## YAML 配置示例

### IQL + NPZ 数据集

```yaml title="iql_npz.yaml" linenums="1"
algo: IQL
env_id: Hopper-v4
seed: 42
total_timesteps: 1_000_000
output_dir: runs/

algo_kwargs:
  dataset_path: data/hopper_medium.npz
  learning_rate: 3.0e-4
  batch_size: 256
  gamma: 0.99
  tau: 0.005
  expectile: 0.7        # IQL 特有参数
  temperature: 3.0       # IQL 温度参数
```

### CQL + Minari 数据集

```yaml title="cql_minari.yaml" linenums="1"
algo: CQL
env_id: PointMaze_Medium-v3
seed: 42
total_timesteps: 500_000
output_dir: runs/

algo_kwargs:
  minari_dataset: pointmaze-medium-v2
  learning_rate: 1.0e-4
  batch_size: 256
  gamma: 0.99
  cql_alpha: 5.0         # CQL 保守惩罚系数
  cql_tau: 10.0
  num_random_actions: 10  # 用于 CQL 损失的随机动作数
```

### Decision Transformer

```yaml title="dt_config.yaml" linenums="1"
algo: DecisionTransformer
env_id: HalfCheetah-v4
seed: 42
total_timesteps: 100_000
output_dir: runs/

algo_kwargs:
  dataset_path: data/halfcheetah_medium.npz
  learning_rate: 1.0e-4
  batch_size: 64
  context_length: 20     # 序列上下文长度
  embed_dim: 128         # 嵌入维度
  n_heads: 4             # 注意力头数
  n_layers: 3            # Transformer 层数
```

## Python API 示例

```python title="offline_train.py" linenums="1"
from axiomrl import TrainConfig, train

# IQL 离线训练
config = TrainConfig(
    algo="IQL",
    env_id="Hopper-v4",
    seed=42,
    total_timesteps=1_000_000,
    output_dir="runs/",
    algo_kwargs={
        "dataset_path": "data/hopper_medium.npz",
        "learning_rate": 3e-4,
        "batch_size": 256,
        "gamma": 0.99,
        "expectile": 0.7,
        "temperature": 3.0,
    },
)

train(config)
```

```python title="minari_train.py" linenums="1"
from axiomrl import TrainConfig, train

# CQL + Minari 数据集
config = TrainConfig(
    algo="CQL",
    env_id="PointMaze_Medium-v3",
    seed=42,
    total_timesteps=500_000,
    output_dir="runs/",
    algo_kwargs={
        "minari_dataset": "pointmaze-medium-v2",
        "learning_rate": 1e-4,
        "batch_size": 256,
        "cql_alpha": 5.0,
    },
)

train(config)
```

## 混合数据集

离线 RL 可以使用混合质量的数据集进行训练，例如将专家数据和随机策略数据合并：

```python title="merge_datasets.py" linenums="1"
import numpy as np

# 加载多个数据集
expert_data = np.load("data/hopper_expert.npz")
random_data = np.load("data/hopper_random.npz")

# 合并数据集
merged = {
    key: np.concatenate([expert_data[key], random_data[key]], axis=0)
    for key in expert_data.files
}

# 保存混合数据集
np.savez_compressed("data/hopper_mixed.npz", **merged)
```

```yaml title="使用混合数据集"
algo: IQL
env_id: Hopper-v4
seed: 42
total_timesteps: 1_000_000

algo_kwargs:
  dataset_path: data/hopper_mixed.npz
```

!!! tip "数据集质量与算法选择"

    - **专家数据**：BC 即可获得不错的结果
    - **中等质量数据**：IQL、CQL、TD3+BC 表现优异
    - **混合质量数据**：IQL、AWAC 能够提取高质量行为
    - **低质量数据**：CQL 的保守估计有助于避免分布外错误

## 奖励变换与归一化

部分场景需要对数据集中的奖励进行变换或归一化处理：

```yaml title="带奖励归一化的配置"
algo: IQL
env_id: Hopper-v4
seed: 42
total_timesteps: 1_000_000

algo_kwargs:
  dataset_path: data/hopper_medium.npz
  reward_scale: 1.0       # 奖励缩放系数
  reward_shift: 0.0       # 奖励偏移量
  normalize_reward: true   # 是否进行标准化
```

!!! info "常用奖励变换"

    | 参数 | 说明 |
    |------|------|
    | `reward_scale` | 对奖励乘以一个常数 |
    | `reward_shift` | 对奖励加上一个常数 |
    | `normalize_reward` | 将奖励标准化为零均值单位方差 |

!!! warning "数据集准备注意事项"

    1. 确保观测和动作的维度与目标环境一致
    2. 检查数据集中的 `terminals` 标记是否正确
    3. 对于连续动作空间，确认动作在合法范围内
    4. 大型数据集建议使用 NPZ 压缩格式以节省存储空间
    5. 使用 Minari 格式可获得更好的可复现性和标准化支持
