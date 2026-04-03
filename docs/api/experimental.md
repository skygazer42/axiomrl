---
title: 实验性 API
---

# 实验性 API

`rl_training.experimental` 提供 AxiomRL 全部 70 余种算法的访问入口。与稳定核心 API 不同，实验性 API 可能在次要版本之间发生变更。

## 概览

```python
from rl_training.experimental import EXPERIMENTAL_ALGORITHMS
```

`rl_training.experimental` 重新导出 `rl_training.api` 的所有内容，并额外提供 `EXPERIMENTAL_ALGORITHMS` 常量，其中包含所有不在 `STABLE_ALGORITHMS` 中的算法名称。

```python
from rl_training.experimental import EXPERIMENTAL_ALGORITHMS
from rl_training.core import STABLE_ALGORITHMS

# 实验性算法 = 全部算法 - 稳定核心算法
print(f"稳定核心算法数量: {len(STABLE_ALGORITHMS)}")    # 10
print(f"实验性算法数量: {len(EXPERIMENTAL_ALGORITHMS)}")  # 60+
```

---

## 导入方式

```python
# 导入特定的实验性算法
from rl_training.experimental import REDQ, TQC, CrossQ

# 导入所有可用算法
from rl_training.experimental import *

# 检查某个算法是否属于实验性
from rl_training.experimental import EXPERIMENTAL_ALGORITHMS
print("REDQ" in EXPERIMENTAL_ALGORITHMS)  # True
print("PPO" in EXPERIMENTAL_ALGORITHMS)   # False (PPO 在稳定核心中)
```

---

## 算法分类

以下按类别列出实验性 API 中提供的典型算法。

!!! note "列表说明"
    此处仅列出各类别中的代表性算法。完整列表请在运行时检查 `EXPERIMENTAL_ALGORITHMS` 常量。

### 在策略算法（On-Policy）

基于当前策略采样数据进行训练的算法。

| 算法 | 说明 |
|------|------|
| `PPO`* | 近端策略优化 |
| `A2C`* | 优势演员-评论家 |
| `TRPO`* | 信赖域策略优化 |
| `RPO` | 鲁棒策略优化 |
| `GRPO` | 分组相对策略优化 |

> 带 * 标记的算法同时属于稳定核心 API。

### 离策略算法（Off-Policy）

使用经验回放缓冲区、可利用历史数据进行训练的算法。

| 算法 | 说明 |
|------|------|
| `SAC`* | 柔性演员-评论家 |
| `TD3`* | 双延迟 DDPG |
| `DDPG` | 深度确定性策略梯度 |
| `TQC` | 截断分位数评论家 |
| `REDQ` | 随机集成双 Q 学习 |
| `CrossQ` | 交叉 Q 学习 |
| `DroQ` | 带 Dropout 的 Q 学习 |

### 基于值的算法（Value-Based）

以值函数估计为核心的算法。

| 算法 | 说明 |
|------|------|
| `DQN`* | 深度 Q 网络 |
| `DiscreteSAC`* | 离散 SAC |
| `C51` | 分布式 DQN |
| `QR-DQN` | 分位数回归 DQN |
| `Rainbow` | Rainbow DQN |
| `IQN` | 隐式分位数网络 |

### 离线 RL 算法（Offline）

从固定数据集中学习，不与环境交互。

| 算法 | 说明 |
|------|------|
| `BC`* | 行为克隆 |
| `CQL`* | 保守 Q 学习 |
| `IQL`* | 隐式 Q 学习 |
| `TD3+BC` | TD3 + 行为克隆正则化 |
| `AWAC` | 优势加权演员-评论家 |
| `ABM` | 基于优势的建模 |

### 基于模型的算法（Model-Based）

学习环境动力学模型以辅助决策。

| 算法 | 说明 |
|------|------|
| `MBPO` | 基于模型的策略优化 |
| `Dreamer` | 世界模型 + 想象空间学习 |
| `PETS` | 概率集成轨迹采样 |
| `SVG` | 随机值梯度 |

---

## rl_training.contrib - 社区扩展

`rl_training.contrib` 模块包含社区贡献的算法和工具，独立于核心和实验性 API 进行维护。

```python
from rl_training.contrib import RecurrentPPO
```

### 可用扩展

| 名称 | 说明 |
|------|------|
| `RecurrentPPO` | 循环 PPO，使用 LSTM/GRU 处理部分可观测环境 |
| `RecurrentPPOAlgorithm` | RecurrentPPO 的底层算法实现 |

```yaml title="使用 RecurrentPPO 的配置示例"
algo: RecurrentPPO
env_id: CartPole-v1
seed: 42
total_timesteps: 100000
output_dir: runs/recurrent-ppo
algo_kwargs:
  learning_rate: 0.0003
  n_steps: 128
  n_epochs: 10
```

!!! info "扩展贡献"
    contrib 模块欢迎社区贡献。如需添加新算法，请参考[贡献指南](../developer/contributing.md)。

---

## 根包弃用导入

!!! warning "弃用警告"
    从 `rl_training` 根包直接导入非稳定核心算法的方式已被弃用，将在未来主要版本中移除。

### 弃用行为

```python
# 稳定名称 - 正常工作，无警告
from rl_training import PPO       # OK
from rl_training import TrainConfig  # OK

# 实验性名称 - 触发 DeprecationWarning
from rl_training import TQC  # DeprecationWarning!
```

触发的警告信息如下：

```
DeprecationWarning: rl_training.TQC is no longer part of the stable root API
and is deprecated; import advanced algorithms from rl_training.experimental
or rl_training.api instead.
```

---

## 迁移指南

### 从根包导入迁移

将非稳定核心算法的导入从根包迁移到 `rl_training.experimental`：

```python
# 旧写法（弃用）
from rl_training import TQC, REDQ, CrossQ

# 新写法（推荐）
from rl_training.experimental import TQC, REDQ, CrossQ
```

### 稳定核心算法无需迁移

稳定核心算法可以从根包或 `rl_training.core` 导入，两种方式均受支持：

```python
# 以下两种写法均可
from rl_training import PPO
from rl_training.core import PPO
```

### 批量检查与迁移

```python
from rl_training.core import STABLE_ALGORITHMS
from rl_training.experimental import EXPERIMENTAL_ALGORITHMS

# 检查算法类别
algo_name = "TQC"
if algo_name in STABLE_ALGORITHMS:
    print(f"从 rl_training.core 导入 {algo_name}")
elif algo_name in EXPERIMENTAL_ALGORITHMS:
    print(f"从 rl_training.experimental 导入 {algo_name}")
```

!!! tip "消除警告"
    如需消除弃用警告，只需将非稳定核心算法的导入路径改为 `rl_training.experimental` 即可。
