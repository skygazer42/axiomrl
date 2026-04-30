---
title: 稳定核心 API
---

# 稳定核心 API

`axiomrl.core` 是 AxiomRL 的稳定核心 API 模块，受语义版本控制（semver）管理。在整个 1.x 发布周期内，此模块的公开接口保证向后兼容。

## 概览

稳定核心 API 导出 **10 种算法** 和 **TrainConfig** 配置类：

```python
from axiomrl.core import (
    A2C, BC, CQL, DQN, DiscreteSAC,
    IQL, PPO, SAC, TD3, TRPO,
    TrainConfig,
    STABLE_ALGORITHMS,
)
```

也可以从根包直接导入（等价方式）：

```python
from axiomrl import PPO, TrainConfig
```

---

## STABLE_ALGORITHMS

`STABLE_ALGORITHMS` 是一个元组常量，包含所有稳定核心算法的名称：

```python
from axiomrl.core import STABLE_ALGORITHMS

print(STABLE_ALGORITHMS)
# ('A2C', 'BC', 'CQL', 'DQN', 'DiscreteSAC', 'IQL', 'PPO', 'SAC', 'TD3', 'TRPO')
```

可用于运行时检查或批量操作：

```python
if algo_name in STABLE_ALGORITHMS:
    print(f"{algo_name} 是稳定核心算法")
```

---

## 算法参考

### A2C

**Advantage Actor-Critic** -- 同步的优势演员-评论家算法，适用于离散和连续动作空间。

```python
from axiomrl.core import A2C
```

```yaml
algo: A2C
env_id: CartPole-v1
seed: 42
total_timesteps: 100000
output_dir: runs/a2c
algo_kwargs:
  learning_rate: 0.0007
  n_steps: 5
  gamma: 0.99
  ent_coef: 0.01
```

---

### BC

**Behavioral Cloning** -- 行为克隆算法，从专家演示数据中学习策略的监督学习方法。

```python
from axiomrl.core import BC
```

```yaml
algo: BC
env_id: HalfCheetah-v4
seed: 42
total_timesteps: 100000
output_dir: runs/bc
algo_kwargs:
  learning_rate: 0.001
  batch_size: 256
```

---

### CQL

**Conservative Q-Learning** -- 保守 Q 学习算法，通过惩罚分布外动作的 Q 值来缓解离线 RL 中的外推误差。

```python
from axiomrl.core import CQL
```

```yaml
algo: CQL
env_id: HalfCheetah-v4
seed: 42
total_timesteps: 1000000
output_dir: runs/cql
algo_kwargs:
  learning_rate: 0.0003
  batch_size: 256
  cql_alpha: 1.0
```

---

### DQN

**Deep Q-Network** -- 深度 Q 网络算法，使用经验回放和目标网络进行值函数逼近，适用于离散动作空间。

```python
from axiomrl.core import DQN
```

```yaml
algo: DQN
env_id: CartPole-v1
seed: 42
total_timesteps: 100000
output_dir: runs/dqn
algo_kwargs:
  learning_rate: 0.0001
  batch_size: 32
  buffer_size: 100000
  target_update_interval: 500
  exploration_fraction: 0.1
```

---

### DiscreteSAC

**Discrete Soft Actor-Critic** -- 离散动作空间版本的 SAC 算法，结合最大熵强化学习框架。

```python
from axiomrl.core import DiscreteSAC
```

```yaml
algo: DiscreteSAC
env_id: CartPole-v1
seed: 42
total_timesteps: 100000
output_dir: runs/discrete-sac
algo_kwargs:
  learning_rate: 0.0003
  batch_size: 64
  buffer_size: 100000
  tau: 0.005
```

---

### IQL

**Implicit Q-Learning** -- 隐式 Q 学习算法，通过期望回归避免对分布外动作进行查询，适用于离线 RL。

```python
from axiomrl.core import IQL
```

```yaml
algo: IQL
env_id: HalfCheetah-v4
seed: 42
total_timesteps: 1000000
output_dir: runs/iql
algo_kwargs:
  learning_rate: 0.0003
  batch_size: 256
  tau: 0.005
  quantile: 0.7
```

---

### PPO

**Proximal Policy Optimization** -- 近端策略优化算法，通过裁剪目标函数实现稳定的策略更新。AxiomRL 中最常用的算法之一。

```python
from axiomrl.core import PPO
```

```yaml
algo: PPO
env_id: HalfCheetah-v4
seed: 42
total_timesteps: 1000000
output_dir: runs/ppo
algo_kwargs:
  learning_rate: 0.0003
  batch_size: 64
  n_steps: 2048
  n_epochs: 10
  gamma: 0.99
  clip_range: 0.2
  ent_coef: 0.0
```

---

### SAC

**Soft Actor-Critic** -- 柔性演员-评论家算法，最大熵框架下的离策略算法，在连续控制任务中表现优异。

```python
from axiomrl.core import SAC
```

```yaml
algo: SAC
env_id: HalfCheetah-v4
seed: 42
total_timesteps: 1000000
output_dir: runs/sac
algo_kwargs:
  learning_rate: 0.0003
  batch_size: 256
  gamma: 0.99
  tau: 0.005
  buffer_size: 1000000
```

---

### TD3

**Twin Delayed DDPG** -- 双延迟深度确定性策略梯度算法，通过双 Q 网络和延迟策略更新改进 DDPG。

```python
from axiomrl.core import TD3
```

```yaml
algo: TD3
env_id: HalfCheetah-v4
seed: 42
total_timesteps: 1000000
output_dir: runs/td3
algo_kwargs:
  learning_rate: 0.001
  batch_size: 256
  gamma: 0.99
  tau: 0.005
  buffer_size: 1000000
  policy_delay: 2
```

---

### TRPO

**Trust Region Policy Optimization** -- 信赖域策略优化算法，使用 KL 散度约束保证每次更新在信赖域内。

```python
from axiomrl.core import TRPO
```

```yaml
algo: TRPO
env_id: HalfCheetah-v4
seed: 42
total_timesteps: 1000000
output_dir: runs/trpo
algo_kwargs:
  learning_rate: 0.001
  n_steps: 2048
  gamma: 0.99
  cg_max_steps: 10
  target_kl: 0.01
```

---

## TrainConfig

`TrainConfig` 是训练配置的核心数据类，定义于 `axiomrl.experiment.config`，并通过稳定核心 API 导出。

```python
from axiomrl.core import TrainConfig
from pathlib import Path

config = TrainConfig(
    algo="PPO",
    env_id="CartPole-v1",
    seed=42,
    total_timesteps=100000,
    output_dir=Path("runs/cartpole-ppo"),
)
```

完整的字段说明请参考 [TrainConfig 完整参考](../configuration/train-config.md)。

---

## 稳定性保证

!!! success "版本兼容承诺"
    `axiomrl.core` 遵循[语义版本控制](https://semver.org/)：

    - **补丁版本**（1.0.x）：修复缺陷，不改变 API。
    - **次要版本**（1.x.0）：可能新增功能，但不会破坏现有 API。
    - **主要版本**（x.0.0）：可能引入不兼容变更。

    在整个 **1.x** 发布周期内，`axiomrl.core` 中的所有公开名称保证可用且行为一致。

!!! warning "注意事项"
    稳定性保证仅覆盖 `axiomrl.core` 模块中的公开名称。算法内部实现细节、私有 API 和实验性模块不在保证范围内。
