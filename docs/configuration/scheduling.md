---
title: 调度器配置
---

# 调度器配置

AxiomRL 支持在训练过程中动态调整超参数。调度器通过 `algo_kwargs` 中的特定字段配置，所有调度器均支持以下四种类型：

| 调度类型 | 说明 |
|----------|------|
| `linear` | 线性插值，从 `start` 到 `end` |
| `cosine` | 余弦退火，从 `start` 到 `end` |
| `step` | 阶梯衰减，在 `milestones` 处乘以 `gamma` |
| `constant` | 保持常数值不变 |

---

## lr_schedule - 学习率调度

控制训练过程中学习率的变化策略。适用于所有算法。

### 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | `str` | 是 | 调度类型：`linear`、`cosine`、`step`、`constant` |
| `start` | `float` | `linear`/`cosine` | 初始学习率 |
| `end` | `float` | `linear`/`cosine` | 最终学习率 |
| `milestones` | `list[float]` | `step` | 衰减里程碑（以训练进度比例表示，0.0-1.0） |
| `gamma` | `float` | `step` | 每个里程碑处的衰减因子 |

### 示例

=== "线性衰减"

    ```yaml
    algo_kwargs:
      lr_schedule:
        type: linear
        start: 0.0003
        end: 0.00001
    ```

=== "余弦退火"

    ```yaml
    algo_kwargs:
      lr_schedule:
        type: cosine
        start: 0.0003
        end: 0.0
    ```

=== "阶梯衰减"

    ```yaml
    algo_kwargs:
      lr_schedule:
        type: step
        start: 0.001
        milestones: [0.3, 0.6, 0.9]
        gamma: 0.1
    ```

=== "常数"

    ```yaml
    algo_kwargs:
      lr_schedule:
        type: constant
    ```

!!! tip "默认行为"
    如果不指定 `lr_schedule`，默认使用 `constant` 类型，即学习率保持不变。

---

## epsilon_schedule - epsilon 贪心调度

控制 DQN 系列算法中 epsilon-greedy 探索策略的 epsilon 值变化。

**适用算法：** DQN、DiscreteSAC

### 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | `str` | 是 | 调度类型：`linear`、`cosine`、`step`、`constant` |
| `start` | `float` | `linear`/`cosine` | 初始 epsilon 值 |
| `end` | `float` | `linear`/`cosine` | 最终 epsilon 值 |
| `milestones` | `list[float]` | `step` | 衰减里程碑 |
| `gamma` | `float` | `step` | 衰减因子 |

### 示例

=== "线性衰减"

    ```yaml
    algo_kwargs:
      epsilon_schedule:
        type: linear
        start: 1.0
        end: 0.05
    ```

=== "余弦衰减"

    ```yaml
    algo_kwargs:
      epsilon_schedule:
        type: cosine
        start: 1.0
        end: 0.01
    ```

!!! info "典型配置"
    DQN 中常见的做法是将 epsilon 从 1.0 线性衰减到 0.05 或 0.01，使训练前期充分探索，后期更多利用已学策略。

---

## entropy_coef_schedule - 熵系数调度

控制策略熵正则化系数的变化，影响探索-利用平衡。

**适用算法：** PPO、A2C

### 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | `str` | 是 | 调度类型：`linear`、`cosine`、`step`、`constant` |
| `start` | `float` | `linear`/`cosine` | 初始熵系数 |
| `end` | `float` | `linear`/`cosine` | 最终熵系数 |
| `milestones` | `list[float]` | `step` | 衰减里程碑 |
| `gamma` | `float` | `step` | 衰减因子 |

### 示例

=== "线性衰减"

    ```yaml
    algo_kwargs:
      entropy_coef_schedule:
        type: linear
        start: 0.01
        end: 0.001
    ```

=== "阶梯衰减"

    ```yaml
    algo_kwargs:
      entropy_coef_schedule:
        type: step
        start: 0.01
        milestones: [0.5, 0.8]
        gamma: 0.5
    ```

!!! tip "使用建议"
    较高的熵系数鼓励更多探索，适合训练早期。随着训练进行逐步降低熵系数，有助于策略收敛。

---

## clip_range_schedule - 裁剪范围调度

控制 PPO 算法中策略更新的裁剪范围。

**适用算法：** PPO

### 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | `str` | 是 | 调度类型：`linear`、`cosine`、`step`、`constant` |
| `start` | `float` | `linear`/`cosine` | 初始裁剪范围 |
| `end` | `float` | `linear`/`cosine` | 最终裁剪范围 |
| `milestones` | `list[float]` | `step` | 衰减里程碑 |
| `gamma` | `float` | `step` | 衰减因子 |

### 示例

=== "线性衰减"

    ```yaml
    algo_kwargs:
      clip_range_schedule:
        type: linear
        start: 0.2
        end: 0.05
    ```

=== "常数"

    ```yaml
    algo_kwargs:
      clip_range_schedule:
        type: constant
    ```

!!! info "原理"
    裁剪范围限制了新旧策略比率的偏差。逐步缩小裁剪范围可以在训练后期实现更精细的策略更新。

---

## temperature_schedule - 温度调度

控制 SAC 算法中熵温度参数的变化，影响探索强度。

**适用算法：** SAC、DiscreteSAC

### 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | `str` | 是 | 调度类型：`linear`、`cosine`、`step`、`constant` |
| `start` | `float` | `linear`/`cosine` | 初始温度 |
| `end` | `float` | `linear`/`cosine` | 最终温度 |
| `milestones` | `list[float]` | `step` | 衰减里程碑 |
| `gamma` | `float` | `step` | 衰减因子 |

### 示例

=== "余弦退火"

    ```yaml
    algo_kwargs:
      temperature_schedule:
        type: cosine
        start: 0.2
        end: 0.05
    ```

=== "线性衰减"

    ```yaml
    algo_kwargs:
      temperature_schedule:
        type: linear
        start: 0.2
        end: 0.01
    ```

!!! warning "自动温度调节"
    SAC 默认使用自动熵温度调节。如果配置了 `temperature_schedule`，将覆盖自动调节机制。请确保理解其对训练稳定性的影响。

---

## root_noise_schedule - 探索噪声调度

控制 TD3 算法中目标策略平滑噪声的强度变化。

**适用算法：** TD3

### 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | `str` | 是 | 调度类型：`linear`、`cosine`、`step`、`constant` |
| `start` | `float` | `linear`/`cosine` | 初始噪声标准差 |
| `end` | `float` | `linear`/`cosine` | 最终噪声标准差 |
| `milestones` | `list[float]` | `step` | 衰减里程碑 |
| `gamma` | `float` | `step` | 衰减因子 |

### 示例

=== "线性衰减"

    ```yaml
    algo_kwargs:
      root_noise_schedule:
        type: linear
        start: 0.2
        end: 0.05
    ```

=== "阶梯衰减"

    ```yaml
    algo_kwargs:
      root_noise_schedule:
        type: step
        start: 0.2
        milestones: [0.5]
        gamma: 0.5
    ```

!!! tip "调参建议"
    适当降低探索噪声有助于训练后期的策略精细化。但噪声过低可能导致训练陷入局部最优。

---

## simulation_schedule - 仿真步调度

控制基于模型的算法中每步环境交互对应的模型仿真步数。

**适用算法：** 基于模型的算法（model-based）

### 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | `str` | 是 | 调度类型：`linear`、`cosine`、`step`、`constant` |
| `start` | `float` | `linear`/`cosine` | 初始仿真步数 |
| `end` | `float` | `linear`/`cosine` | 最终仿真步数 |
| `milestones` | `list[float]` | `step` | 增长里程碑 |
| `gamma` | `float` | `step` | 增长因子 |

### 示例

=== "线性增长"

    ```yaml
    algo_kwargs:
      simulation_schedule:
        type: linear
        start: 1
        end: 10
    ```

=== "阶梯增长"

    ```yaml
    algo_kwargs:
      simulation_schedule:
        type: step
        start: 1
        milestones: [0.25, 0.5, 0.75]
        gamma: 2.0
    ```

!!! info "设计思路"
    随着学习到的世界模型精度提高，逐步增加仿真步数可以加速训练收敛，同时避免早期因模型不准确导致的偏差累积。

---

## 组合使用

多个调度器可以在同一个配置中组合使用：

```yaml title="configs/ppo-full-schedule.yaml"
algo: PPO
env_id: HalfCheetah-v4
seed: 42
total_timesteps: 1000000
output_dir: runs/ppo-scheduled

algo_kwargs:
  learning_rate: 0.0003
  batch_size: 64
  gamma: 0.99
  n_steps: 2048
  n_epochs: 10

  lr_schedule:
    type: cosine
    start: 0.0003
    end: 0.00001

  entropy_coef_schedule:
    type: linear
    start: 0.01
    end: 0.001

  clip_range_schedule:
    type: linear
    start: 0.2
    end: 0.05
```
