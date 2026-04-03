---
title: TrainConfig 完整参考
---

# TrainConfig 完整参考

`TrainConfig` 是 AxiomRL 训练流程的核心配置数据类，定义于 `rl_training.experiment.config`。所有 YAML 配置文件最终解析为该类的实例。

```python
from rl_training.experiment.config import TrainConfig
# 或者通过稳定核心 API 导入
from rl_training.core import TrainConfig
```

---

## 字段总览

### 必填字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `algo` | `str` | 算法名称（如 `PPO`、`SAC`、`DQN` 等） |
| `env_id` | `str` | Gymnasium 环境 ID（如 `CartPole-v1`、`HalfCheetah-v4`） |
| `seed` | `int` | 随机种子，用于实验可复现性 |
| `total_timesteps` | `int` | 总训练步数 |
| `output_dir` | `Path` | 输出目录路径，用于存放检查点、日志等产物 |

### 可选字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `execution_backend` | `str` | `"local_sync"` | 执行后端，控制训练的运行方式 |
| `device` | `str` | `"auto"` | 计算设备：`auto`（自动检测）、`cpu`、`cuda` |
| `num_envs` | `int` | `1` | 并行环境数量 |
| `eval_episodes` | `int` | `5` | 每次评估运行的回合数 |
| `log_interval` | `int` | `1` | 日志记录间隔（以训练迭代为单位） |
| `checkpoint_interval` | `int` | `1` | 检查点保存间隔（以训练迭代为单位） |
| `tags` | `tuple[str, ...]` | `()` | 运行标签，用于标记和筛选实验 |
| `benchmark` | `dict` | `{}` | 基准测试配置（详见[基准配置](#benchmark-config)） |
| `algo_kwargs` | `dict` | `{}` | 算法超参数（详见[算法超参数](#algo-kwargs)） |
| `env_kwargs` | `dict` | `{}` | 环境参数（详见[环境参数](#env-kwargs)） |

---

## 配置形式

### 形式一：直接配置

所有字段直接写在 YAML 文件顶层：

```yaml title="configs/cartpole-ppo.yaml"
algo: PPO
env_id: CartPole-v1
seed: 42
total_timesteps: 100000
output_dir: runs/cartpole-ppo
device: auto
num_envs: 8
eval_episodes: 10
log_interval: 1
checkpoint_interval: 5
tags:
  - baseline
  - cartpole

algo_kwargs:
  learning_rate: 0.0003
  batch_size: 64
  gamma: 0.99
  n_steps: 2048
  n_epochs: 10
  clip_range: 0.2
  ent_coef: 0.01

env_kwargs:
  max_episode_steps: 500
```

### 形式二：预设链接配置

通过 `config` 字段引用预设文件，并覆盖部分字段：

```yaml title="configs/my-experiment.yaml"
config: presets/atari-ppo.yaml
seed: 123
total_timesteps: 500000
output_dir: runs/atari-custom
tags:
  - experiment-v2
algo_kwargs:
  learning_rate: 0.00025
```

!!! info "预设合并规则"
    覆盖字段会与预设中的值合并。对于 `algo_kwargs`、`env_kwargs`、`benchmark` 等字典类型字段，覆盖值会 **深度合并** 到预设值中。

---

## YAML 完整配置示例

```yaml title="configs/halfcheetah-sac.yaml"
algo: SAC
env_id: HalfCheetah-v4
seed: 0
total_timesteps: 1000000
output_dir: runs/halfcheetah-sac
execution_backend: local_sync
device: cuda
num_envs: 1
eval_episodes: 10
log_interval: 1
checkpoint_interval: 10
tags:
  - mujoco
  - sac-baseline

benchmark:
  seeds: [0, 1, 2, 3, 4]
  best_metric: eval/mean_reward
  best_metric_mode: max
  score_normalization: min-max
  suite: mujoco
  preset_name: sac-default
  protocol_name: standard-1M

algo_kwargs:
  learning_rate: 0.0003
  batch_size: 256
  gamma: 0.99
  tau: 0.005
  buffer_size: 1000000
  learning_starts: 10000
  train_freq: 1
  gradient_steps: 1
  lr_schedule:
    type: constant

env_kwargs:
  max_episode_steps: 1000
```

---

## algo_kwargs 算法超参数 { #algo-kwargs }

`algo_kwargs` 字典传递算法特定的超参数。常用参数如下：

| 参数 | 类型 | 说明 | 适用算法 |
|------|------|------|----------|
| `learning_rate` | `float` | 学习率 | 所有算法 |
| `batch_size` | `int` | 批量大小 | 所有算法 |
| `gamma` | `float` | 折扣因子 | 所有算法 |
| `tau` | `float` | 软更新系数 | SAC, TD3, DQN, DiscreteSAC |
| `buffer_size` | `int` | 回放缓冲区大小 | 离策略算法 |
| `learning_starts` | `int` | 开始训练前的随机采样步数 | 离策略算法 |
| `train_freq` | `int` | 每 N 步训练一次 | 离策略算法 |
| `gradient_steps` | `int` | 每次训练的梯度更新次数 | 离策略算法 |
| `n_steps` | `int` | 每次更新收集的步数 | PPO, A2C, TRPO |
| `n_epochs` | `int` | 每次更新的训练轮数 | PPO |
| `clip_range` | `float` | PPO 裁剪范围 | PPO |
| `ent_coef` | `float` | 熵正则化系数 | PPO, A2C |
| `target_update_interval` | `int` | 目标网络更新间隔 | DQN |
| `exploration_fraction` | `float` | epsilon 衰减阶段占比 | DQN |

!!! tip "调度器参数"
    `algo_kwargs` 还支持多种调度器配置（如 `lr_schedule`、`epsilon_schedule` 等），详见[调度器配置](scheduling.md)页面。

### 示例

=== "PPO"

    ```yaml
    algo_kwargs:
      learning_rate: 0.0003
      batch_size: 64
      gamma: 0.99
      n_steps: 2048
      n_epochs: 10
      clip_range: 0.2
      ent_coef: 0.01
      vf_coef: 0.5
      max_grad_norm: 0.5
    ```

=== "SAC"

    ```yaml
    algo_kwargs:
      learning_rate: 0.0003
      batch_size: 256
      gamma: 0.99
      tau: 0.005
      buffer_size: 1000000
      learning_starts: 10000
      train_freq: 1
      gradient_steps: 1
    ```

=== "DQN"

    ```yaml
    algo_kwargs:
      learning_rate: 0.0001
      batch_size: 32
      gamma: 0.99
      tau: 1.0
      buffer_size: 100000
      learning_starts: 1000
      target_update_interval: 500
      exploration_fraction: 0.1
      exploration_final_eps: 0.05
    ```

---

## env_kwargs 环境参数 { #env-kwargs }

`env_kwargs` 字典传递给 Gymnasium 环境构造函数。常用参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| `max_episode_steps` | `int` | 每个回合的最大步数 |
| `render_mode` | `str` | 渲染模式（`human`、`rgb_array` 等） |
| `width` | `int` | 渲染宽度（像素观测时） |
| `height` | `int` | 渲染高度（像素观测时） |

```yaml
env_kwargs:
  max_episode_steps: 1000
  render_mode: rgb_array
```

!!! warning "环境兼容性"
    不同环境支持的参数各不相同。请参考目标环境的 Gymnasium 文档确认可用参数。

---

## benchmark 配置 { #benchmark-config }

`benchmark` 字典用于基准测试工作流，控制多种子扫描和结果聚合。

| 键 | 类型 | 说明 |
|----|------|------|
| `seeds` | `list[int]` | 种子列表，用于多种子基准扫描 |
| `best_metric` | `str` | 用于选择最佳检查点的指标名称 |
| `best_metric_mode` | `str` | 指标优化方向：`max` 或 `min` |
| `score_normalization` | `str` | 分数归一化方式（如 `min-max`） |
| `suite` | `str` | 基准测试套件名称（如 `atari`、`mujoco`） |
| `preset_name` | `str` | 预设名称，用于标识配置组合 |
| `protocol_name` | `str` | 协议名称，用于标识评估标准 |

```yaml
benchmark:
  seeds: [0, 1, 2, 3, 4]
  best_metric: eval/mean_reward
  best_metric_mode: max
  score_normalization: min-max
  suite: atari
  preset_name: ppo-default
  protocol_name: standard-10M
```

---

## CLI 覆盖选项

通过命令行参数可以覆盖配置文件中的字段，无需修改 YAML 文件：

| CLI 参数 | 覆盖字段 | 类型 |
|----------|----------|------|
| `--output-dir` | `output_dir` | `str` |
| `--execution-backend` | `execution_backend` | `str` |
| `--total-timesteps` | `total_timesteps` | `int` |
| `--num-envs` | `num_envs` | `int` |
| `--eval-episodes` | `eval_episodes` | `int` |
| `--seeds` | `benchmark.seeds` | `str`（逗号分隔） |

### 示例

```bash
# 覆盖输出目录和总步数
axiomrl train --config configs/ppo.yaml \
  --output-dir runs/experiment-v2 \
  --total-timesteps 500000

# 覆盖并行环境数和种子
axiomrl train --config configs/ppo.yaml \
  --num-envs 16 \
  --seeds 0,1,2,3,4
```

!!! info "优先级"
    CLI 参数的优先级高于配置文件中的值。对于预设链接配置，优先级为：**CLI 参数 > 覆盖文件 > 预设文件**。

---

## 配置检查

使用 `axiomrl config` 命令可以查看解析后的最终配置，验证合并和覆盖是否正确：

```bash
# 以 JSON 格式查看解析后的配置
axiomrl config --config configs/my-experiment.yaml

# 以 YAML 格式输出
axiomrl config --config configs/my-experiment.yaml --format yaml

# 应用 CLI 覆盖后查看
axiomrl config --config configs/my-experiment.yaml \
  --total-timesteps 200000 --num-envs 4

# 输出到文件
axiomrl config --config configs/my-experiment.yaml \
  --format yaml --output resolved-config.yaml
```

!!! tip "调试建议"
    在启动训练前，建议先使用 `axiomrl config` 确认配置解析结果，避免因配置错误浪费计算资源。
