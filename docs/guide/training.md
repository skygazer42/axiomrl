---
title: 训练详解
icon: material/play-circle
---

# 训练详解

本章节详细介绍 AxiomRL 的训练流程，包括配置文件编写、CLI 命令、Python API 以及各种高级训练选项。

## YAML 配置文件结构

AxiomRL 使用 YAML 文件定义训练的全部参数。一个完整的配置文件结构如下：

```yaml title="config.yaml" linenums="1"
# 算法选择
algo: PPO

# 环境 ID（Gymnasium 格式）
env_id: CartPole-v1

# 随机种子
seed: 42

# 总训练步数
total_timesteps: 1_000_000

# 输出目录
output_dir: runs/

# 执行后端
execution_backend: local_sync

# 设备选择："auto", "cpu", "cuda", "cuda:0" 等
device: auto

# 并行环境数量
num_envs: 4

# 评估回合数
eval_episodes: 10

# 日志记录间隔（步数）
log_interval: 1

# 检查点保存间隔（步数）
checkpoint_interval: 1

# 标签（用于实验管理）
tags:
  - baseline
  - cartpole

# 基准配置
benchmark: {}

# 算法特定参数
algo_kwargs:
  learning_rate: 3.0e-4
  batch_size: 64
  gamma: 0.99
  n_steps: 2048

# 环境特定参数
env_kwargs: {}
```

!!! info "TrainConfig 完整字段"

    | 字段 | 类型 | 默认值 | 说明 |
    |------|------|--------|------|
    | `algo` | `str` | **必填** | 算法名称 |
    | `env_id` | `str` | **必填** | Gymnasium 环境 ID |
    | `seed` | `int` | **必填** | 随机种子 |
    | `total_timesteps` | `int` | **必填** | 总训练步数 |
    | `output_dir` | `str` | `"runs/"` | 输出根目录 |
    | `execution_backend` | `str` | `"local_sync"` | 执行后端 |
    | `device` | `str` | `"auto"` | 计算设备 |
    | `num_envs` | `int` | `1` | 并行环境数 |
    | `eval_episodes` | `int` | `5` | 评估回合数 |
    | `log_interval` | `int` | `1` | 日志间隔 |
    | `checkpoint_interval` | `int` | `1` | 检查点间隔 |
    | `tags` | `tuple` | `()` | 实验标签 |
    | `benchmark` | `dict` | `{}` | 基准配置 |
    | `algo_kwargs` | `dict` | `{}` | 算法参数 |
    | `env_kwargs` | `dict` | `{}` | 环境参数 |

## 启动训练

=== "CLI 命令"

    使用 `axiomrl train` 命令启动训练：

    ```bash
    # 基本用法：指定配置文件
    axiomrl train --config config.yaml

    # 覆盖输出目录
    axiomrl train --config config.yaml --output-dir my_runs/

    # 指定执行后端和总步数
    axiomrl train --config config.yaml \
        --execution-backend local_sync \
        --total-timesteps 500000

    # 指定并行环境数和评估回合数
    axiomrl train --config config.yaml \
        --num-envs 8 \
        --eval-episodes 20
    ```

    !!! tip "CLI 参数优先级"

        CLI 参数会覆盖 YAML 配置文件中的同名字段。这使你可以使用同一个配置文件，仅通过 CLI 调整部分参数。

=== "Python API"

    使用 Python API 进行更灵活的训练控制：

    ```python title="train.py" linenums="1"
    from axiomrl import TrainConfig, train

    # 方式一：直接构造配置
    config = TrainConfig(
        algo="PPO",
        env_id="CartPole-v1",
        seed=42,
        total_timesteps=1_000_000,
        output_dir="runs/",
        device="auto",
        num_envs=4,
        eval_episodes=10,
        algo_kwargs={
            "learning_rate": 3e-4,
            "batch_size": 64,
            "gamma": 0.99,
        },
    )

    # 启动训练
    train(config)
    ```

    ```python title="从 YAML 加载配置" linenums="1"
    from axiomrl import TrainConfig, train

    # 方式二：从 YAML 文件加载
    config = TrainConfig.from_yaml("config.yaml")

    # 可以在加载后修改参数
    config.total_timesteps = 2_000_000
    config.num_envs = 8

    train(config)
    ```

## 多 Seed 扫描

为了获得统计上可靠的实验结果，AxiomRL 支持多 seed 扫描功能。

=== "CLI 多 Seed"

    ```bash
    # 使用 --seeds 参数指定多个种子
    axiomrl train --config config.yaml --seeds 1 2 3

    # 结合其他参数
    axiomrl train --config config.yaml \
        --seeds 1 2 3 4 5 \
        --output-dir benchmark_runs/ \
        --num-envs 4
    ```

    !!! note "运行目录命名"

        每个 seed 会生成独立的运行目录，命名格式为：

        ```
        <algo>__<env_id>__seed<seed>__<timestamp>
        ```

        例如：`PPO__CartPole-v1__seed1__20260401_120000`

=== "YAML 多 Seed"

    在 YAML 配置中通过 `benchmark.seeds` 指定：

    ```yaml title="benchmark_config.yaml"
    algo: PPO
    env_id: CartPole-v1
    total_timesteps: 1_000_000
    output_dir: benchmark_runs/

    benchmark:
      seeds:
        - 1
        - 2
        - 3
        - 4
        - 5
    ```

## TensorBoard 监控

训练过程中，AxiomRL 自动将日志写入运行目录下的 `tensorboard/` 文件夹。

```bash
# 启动 TensorBoard
tensorboard --logdir runs/

# 指定端口
tensorboard --logdir runs/ --port 6007

# 监控特定实验
tensorboard --logdir runs/PPO__CartPole-v1__seed42__20260401_120000/tensorboard/
```

!!! tip "TensorBoard 常用面板"

    - **Scalars**：训练奖励、损失函数、学习率等标量指标
    - **Histograms**：网络参数分布变化
    - **Images**：像素观测场景下的环境帧

## algo_kwargs 常用参数

`algo_kwargs` 用于传递算法特定的超参数。以下是各算法常用的参数：

### 通用参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `learning_rate` | `float` | 学习率 |
| `batch_size` | `int` | 批量大小 |
| `gamma` | `float` | 折扣因子 |
| `tau` | `float` | 软更新系数 |
| `buffer_size` | `int` | 回放缓冲区大小 |
| `learning_starts` | `int` | 开始学习前的步数 |

### 策略梯度算法（PPO / TRPO / A2C）

```yaml
algo_kwargs:
  learning_rate: 3.0e-4
  n_steps: 2048          # 每次更新的步数
  batch_size: 64
  n_epochs: 10           # PPO 每次更新的轮数
  gamma: 0.99
  gae_lambda: 0.95       # GAE lambda 参数
  clip_range: 0.2        # PPO 裁剪范围
  ent_coef: 0.01         # 熵正则化系数
  vf_coef: 0.5           # 价值函数损失系数
  max_grad_norm: 0.5     # 梯度裁剪
```

### 离线策略算法（SAC / TD3 / DQN）

```yaml
algo_kwargs:
  learning_rate: 3.0e-4
  batch_size: 256
  gamma: 0.99
  tau: 0.005             # 软更新系数
  buffer_size: 1_000_000
  learning_starts: 10000
  train_freq: 1          # 训练频率
  gradient_steps: 1      # 每次更新的梯度步数
```

## 超参数调度

AxiomRL 支持在训练过程中动态调整超参数。通过 `algo_kwargs` 中的调度配置实现：

```yaml title="使用学习率调度"
algo_kwargs:
  lr_schedule:
    type: cosine       # 支持: linear, cosine, step, constant
    initial: 3.0e-4
    final: 1.0e-5

  epsilon_schedule:    # DQN epsilon-greedy 调度
    type: linear
    initial: 1.0
    final: 0.05
    duration: 100000   # 调度持续步数

  entropy_coef_schedule:
    type: linear
    initial: 0.01
    final: 0.001

  clip_range_schedule:   # PPO clip range 调度
    type: constant
    value: 0.2

  temperature_schedule:  # SAC 温度调度
    type: cosine
    initial: 0.2
    final: 0.05
```

!!! info "可用调度类型"

    | 类型 | 说明 |
    |------|------|
    | `linear` | 线性衰减/增长 |
    | `cosine` | 余弦退火 |
    | `step` | 阶梯式调整 |
    | `constant` | 保持不变 |

    支持的调度参数包括：`lr_schedule`、`epsilon_schedule`、`entropy_coef_schedule`、`clip_range_schedule`、`temperature_schedule`、`root_noise_schedule`、`simulation_schedule`。

## env_kwargs 用法

`env_kwargs` 用于配置 Gymnasium 环境的额外参数：

```yaml title="环境参数示例"
env_kwargs:
  # 视频录制
  capture_video: true
  video_folder: videos/
  video_episode_trigger: 100  # 每 100 回合录制一次

  # 环境特定参数（取决于具体环境）
  max_episode_steps: 1000
  render_mode: rgb_array
```

## 执行后端选项

`execution_backend` 控制训练任务的执行方式：

| 后端 | 说明 | 适用场景 |
|------|------|----------|
| `local_sync` | 本地同步执行（默认） | 单机调试和小规模实验 |

```yaml
# 默认本地同步执行
execution_backend: local_sync
```

!!! warning "注意"

    当前稳定版本主要支持 `local_sync` 后端。其他执行后端可能在未来版本中提供。

## 完整训练示例

以下是一个完整的 PPO 训练 MuJoCo 环境的示例：

```yaml title="ppo_halfcheetah.yaml" linenums="1"
algo: PPO
env_id: HalfCheetah-v4
seed: 42
total_timesteps: 1_000_000
output_dir: runs/
device: auto
num_envs: 8
eval_episodes: 10
log_interval: 1
checkpoint_interval: 10

tags:
  - mujoco
  - ppo-baseline

algo_kwargs:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5
  lr_schedule:
    type: linear
    initial: 3.0e-4
    final: 0.0

env_kwargs: {}
```

```bash
# 启动训练
axiomrl train --config ppo_halfcheetah.yaml

# 多 seed 训练
axiomrl train --config ppo_halfcheetah.yaml --seeds 1 2 3 4 5
```
