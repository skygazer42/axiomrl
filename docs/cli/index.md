---
title: CLI 完整参考
---

# CLI 完整参考

AxiomRL 提供 `axiomrl` 命令行工具，涵盖训练、评估、检查点恢复、基准测试和环境诊断等完整工作流。

```bash
axiomrl <子命令> [参数]
```

---

## axiomrl train

启动训练任务。从 YAML 配置文件读取训练参数，支持通过命令行覆盖关键字段。

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | `str` | _(必填)_ | YAML 配置文件路径 |
| `--output-dir` | `str` | 配置文件值 | 覆盖输出目录 |
| `--execution-backend` | `str` | 配置文件值 | 覆盖执行后端 |
| `--total-timesteps` | `int` | 配置文件值 | 覆盖总训练步数 |
| `--num-envs` | `int` | 配置文件值 | 覆盖并行环境数 |
| `--eval-episodes` | `int` | 配置文件值 | 覆盖评估回合数 |
| `--seeds` | `str` | 配置文件值 | 覆盖基准种子列表（逗号分隔） |

### 示例

```bash
# 基本训练
axiomrl train --config configs/cartpole-ppo.yaml

# 覆盖输出目录和步数
axiomrl train --config configs/cartpole-ppo.yaml \
  --output-dir runs/experiment-v2 \
  --total-timesteps 500000

# 多种子基准训练
axiomrl train --config configs/atari-ppo.yaml \
  --seeds 0,1,2,3,4

# 使用多环境并行加速
axiomrl train --config configs/mujoco-sac.yaml \
  --num-envs 8 \
  --execution-backend local_sync
```

### 输出

训练完成后输出：

```
run_dir=runs/cartpole-ppo/PPO_CartPole-v1_42
checkpoint_path=runs/cartpole-ppo/PPO_CartPole-v1_42/checkpoint.pt
metrics={'eval/mean_reward': 500.0, 'eval/std_reward': 0.0}
```

---

## axiomrl eval

对已保存的检查点进行评估，运行指定数量的回合并输出性能指标。

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | `str` | _(必填)_ | 检查点文件路径 |
| `--num-episodes` | `int` | `None` | 评估回合数（不指定时使用配置值） |

### 示例

```bash
# 评估检查点
axiomrl eval --checkpoint runs/cartpole-ppo/checkpoint.pt

# 指定评估回合数
axiomrl eval --checkpoint runs/cartpole-ppo/checkpoint.pt \
  --num-episodes 100
```

### 输出

```
{'eval/mean_reward': 487.5, 'eval/std_reward': 12.3, 'eval/episodes': 100}
```

---

## axiomrl resume

从检查点恢复训练，支持修改部分训练参数。

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | `str` | _(必填)_ | 检查点文件路径 |
| `--total-timesteps` | `int` | `None` | 新的总训练步数 |
| `--output-dir` | `str` | `None` | 新的输出目录 |
| `--execution-backend` | `str` | `None` | 覆盖执行后端 |
| `--eval-episodes` | `int` | `None` | 覆盖评估回合数 |

### 示例

```bash
# 从检查点恢复训练
axiomrl resume --checkpoint runs/cartpole-ppo/checkpoint.pt

# 恢复并延长训练
axiomrl resume --checkpoint runs/cartpole-ppo/checkpoint.pt \
  --total-timesteps 2000000

# 恢复到新目录
axiomrl resume --checkpoint runs/cartpole-ppo/checkpoint.pt \
  --total-timesteps 2000000 \
  --output-dir runs/cartpole-ppo-resumed
```

### 输出

```
run_dir=runs/cartpole-ppo-resumed/PPO_CartPole-v1_42
checkpoint_path=runs/cartpole-ppo-resumed/PPO_CartPole-v1_42/checkpoint.pt
metrics={'eval/mean_reward': 500.0, 'eval/std_reward': 0.0}
```

---

## axiomrl zoo

Zoo 基准测试工具，管理和分析基准实验结果。支持多种输出格式和灵活的筛选条件。

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--manifest` | `str` | `zoo/atari/benchmark.yaml` | 基准清单文件路径 |
| `--format` | `str` | `table` | 输出格式：`table`、`commands`、`report`、`leaderboard` |
| `--runs-dir` | `str` | `runs` | 运行结果目录 |
| `--report-output` | `str` | `text` | 报告输出格式：`text`、`json`、`csv` |
| `--algo` | `str` | `None` | 按算法名称筛选 |
| `--env-id` | `str` | `None` | 按环境 ID 筛选 |
| `--group-by` | `str` | `algo-env` | 分组方式：`algo-env`、`preset` |
| `--min-seeds` | `int` | `None` | 最少种子数筛选 |
| `--top-k` | `int` | `None` | 仅显示前 K 个结果 |
| `--baseline-preset` | `str` | `None` | 基线预设名称 |
| `--leaderboard-metric` | `str` | `None` | 排行榜排序指标（见下方列表） |
| `--compare-to` | `str` | `None` | 比较基准：`best`、`latest` |
| `--score-view` | `str` | `None` | 分数视图：`return`、`normalized` |
| `--sort-by` | `str` | `None` | 排序字段 |
| `--descending` | `flag` | `False` | 降序排列 |
| `--fail-on-manifest-drift` | `flag` | `False` | 清单漂移时报错 |
| `--fail-on-manifest-drift-severity` | `str` | `None` | 漂移严重级别：`warning`、`error` |
| `--fail-on-manifest-drift-type` | `str` | `None` | 漂移类型筛选：`unknown-preset`、`protocol-mismatch`（可多次指定） |
| `--output` | `str` | `None` | 输出文件路径（不指定时打印到终端） |

??? info "排行榜指标选项"
    `--leaderboard-metric` 支持以下值：

    **基于原始回报：**
    `best-return`、`latest-return`、`gap-return`、`stability-return`、`confidence-return`、`median-return`、`iqr-return`、`delta-vs-baseline-return`、`ratio-vs-baseline-return`

    **基于归一化分数：**
    `best-normalized`、`latest-normalized`、`gap-normalized`、`stability-normalized`、`confidence-normalized`、`median-normalized`、`iqr-normalized`、`delta-vs-baseline-normalized`、`ratio-vs-baseline-normalized`

### 示例

```bash
# 查看基准总表
axiomrl zoo --manifest zoo/atari/benchmark.yaml

# 生成训练命令
axiomrl zoo --manifest zoo/atari/benchmark.yaml --format commands

# 生成报告
axiomrl zoo --manifest zoo/atari/benchmark.yaml \
  --format report --report-output json --output report.json

# 排行榜视图
axiomrl zoo --manifest zoo/atari/benchmark.yaml \
  --format leaderboard \
  --leaderboard-metric best-normalized \
  --descending

# 按算法和环境筛选
axiomrl zoo --manifest zoo/atari/benchmark.yaml \
  --algo PPO --env-id BreakoutNoFrameskip-v4

# 仅显示前 5 名
axiomrl zoo --manifest zoo/atari/benchmark.yaml \
  --format leaderboard --top-k 5

# CI 中检测清单漂移
axiomrl zoo --manifest zoo/atari/benchmark.yaml \
  --fail-on-manifest-drift \
  --fail-on-manifest-drift-severity error
```

---

## axiomrl report

生成基准报告的快捷命令，等价于 `axiomrl zoo --format report`。

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--manifest` | `str` | `zoo/atari/benchmark.yaml` | 基准清单文件路径 |
| `--runs-dir` | `str` | `runs` | 运行结果目录 |
| `--report-output` | `str` | `text` | 报告输出格式：`text`、`json`、`csv` |
| `--algo` | `str` | `None` | 按算法名称筛选 |
| `--env-id` | `str` | `None` | 按环境 ID 筛选 |
| `--group-by` | `str` | `algo-env` | 分组方式：`algo-env`、`preset` |
| `--min-seeds` | `int` | `None` | 最少种子数筛选 |
| `--top-k` | `int` | `None` | 仅显示前 K 个结果 |
| `--baseline-preset` | `str` | `None` | 基线预设名称 |
| `--sort-by` | `str` | `None` | 排序字段 |
| `--descending` | `flag` | `False` | 降序排列 |
| `--fail-on-manifest-drift` | `flag` | `False` | 清单漂移时报错 |
| `--fail-on-manifest-drift-severity` | `str` | `None` | 漂移严重级别：`warning`、`error` |
| `--fail-on-manifest-drift-type` | `str` | `None` | 漂移类型筛选（可多次指定） |
| `--output` | `str` | `None` | 输出文件路径 |

### 示例

```bash
# 生成文本报告
axiomrl report --manifest zoo/atari/benchmark.yaml

# 生成 JSON 格式报告并保存
axiomrl report --manifest zoo/atari/benchmark.yaml \
  --report-output json --output benchmark-report.json

# 筛选特定算法的报告
axiomrl report --manifest zoo/mujoco/benchmark.yaml \
  --algo SAC --report-output csv --output sac-report.csv

# 按预设分组
axiomrl report --manifest zoo/atari/benchmark.yaml \
  --group-by preset --sort-by preset_name
```

---

## axiomrl leaderboard

生成排行榜视图的快捷命令，提供排行榜专用参数。

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--manifest` | `str` | `zoo/atari/benchmark.yaml` | 基准清单文件路径 |
| `--runs-dir` | `str` | `runs` | 运行结果目录 |
| `--report-output` | `str` | `text` | 输出格式：`text`、`json`、`csv` |
| `--algo` | `str` | `None` | 按算法名称筛选 |
| `--env-id` | `str` | `None` | 按环境 ID 筛选 |
| `--group-by` | `str` | `algo-env` | 分组方式：`algo-env`、`preset` |
| `--min-seeds` | `int` | `None` | 最少种子数筛选 |
| `--top-k` | `int` | `None` | 仅显示前 K 个结果 |
| `--baseline-preset` | `str` | `None` | 基线预设名称 |
| `--leaderboard-metric` | `str` | `None` | 排行榜排序指标 |
| `--compare-to` | `str` | `None` | 比较基准：`best`、`latest` |
| `--score-view` | `str` | `None` | 分数视图：`return`、`normalized` |
| `--sort-by` | `str` | `None` | 排序字段 |
| `--descending` | `flag` | `False` | 降序排列 |
| `--fail-on-manifest-drift` | `flag` | `False` | 清单漂移时报错 |
| `--fail-on-manifest-drift-severity` | `str` | `None` | 漂移严重级别 |
| `--fail-on-manifest-drift-type` | `str` | `None` | 漂移类型筛选（可多次指定） |
| `--output` | `str` | `None` | 输出文件路径 |

### 示例

```bash
# 默认排行榜
axiomrl leaderboard --manifest zoo/atari/benchmark.yaml

# 按归一化最佳分数排序（降序）
axiomrl leaderboard --manifest zoo/atari/benchmark.yaml \
  --leaderboard-metric best-normalized \
  --descending

# 与基线比较
axiomrl leaderboard --manifest zoo/atari/benchmark.yaml \
  --leaderboard-metric ratio-vs-baseline-return \
  --compare-to best \
  --baseline-preset dqn-default

# 仅查看归一化视图的前 10 名
axiomrl leaderboard --manifest zoo/atari/benchmark.yaml \
  --score-view normalized \
  --top-k 10 \
  --descending

# 输出 CSV 格式
axiomrl leaderboard --manifest zoo/atari/benchmark.yaml \
  --leaderboard-metric median-normalized \
  --report-output csv --output leaderboard.csv
```

---

## axiomrl config

解析并显示合并后的最终训练配置。适用于调试和验证配置文件。

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | `str` | _(必填)_ | YAML 配置文件路径 |
| `--format` | `str` | `json` | 输出格式：`json`、`yaml` |
| `--output` | `str` | `None` | 输出文件路径（不指定时打印到终端） |
| `--output-dir` | `str` | 配置文件值 | 覆盖输出目录 |
| `--execution-backend` | `str` | 配置文件值 | 覆盖执行后端 |
| `--total-timesteps` | `int` | 配置文件值 | 覆盖总训练步数 |
| `--num-envs` | `int` | 配置文件值 | 覆盖并行环境数 |
| `--eval-episodes` | `int` | 配置文件值 | 覆盖评估回合数 |
| `--seeds` | `str` | 配置文件值 | 覆盖基准种子列表 |

### 示例

```bash
# 以 JSON 格式查看解析后的配置
axiomrl config --config configs/cartpole-ppo.yaml

# 以 YAML 格式查看
axiomrl config --config configs/cartpole-ppo.yaml --format yaml

# 应用 CLI 覆盖后查看最终配置
axiomrl config --config configs/cartpole-ppo.yaml \
  --total-timesteps 200000 \
  --num-envs 4

# 保存解析后的配置到文件
axiomrl config --config configs/my-experiment.yaml \
  --format yaml --output resolved.yaml
```

!!! tip "使用场景"
    在启动训练之前，使用 `axiomrl config` 验证预设链接、CLI 覆盖和字段合并是否符合预期。这可以避免因配置错误导致的训练失败。

---

## axiomrl doctor

诊断当前运行环境，输出 AxiomRL 和关键依赖的版本、CUDA 可用性等信息。

### 说明

`axiomrl doctor` 不接受任何参数。它自动检测并输出以下信息：

- AxiomRL 版本
- Python 可执行文件路径和版本
- 操作系统平台
- PyTorch 版本
- Gymnasium 版本
- NumPy 版本
- OpenCV 版本
- Pygame 版本
- Minari 版本
- CUDA 可用性、设备数量和名称
- PyTorch CUDA 版本

### 输出示例

```bash
$ axiomrl doctor
axiomrl_version=1.0.0
python_executable=/home/user/.venv/bin/python
python_version=3.11.7
platform=Linux-6.8.0-90-generic-x86_64-with-glibc2.39
torch_version=2.2.1
gymnasium_version=0.29.1
numpy_version=1.26.4
opencv_python_version=4.9.0.80
pygame_version=2.5.2
minari_version=0.4.3
cuda_available=True
cuda_device_count=1
cuda_device_name=NVIDIA GeForce RTX 4090
torch_cuda_version=12.1
```

!!! tip "故障排查"
    遇到环境问题时，首先运行 `axiomrl doctor` 收集环境信息。在提交问题报告时附上此输出可以加速问题定位。

---

## axiomrl --version

显示 AxiomRL 的版本号。

```bash
$ axiomrl --version
axiomrl 1.0.0
```

也可以使用短参数形式：

```bash
$ axiomrl -V
axiomrl 1.0.0
```

---

## 子命令速查

| 子命令 | 说明 | 示例 |
|--------|------|------|
| `train` | 启动训练 | `axiomrl train --config cfg.yaml` |
| `eval` | 评估检查点 | `axiomrl eval --checkpoint ckpt.pt` |
| `resume` | 恢复训练 | `axiomrl resume --checkpoint ckpt.pt` |
| `zoo` | 基准管理 | `axiomrl zoo --manifest manifest.yaml` |
| `report` | 生成报告 | `axiomrl report --manifest manifest.yaml` |
| `leaderboard` | 排行榜 | `axiomrl leaderboard --manifest manifest.yaml` |
| `config` | 配置检查 | `axiomrl config --config cfg.yaml` |
| `doctor` | 环境诊断 | `axiomrl doctor` |
| `--version` | 版本信息 | `axiomrl --version` |
