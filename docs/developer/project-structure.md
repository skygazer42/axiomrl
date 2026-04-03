---
title: 项目结构详解
---

# 项目结构详解

本文档详细说明 AxiomRL 的源码目录布局、模块职责和构建系统配置。

## 顶层目录结构

```
axiomrl/
├── src/rl_training/        # 主包源码
├── configs/                # 88 个算法配置目录
├── examples/               # 42 个示例脚本
├── zoo/                    # 基准预设和清单
├── tests/                  # 测试套件
├── docs/                   # 文档源文件
├── pyproject.toml          # 包配置与构建元数据
├── Makefile                # 开发命令集
├── CONTRIBUTING.md         # 贡献指南
├── CHANGELOG.md            # 更新日志
└── LICENSE                 # MIT 许可证
```

### 各顶层目录说明

| 目录/文件 | 说明 |
|---|---|
| `src/rl_training/` | 项目主包，包含所有核心源码 |
| `configs/` | 算法默认配置文件，共 88 个目录，每个目录对应一种算法 |
| `examples/` | 示例脚本，共 42 个，展示各种使用场景 |
| `zoo/` | Zoo 基准预设和清单文件，用于标准化基准测试 |
| `tests/` | 测试套件，包含 unit、integration、smoke、slow 四级测试 |
| `docs/` | MkDocs 文档源文件 |
| `pyproject.toml` | 包元数据、依赖声明和工具配置 |
| `Makefile` | 统一的开发命令入口 |

## 主包结构 — `src/rl_training/`

```
src/rl_training/
├── __init__.py
├── algorithms/             # 70+ 算法实现
├── api/                    # 公共 API 封装
├── assets/                 # 打包配置和 Zoo YAML
├── cli.py                  # 主 CLI 入口
├── cli_config.py           # 配置加载逻辑
├── cli_doctor.py           # 诊断工具
├── cli_zoo.py              # Zoo 参数转发
├── contrib/                # 社区扩展
├── core.py                 # 稳定核心 API
├── data/                   # 缓冲区、数据集、采样器
├── envs/                   # 环境工厂、包装器
├── experiment/             # 配置、检查点、日志、注册表
├── experimental.py         # 实验性 API
├── models/                 # 神经网络模型
├── policies/               # 策略抽象
├── runtime/                # 训练器、评估器、收集器
├── version.py              # 版本号定义
└── zoo/                    # Zoo 核心、清单、报告
```

### 模块详解

#### `algorithms/` — 算法实现

包含 70 余种强化学习算法的完整实现，涵盖：

- **在线策略（On-Policy）**：A2C、PPO、TRPO 等
- **离线策略（Off-Policy）**：DQN、SAC、TD3、DiscreteSAC 等
- **离线学习（Offline）**：BC、CQL、IQL 等
- **其他**：模仿学习、多智能体算法等

每种算法通常包含策略定义、损失计算和训练循环等组件。

#### `api/` — 公共 API 封装

提供面向应用工程师的高层 API，封装了常用操作（如训练、评估、加载模型等），简化使用流程。

#### `core.py` — 稳定核心 API

`rl_training.core` 命名空间的入口，提供受语义化版本保护的稳定接口。当前包含的稳定算法：

- A2C, BC, CQL, DQN, DiscreteSAC, IQL, PPO, SAC, TD3, TRPO

#### `experimental.py` — 实验性 API

`rl_training.experimental` 命名空间的入口，包含尚在开发中的新特性。这些 API 可能在次版本（minor）中变更。

#### `data/` — 数据管理

| 组件 | 说明 |
|---|---|
| 缓冲区（Buffers） | 经验回放缓冲区的各种实现 |
| 数据集（Datasets） | 离线数据集的加载与管理 |
| 采样器（Samplers） | 数据采样策略 |

#### `envs/` — 环境管理

- **环境工厂**：统一的环境创建接口
- **包装器（Wrappers）**：观测处理、奖励整形、动作重映射等

#### `experiment/` — 实验管理

| 组件 | 说明 |
|---|---|
| 配置系统 | YAML 配置文件的加载、合并与验证 |
| 检查点 | 训练状态的保存与恢复 |
| 日志 | TensorBoard 日志记录 |
| 注册表 | 算法、环境等组件的注册与发现 |

#### `models/` — 神经网络模型

提供可复用的网络模块：

- **MLP**：多层感知器，用于低维观测空间
- **CNN**：卷积网络，用于图像观测空间
- **LSTM**：循环网络，用于部分可观测环境

#### `policies/` — 策略抽象

定义策略的基类与通用接口，算法通过继承策略基类实现具体行为。

#### `runtime/` — 运行时组件

| 组件 | 说明 |
|---|---|
| 训练器（Trainer） | 统一的训练循环管理 |
| 评估器（Evaluator） | 策略评估与指标收集 |
| 收集器（Collector） | 环境交互数据的收集 |

#### `zoo/` — Zoo 系统

Zoo 子系统负责管理标准化基准预设：

- **Zoo 核心**：预设加载与执行
- **清单（Manifest）**：预设元数据与组织
- **报告（Report）**：基准测试结果的生成与比较

#### CLI 模块

| 模块 | 说明 |
|---|---|
| `cli.py` | 主 CLI 入口点 |
| `cli_config.py` | 配置文件的发现与加载 |
| `cli_doctor.py` | 环境诊断工具，检查依赖和配置状态 |
| `cli_zoo.py` | Zoo 基准测试的命令行参数转发 |

#### 其他模块

| 模块 | 说明 |
|---|---|
| `assets/` | 打包所需的静态资源与 Zoo YAML 模板 |
| `contrib/` | 社区贡献的扩展组件，无稳定性保证 |
| `version.py` | 版本号定义：`__version__ = "1.0.0"` |

## 配置目录 — `configs/`

包含 88 个算法配置目录，每个目录中存放该算法的默认 YAML 配置文件。目录命名与算法名称一一对应，例如：

```
configs/
├── a2c/
├── dqn/
├── ppo/
├── sac/
├── td3/
└── ...（共 88 个）
```

## 示例目录 — `examples/`

包含 42 个示例脚本，覆盖以下场景：

- 基础训练与评估
- 自定义环境接入
- 超参数调优
- 离线学习
- Zoo 基准测试

## Zoo 目录 — `zoo/`

存放基准预设和清单文件，用于标准化的算法评估：

- 预设文件定义了特定环境-算法组合的超参数
- 清单文件管理预设的组织与元数据

## 测试目录 — `tests/`

测试组织遵循源码目录结构，使用 pytest 标记区分测试层级：

| 标记 | 说明 |
|---|---|
| `unit`（默认） | 快速单元测试 |
| `integration` | 跨模块集成测试 |
| `smoke` | 端到端冒烟测试 |
| `slow` | 耗时长的测试，快速 CI 中排除 |

## 构建与打包

### 构建系统

AxiomRL 使用 `setuptools`（>= 77.0.3）作为构建后端，配置集中在 `pyproject.toml` 中：

```toml
[build-system]
requires = ["setuptools>=77.0.3"]
build-backend = "setuptools.build_meta"
```

- **包名**：`axiomrl`
- **版本**：`1.0.0`
- **Python 要求**：`>= 3.10`
- **许可证**：MIT
- **包目录**：`src/`

### 依赖分组

| 分组 | 依赖 |
|---|---|
| **基础** | gymnasium, numpy, PyYAML, tensorboard, torch |
| **Atari** | ale-py |
| **离线** | minari |
| **开发** | build, mypy, pre-commit, pytest, pytest-cov, ruff, tomli, twine, opencv-python, pygame |

### 入口点（Entry Points）

项目注册了以下控制台脚本入口点：

| 命令 | 入口 | 说明 |
|---|---|---|
| `axiomrl` | `cli:main` | 主 CLI，用于训练、评估、恢复等操作 |
| `axiomrl-zoo` | `zoo_cli:main` | Zoo 基准测试专用 CLI |
