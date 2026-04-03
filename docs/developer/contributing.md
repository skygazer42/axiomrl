---
title: 贡献指南
---

# 贡献指南

感谢你对 AxiomRL 的关注！本文档详细说明了参与贡献的完整流程，包括环境搭建、开发规范、测试要求和 PR 流程。

## 开发环境搭建

### 前提条件

- Python >= 3.10
- Git
- （可选）CUDA 工具包（如需 GPU 测试）

### 安装步骤

```bash
# 克隆仓库
git clone <repo-url> && cd RL

# 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate

# 以可编辑模式安装，包含全部开发依赖
pip install -e ".[dev]"

# 安装预提交钩子
pre-commit install
```

!!! note "开发依赖一览"

    `.[dev]` 额外依赖组包含以下工具：

    | 工具 | 用途 |
    |---|---|
    | `ruff` | 代码风格检查与格式化 |
    | `mypy` | 静态类型检查 |
    | `pytest` / `pytest-cov` | 测试框架与覆盖率 |
    | `pre-commit` | Git 预提交钩子 |
    | `build` / `twine` | 包构建与发布 |
    | `tomli` | TOML 配置解析 |
    | `opencv-python` / `pygame` | 环境渲染依赖 |

## 日常开发工作流

AxiomRL 通过 `Makefile` 提供统一的开发命令。推荐的日常工作流如下：

### 1. 代码检查

```bash
# 运行 Ruff 代码风格检查
make lint

# 运行 Mypy 类型检查
make typecheck
```

### 2. 快速测试

```bash
# 运行快速单元测试（排除 slow 标记的用例）
make test-fast

# 运行全量测试
make test
```

### 3. 针对性测试

```bash
# 仅运行集成测试
make test-integration

# 仅运行冒烟测试
make test-smoke
```

### 4. 完整验证

```bash
# 提交前必须通过的完整验证流水线
make verify
```

!!! warning "`make verify` 包含的检查"

    `make verify` 会按顺序执行以下步骤，任一步骤失败即中止：

    1. **lint** — Ruff 代码风格检查
    2. **typecheck** — Mypy 静态类型检查
    3. **test-fast** — 快速单元测试
    4. **test-integration** — 集成测试
    5. **test-smoke** — 冒烟测试
    6. **build** — 构建分发包

## 测试层级

AxiomRL 的测试分为四个层级，通过 pytest 标记（marker）区分：

| 标记 | 说明 | 运行方式 |
|---|---|---|
| `unit`（默认） | 单元测试，验证单个模块的行为。无标记的测试默认归入此类。 | `make test-fast` |
| `integration` | 集成测试，验证多个模块间的协作。 | `make test-integration` |
| `smoke` | 冒烟测试，对代表性场景做端到端运行验证。 | `make test-smoke` |
| `slow` | 耗时较长的测试，快速 CI 中会被排除。 | `make test`（完整运行时包含） |

### 测试编写规范

- **每个变更都必须附带测试**。修复 bug 时应先编写复现测试。
- 单元测试应保持独立，不依赖外部资源（网络、GPU 等）。
- 集成测试用 `@pytest.mark.integration` 标记。
- 冒烟测试用 `@pytest.mark.smoke` 标记。
- 耗时超过数秒的测试用 `@pytest.mark.slow` 标记。

```python
import pytest

# 单元测试（默认，无需额外标记）
def test_replay_buffer_push():
    ...

# 集成测试
@pytest.mark.integration
def test_dqn_training_loop():
    ...

# 冒烟测试
@pytest.mark.smoke
def test_ppo_cartpole_smoke():
    ...

# 慢速测试
@pytest.mark.slow
def test_sac_full_training():
    ...
```

## 代码规范

### Ruff 配置

AxiomRL 使用 Ruff 进行代码风格检查和格式化，主要配置如下：

| 配置项 | 值 |
|---|---|
| 目标 Python 版本 | `py310` |
| 最大行宽 | `120` |

```bash
# 检查代码风格
make lint

# 自动格式化（如已配置）
ruff format src/
```

### Mypy 配置

使用 Mypy 进行静态类型检查，确保类型标注的正确性：

```bash
make typecheck
```

!!! tip "类型标注建议"

    - 所有公共 API 函数必须有完整的类型标注。
    - 内部函数建议添加类型标注以提高可维护性。
    - 使用 `from __future__ import annotations` 以启用延迟求值。

## 变更期望

每个 PR 都应满足以下要求：

- [x] 所有变更附带对应的测试用例
- [x] 更新 `CHANGELOG.md`（在 `[Unreleased]` 部分添加条目）
- [x] 涉及稳定 API（`rl_training.core`）的变更需同步更新文档
- [x] `make verify` 全部通过
- [x] 预提交钩子检查通过

## 功能贡献注意事项

### 新增 CLI 命令

- 必须在 `tests/` 中添加对应的命令行测试
- 确保 `--help` 输出正确
- 在文档中补充用法说明

### 新增算法

- 必须附带冒烟测试（`@pytest.mark.smoke`），验证算法在简单环境上能正常训练
- 在 `configs/` 中提供默认配置文件
- 建议在 `examples/` 中提供示例脚本

### 新增环境包装器

- 必须附带单元测试
- 确保与 Gymnasium API 兼容

## 分层贡献

AxiomRL 采用三层架构，不同层级的贡献标准如下：

### :material-shield-check: 核心层（Core） — `rl_training.core`

最严格的审查标准：

- 所有变更需经过至少两名维护者审查
- 必须附带完整的单元测试和集成测试
- 必须更新 API 文档
- 遵循语义化版本，破坏性变更仅在主版本（major）发布
- 弃用需提前一个次版本（minor）发出警告

**当前稳定核心算法：** A2C, BC, CQL, DQN, DiscreteSAC, IQL, PPO, SAC, TD3, TRPO

### :material-flask: 实验层（Experimental） — `rl_training.experimental`

中等审查标准：

- 至少一名维护者审查
- 必须附带冒烟测试
- API 可能在次版本中变更

### :material-account-group: 社区层（Contrib） — `rl_training.contrib`

社区驱动：

- 一名维护者审查
- 必须附带基础测试
- 无稳定性保证

### :material-paw: 预设层（Zoo） — `zoo/`

基准预设：

- 需冒烟测试通过
- 提供完整的 YAML 配置
- 无稳定性保证

## PR 流程

### 1. 创建分支

```bash
git checkout -b feature/your-feature-name
```

### 2. 开发与测试

```bash
# 编写代码和测试
# ...

# 本地验证
make verify
```

### 3. 提交

```bash
git add .
git commit -m "feat: 简要描述你的变更"
```

!!! info "提交信息规范"

    推荐使用 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/) 格式：

    - `feat:` — 新功能
    - `fix:` — 修复缺陷
    - `docs:` — 文档变更
    - `test:` — 测试相关
    - `refactor:` — 代码重构
    - `chore:` — 构建/工具链变更

### 4. 推送并创建 PR

```bash
git push origin feature/your-feature-name
```

在 GitHub 上创建 Pull Request，并在描述中说明：

- 变更的动机和背景
- 主要的实现思路
- 测试方法和结果
- 是否涉及破坏性变更

### 5. 代码审查

- 维护者会在合理时间内进行审查
- 根据反馈进行修改后重新推送
- 所有 CI 检查必须通过

### 6. 合并

审查通过且 CI 绿灯后，维护者将合并你的 PR。感谢你的贡献！
