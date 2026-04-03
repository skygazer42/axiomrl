---
title: 开发者指南
icon: material/code-braces
---

# 开发者指南

欢迎参与 AxiomRL 的开发！本指南面向希望为项目贡献代码、修复缺陷或扩展功能的开发者。

## 快速导航

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } **贡献指南**

    ---

    开发环境搭建、工作流、代码规范与 PR 流程

    [:octicons-arrow-right-24: 阅读贡献指南](contributing.md)

-   :material-file-tree:{ .lg .middle } **项目结构详解**

    ---

    源码目录布局、模块职责与构建系统说明

    [:octicons-arrow-right-24: 了解项目结构](project-structure.md)

</div>

## 快速搭建开发环境

只需两步即可启动本地开发：

```bash
# 1. 以可编辑模式安装，包含所有开发依赖
pip install -e ".[dev]"

# 2. 安装 Git 预提交钩子
pre-commit install
```

!!! tip "推荐使用虚拟环境"

    建议在 `venv` 或 `conda` 环境中进行开发，以避免依赖冲突：

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    pip install -e ".[dev]"
    pre-commit install
    ```

## 开发工作流概览

AxiomRL 使用 `Makefile` 统一管理开发命令，日常开发遵循以下流程：

```mermaid
graph LR
    A[编写代码] --> B[make lint]
    B --> C[make typecheck]
    C --> D[make test-fast]
    D --> E[make verify]
    E --> F[提交 PR]
```

| 命令 | 说明 |
|---|---|
| `make install-dev` | 安装开发依赖 |
| `make lint` | Ruff 代码风格检查 |
| `make typecheck` | Mypy 静态类型检查 |
| `make test` | 运行全部测试 |
| `make test-fast` | 运行快速测试（排除 slow 标记） |
| `make test-integration` | 运行集成测试 |
| `make test-smoke` | 运行冒烟测试 |
| `make verify` | 完整验证流水线 |
| `make build` | 构建分发包 |
| `make precommit` | 手动触发预提交钩子 |

!!! info "提交前请运行 `make verify`"

    `make verify` 会依次执行 lint、typecheck、test-fast、test-integration、test-smoke 和 build，确保你的变更通过所有质量关卡。

## 分层贡献模型

AxiomRL 采用三层架构，不同层级对代码质量和审查力度有不同要求：

| 层级 | 路径 | 审查标准 |
|---|---|---|
| **核心（Core）** | `rl_training.core` | 严格审查，需完整测试和文档 |
| **社区扩展（Contrib）** | `rl_training.contrib` | 社区驱动，需基础测试 |
| **预设（Zoo）** | `zoo/` | 基准预设，需冒烟测试通过 |

更多细节请参阅 [贡献指南](contributing.md) 和 [兼容性与版本策略](../compatibility.md)。
