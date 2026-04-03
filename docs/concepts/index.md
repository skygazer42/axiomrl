---
title: 核心概念
icon: material/cube-outline
---

# 核心概念

AxiomRL 的设计理念围绕**可靠性**、**可扩展性**和**易用性**展开。本章节将帮助您理解框架的核心组件及其协作方式。

---

## 快速导航

<div class="grid cards" markdown>

-   :material-layers-triple:{ .lg .middle } **系统架构**

    ---

    了解 AxiomRL 的三层架构设计：Core 稳定层、Experimental 实验层、Contrib 社区层和 Zoo 基准层，以及它们各自的职责与稳定性保证。

    [:octicons-arrow-right-24: 查看架构详情](architecture.md)

-   :material-chart-timeline-variant-shimmer:{ .lg .middle } **训练流程**

    ---

    从 YAML 配置到模型检查点，完整了解 AxiomRL 的训练管线：配置加载、算法初始化、数据收集、经验缓存、训练更新、评估与日志。

    [:octicons-arrow-right-24: 查看训练流程](training-workflow.md)

-   :material-cog-outline:{ .lg .middle } **配置系统**

    ---

    AxiomRL 以 `TrainConfig` 数据类为核心，支持 YAML 文件、CLI 覆盖和预设配置的多层配置体系，实现配置驱动的实验管理。

    [:octicons-arrow-right-24: 查看配置详情](../configuration/index.md)

</div>

---

## 设计原则

AxiomRL 遵循以下核心设计原则：

### :material-layers-outline: 分层架构

框架采用 **Core / Experimental / Contrib / Zoo** 四层结构。Core 层包含 10 个经过充分验证的算法，API 稳定且遵循语义化版本控制；Experimental 层提供 70+ 实验性算法，迭代更快但 API 可能变更；Contrib 层承载社区贡献；Zoo 层提供基准预设和排行榜。

### :material-tag-check-outline: 语义化版本

Core 层严格遵循 [语义化版本控制（SemVer）](https://semver.org/lang/zh-CN/) 规范。在同一个主版本内，Core API 保证向后兼容，让您的训练脚本和集成代码不会因升级而意外中断。

### :material-file-cog-outline: 配置驱动

所有训练实验均由 `TrainConfig` 数据类驱动。从算法选择、环境参数到检查点策略，全部通过声明式配置完成，便于复现、共享和版本管理。
