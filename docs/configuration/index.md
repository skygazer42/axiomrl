---
title: 配置参考
icon: material/cog
---

# 配置参考

AxiomRL 使用 YAML 文件驱动训练配置。所有配置最终解析为 [`TrainConfig`](train-config.md) 数据类，确保类型安全和可验证性。

## 配置形式

AxiomRL 支持两种配置形式：

### 1. 直接配置

将所有 `TrainConfig` 字段直接写在 YAML 文件顶层：

```yaml
algo: PPO
env_id: CartPole-v1
seed: 42
total_timesteps: 100000
output_dir: runs/cartpole-ppo
```

### 2. 预设链接配置

通过 `config` 字段引用一个预设文件，并在同一文件中覆盖部分字段：

```yaml
config: presets/atari-ppo.yaml
seed: 123
total_timesteps: 500000
output_dir: runs/atari-override
```

!!! tip "选择建议"
    - **直接配置**适合独立实验和快速原型开发。
    - **预设链接配置**适合团队协作和基准复现，可以在共享预设的基础上做最小修改。

## 文档导航

<div class="grid cards" markdown>

-   :material-file-cog:{ .lg .middle } **TrainConfig 完整参考**

    ---

    所有 15 个配置字段的完整说明、类型、默认值，以及 YAML 示例和 CLI 覆盖选项。

    [:octicons-arrow-right-24: 查看详情](train-config.md)

-   :material-chart-timeline-variant:{ .lg .middle } **调度器配置**

    ---

    学习率、epsilon、熵系数、裁剪范围等超参数的动态调度策略。

    [:octicons-arrow-right-24: 查看详情](scheduling.md)

</div>
