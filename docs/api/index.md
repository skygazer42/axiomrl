---
title: API 参考
icon: material/code-tags
---

# API 参考

AxiomRL 的 Python API 采用分层设计，提供不同稳定性级别的接口。

## 导入层级

```
rl_training              # 根包（稳定核心 + 弃用兼容）
rl_training.core         # 稳定核心 API
rl_training.experimental # 实验性 API
rl_training.contrib      # 社区扩展
```

### rl_training（根包）

根包通过延迟导入转发稳定核心 API 的所有名称。对于不在稳定核心中的算法名称，根包会发出 `DeprecationWarning` 并从 `rl_training.api` 中回退加载。

```python
import rl_training

# 稳定名称 - 直接可用
algo = rl_training.PPO
config = rl_training.TrainConfig

# 实验性名称 - 触发弃用警告
# algo = rl_training.SomeExperimentalAlgo  # DeprecationWarning
```

### rl_training.core（稳定核心）

受语义版本控制管理，在 1.x 版本内保证 API 稳定。包含 10 种核心算法和 `TrainConfig`。

```python
from rl_training.core import PPO, SAC, TrainConfig, STABLE_ALGORITHMS
```

### rl_training.experimental（实验性）

包含 70 余种算法的完整集合。实验性 API 可能在次要版本之间发生变化。

```python
from rl_training.experimental import EXPERIMENTAL_ALGORITHMS
```

### rl_training.contrib（社区扩展）

社区贡献的算法和工具，如 `RecurrentPPO`。

```python
from rl_training.contrib import RecurrentPPO
```

## 文档导航

<div class="grid cards" markdown>

-   :material-shield-check:{ .lg .middle } **稳定核心 API**

    ---

    10 种核心算法 + TrainConfig，受语义版本管理，1.x 版本内 API 稳定。

    [:octicons-arrow-right-24: 查看详情](stable-core.md)

-   :material-flask:{ .lg .middle } **实验性 API**

    ---

    70+ 算法的完整集合、社区扩展以及从根包导入的迁移指南。

    [:octicons-arrow-right-24: 查看详情](experimental.md)

</div>
