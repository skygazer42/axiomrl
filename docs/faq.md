---
title: 常见问题
---

# 常见问题

本页汇总了 AxiomRL 的常见问题及其解答。点击问题即可展开查看。

---

??? question "如何选择合适的算法？"

    选择算法时需要考虑以下几个维度：

    **1. 动作空间类型**

    | 动作空间 | 推荐算法 |
    |---|---|
    | 离散（Discrete） | DQN, DiscreteSAC, A2C, PPO |
    | 连续（Continuous） | SAC, TD3, PPO, TRPO, A2C |

    **2. 在线策略 vs 离线策略**

    | 类型 | 特点 | 推荐算法 |
    |---|---|---|
    | 在线策略（On-Policy） | 样本效率低，但稳定性好 | PPO, A2C, TRPO |
    | 离线策略（Off-Policy） | 样本效率高，支持经验回放 | SAC, TD3, DQN, DiscreteSAC |

    **3. 在线学习 vs 离线学习**

    | 类型 | 特点 | 推荐算法 |
    |---|---|---|
    | 在线学习（Online） | 需要与环境实时交互 | PPO, SAC, DQN 等 |
    | 离线学习（Offline） | 仅从已有数据集学习 | BC, CQL, IQL |

    **快速建议：**

    - 不确定时，先试 **PPO**（通用性最强）
    - 需要高样本效率时，选 **SAC**（连续）或 **DQN**（离散）
    - 只有离线数据时，选 **CQL** 或 **IQL**

??? question "如何使用 GPU 训练？"

    AxiomRL 通过配置中的 `device` 参数控制训练设备：

    ```yaml
    # 配置文件中设置
    device: auto    # 自动检测，优先使用 GPU
    device: cuda    # 强制使用 GPU
    device: cuda:0  # 指定 GPU 编号
    device: cpu     # 强制使用 CPU
    ```

    ```bash
    # 命令行指定
    axiomrl train --config my_config.yaml --device cuda
    ```

    **GPU 环境准备：**

    1. 安装与你的 CUDA 版本匹配的 PyTorch：
       ```bash
       pip install torch --index-url https://download.pytorch.org/whl/cu121
       ```
    2. 验证 CUDA 可用性：
       ```python
       import torch
       print(torch.cuda.is_available())  # 应输出 True
       print(torch.cuda.device_count())  # 显示 GPU 数量
       ```
    3. 使用 `axiomrl doctor` 诊断环境配置：
       ```bash
       axiomrl doctor
       ```

??? question "如何恢复中断的训练？"

    使用 `axiomrl resume` 命令即可从上次检查点恢复训练：

    ```bash
    # 从最近的检查点恢复
    axiomrl resume --run-dir runs/my_experiment/

    # 从指定检查点恢复
    axiomrl resume --checkpoint runs/my_experiment/checkpoints/step_10000.pt
    ```

    **工作原理：**

    - AxiomRL 在训练过程中定期保存检查点，包含模型参数、优化器状态、训练步数等完整状态
    - `resume` 命令会加载检查点并从中断处继续训练
    - 随机数状态也会被恢复，确保训练的确定性

    !!! tip "建议"
        建议在配置中启用定期检查点保存，以减少因中断导致的进度损失。

??? question "如何使用自定义环境？"

    有两种方式接入自定义环境：

    **方式一：直接通过 `env_id` 和 `env_kwargs`**

    ```yaml
    env_id: MyCustomEnv-v0
    env_kwargs:
      param1: value1
      param2: value2
    ```

    **方式二：通过 Gymnasium 注册**

    ```python
    import gymnasium as gym

    # 注册自定义环境
    gym.register(
        id="MyCustomEnv-v0",
        entry_point="my_package.envs:MyCustomEnv",
        max_episode_steps=1000,
    )
    ```

    然后在配置中引用：

    ```yaml
    env_id: MyCustomEnv-v0
    ```

    **自定义环境要求：**

    - 必须遵循 Gymnasium API（继承 `gymnasium.Env`）
    - 必须正确定义 `observation_space` 和 `action_space`
    - 必须实现 `reset()` 和 `step()` 方法

??? question "如何进行多 seed 实验？"

    AxiomRL 支持通过命令行或配置文件运行多 seed 实验：

    **命令行方式：**

    ```bash
    # 运行 5 个不同 seed 的实验
    axiomrl train --config my_config.yaml --seeds 5
    ```

    **配置文件方式：**

    ```yaml
    benchmark:
      seeds: 5
    ```

    **说明：**

    - 每个 seed 会启动独立的训练运行
    - 结果保存在各自的子目录中
    - 支持对多 seed 结果进行统计汇总（均值、标准差等）

??? question "stable API 和 experimental API 有什么区别？"

    AxiomRL 使用三层稳定性模型，其中 stable 和 experimental 的主要区别如下：

    | 特性 | Stable（稳定） | Experimental（实验） |
    |---|---|---|
    | 命名空间 | `axiomrl.core` | `axiomrl.experimental` |
    | 版本约束 | 语义化版本严格约束 | 可能在次版本中变更 |
    | 破坏性变更 | 仅在主版本发布 | 可能在次版本发布 |
    | 弃用流程 | 至少提前一个次版本警告 | 不保证弃用过渡期 |
    | 适用场景 | 生产环境 | 新特性试用 |

    ```python
    # 稳定 API — 受语义化版本保护
    from axiomrl.core import PPO, SAC

    # 实验性 API — 可能在次版本中变更
    from axiomrl.experimental import some_new_feature
    ```

    详细说明请参阅 [兼容性与版本策略](compatibility.md)。

??? question "如何贡献新算法？"

    贡献新算法推荐遵循以下步骤：

    **1. 确定贡献层级**

    - 新算法通常先进入 **Contrib 层**（`axiomrl.contrib`）
    - 经过充分验证后可提升至实验层或稳定层

    **2. 实现要求**

    - 在 `src/axiomrl/contrib/` 或 `src/axiomrl/algorithms/` 下创建算法模块
    - 在 `configs/` 中提供默认配置
    - 在 `examples/` 中提供示例脚本

    **3. 测试要求**

    - 必须包含冒烟测试（`@pytest.mark.smoke`），验证在简单环境上能正常训练
    - 建议包含单元测试，覆盖核心逻辑
    - 运行 `make verify` 确保所有检查通过

    **4. 提交 PR**

    详细流程请参阅 [贡献指南](developer/contributing.md)。

??? question "如何查看训练进度？"

    AxiomRL 提供多种方式监控训练进度：

    **TensorBoard**

    ```bash
    # 启动 TensorBoard
    tensorboard --logdir runs/

    # 然后在浏览器中打开 http://localhost:6006
    ```

    TensorBoard 中可查看：

    - 奖励曲线（episode reward）
    - 损失曲线（loss）
    - 学习率变化
    - 其他自定义指标

    **元数据文件**

    每次训练运行会在运行目录下生成 `metadata.json`，包含：

    - 训练配置
    - 当前训练步数
    - 最优奖励
    - 运行状态

    ```bash
    # 查看运行元数据
    cat runs/my_experiment/metadata.json
    ```

??? question "支持分布式训练吗？"

    分布式训练目前在 AxiomRL 的**路线图**中，尚未正式支持。

    **当前状态：**

    - 单机单 GPU 训练已完全支持
    - 多 seed 并行实验可通过独立进程实现
    - 分布式训练（多机多卡）正在规划中

    **替代方案：**

    - 对于需要大规模并行的场景，可以手动启动多个独立训练进程
    - 可以通过脚本管理多组实验的并行执行

    如果你对分布式训练有需求或想法，欢迎在 GitHub Issues 中参与讨论。

??? question "如何报告 bug？"

    如果你发现了 bug，请通过 GitHub Issues 提交报告。

    **Bug 报告应包含：**

    1. **环境信息**：运行 `axiomrl doctor` 并附上输出
    2. **复现步骤**：清晰描述如何触发 bug
    3. **预期行为**：描述你期望看到的结果
    4. **实际行为**：描述实际发生的情况
    5. **错误日志**：附上完整的错误堆栈信息

    **报告模板：**

    ```markdown
    ## 环境信息
    <!-- 粘贴 axiomrl doctor 输出 -->

    ## 复现步骤
    1. ...
    2. ...

    ## 预期行为
    ...

    ## 实际行为
    ...

    ## 错误日志
    ```

    !!! tip "提交前请检查"
        在提交新 Issue 前，请先搜索已有 Issues，确认该问题尚未被报告。
