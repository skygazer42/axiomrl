---
title: 算法参考
icon: material/brain
---

# 算法参考

AxiomRL 提供了超过 **80 种**强化学习算法的统一实现，涵盖 **6 大类别**：策略梯度（On-Policy）、离策略连续控制（Off-Policy Continuous）、基于值的离散控制（Value-Based Discrete）、离线强化学习（Offline RL）、模型基础与世界模型（Model-Based & World Models）以及目标条件（Goal-Conditioned）。每种算法均按照稳定性分为三个层级：

<div class="grid cards" markdown>

- :material-shield-check:{ .lg .middle } **Core（核心层）**

    ---

    经过充分测试与验证的稳定算法，适合生产环境使用。

    导入路径：`from rl_training.core import ...`

- :material-flask:{ .lg .middle } **Experimental（实验层）**

    ---

    功能完整但仍在持续优化中的算法，API 可能变更。

    导入路径：`from rl_training.experimental import ...`

- :material-account-group:{ .lg .middle } **Contrib（社区层）**

    ---

    社区贡献的扩展算法，由社区维护。

    导入路径：`from rl_training.contrib import ...`

</div>

---

## 完整算法总览

下表列出了 AxiomRL 中所有可用的算法及其基本信息。

### 策略梯度算法（On-Policy）

| 算法名称 | 类型 | 动作空间 | 稳定性 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| PPO (Proximal Policy Optimization) | 策略梯度 | Discrete + Continuous | <span class="badge badge-stable">Core</span> | [详情](on-policy.md#ppo) |
| A2C (Advantage Actor-Critic) | 策略梯度 | Discrete + Continuous | <span class="badge badge-stable">Core</span> | [详情](on-policy.md#a2c) |
| TRPO (Trust Region Policy Optimization) | 策略梯度 | Discrete + Continuous | <span class="badge badge-stable">Core</span> | [详情](on-policy.md#trpo) |
| IMPALA (Importance Weighted Actor-Learner) | 策略梯度 | Discrete + Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](on-policy.md#impala) |
| APPO (Asynchronous PPO) | 策略梯度 | Discrete + Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](on-policy.md#appo) |
| PPG (Phasic Policy Gradient) | 策略梯度 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](on-policy.md#ppg) |
| GAIL (Generative Adversarial Imitation Learning) | 模仿学习 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](on-policy.md#gail) |
| MARWIL (Monotonic Advantage Re-Weighted IL) | 模仿学习 | Discrete + Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](on-policy.md#marwil) |
| AWR (Advantage Weighted Regression) | 策略梯度 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](on-policy.md#awr) |
| OpenAI-ES (Evolution Strategy) | 进化策略 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](on-policy.md#openai-es) |
| ARS (Augmented Random Search) | 进化策略 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](on-policy.md#ars) |
| RecurrentPPO | 策略梯度 | Discrete + Continuous | <span class="badge badge-contrib">Contrib</span> | [详情](on-policy.md#recurrentppo) |

### 离策略连续控制算法（Off-Policy Continuous）

| 算法名称 | 类型 | 动作空间 | 稳定性 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| SAC (Soft Actor-Critic) | 离策略 | Continuous | <span class="badge badge-stable">Core</span> | [详情](off-policy.md#sac) |
| TD3 (Twin Delayed DDPG) | 离策略 | Continuous | <span class="badge badge-stable">Core</span> | [详情](off-policy.md#td3) |
| DDPG (Deep Deterministic Policy Gradient) | 离策略 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#ddpg) |
| D4PG (Distributed DDPG) | 离策略 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#d4pg) |
| TQC (Truncated Quantile Critics) | 离策略 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#tqc) |
| CrossQ | 离策略 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#crossq) |
| REDQ (Randomized Ensemble Double Q) | 离策略 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#redq) |
| RLPD (RL with Prior Data) | 离策略 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#rlpd) |
| NAF (Normalized Advantage Functions) | 离策略 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#naf) |
| CURL (Contrastive Unsupervised RL) | 离策略 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#curl) |
| DrQ (Data-regularized Q) | 离策略 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#drq) |
| DrQ-v2 | 离策略 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#drq-v2) |

### 基于值的离散控制算法（Value-Based Discrete）

| 算法名称 | 类型 | 动作空间 | 稳定性 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| DQN (Deep Q-Network) | 值函数 | Discrete | <span class="badge badge-stable">Core</span> | [详情](off-policy.md#dqn) |
| DiscreteSAC | 值函数 | Discrete | <span class="badge badge-stable">Core</span> | [详情](off-policy.md#discretesac) |
| Double DQN | 值函数 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#double-dqn) |
| Dueling DQN | 值函数 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#dueling-dqn) |
| Noisy DQN | 值函数 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#noisy-dqn) |
| N-step DQN | 值函数 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#n-step-dqn) |
| Prioritized DQN (PER) | 值函数 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#prioritized-dqn) |
| Rainbow DQN | 值函数 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#rainbow-dqn) |
| C51 DQN (Categorical DQN) | 值函数 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#c51-dqn) |
| QR-DQN (Quantile Regression DQN) | 值函数 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#qr-dqn) |
| IQN (Implicit Quantile Network) | 值函数 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#iqn) |
| FQF (Fully Parameterized Quantile Function) | 值函数 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#fqf) |
| R2D2 (Recurrent Replay Distributed DQN) | 值函数 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#r2d2) |
| DRQN (Deep Recurrent Q-Network) | 值函数 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#drqn) |
| Agent57 | 值函数 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#agent57) |
| SPR (Self-Predictive Representations) | 值函数 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](off-policy.md#spr) |

### 离线强化学习算法（Offline RL）

| 算法名称 | 类型 | 动作空间 | 稳定性 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| IQL (Implicit Q-Learning) | 离线 RL | Continuous | <span class="badge badge-stable">Core</span> | [详情](offline.md#iql) |
| CQL (Conservative Q-Learning) | 离线 RL | Continuous | <span class="badge badge-stable">Core</span> | [详情](offline.md#cql) |
| BC (Behavioral Cloning) | 离线 RL | Discrete + Continuous | <span class="badge badge-stable">Core</span> | [详情](offline.md#bc) |
| BCQ (Batch-Constrained Q-learning) | 离线 RL | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](offline.md#bcq) |
| BEAR (Bootstrapping Error Accumulation Reduction) | 离线 RL | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](offline.md#bear) |
| CRR (Critic Regularized Regression) | 离线 RL | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](offline.md#crr) |
| TD3+BC | 离线 RL | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](offline.md#td3bc) |
| AWAC (Advantage Weighted Actor-Critic) | 离线 RL | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](offline.md#awac) |
| ReBRAC | 离线 RL | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](offline.md#rebrac) |
| XQL (Extreme Q-Learning) | 离线 RL | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](offline.md#xql) |
| EDAC (Error-Diversified Ensemble Actor-Critic) | 离线 RL | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](offline.md#edac) |
| Cal-QL (Calibrated Conservative Q-Learning) | 离线 RL | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](offline.md#cal-ql) |
| MOPO (Model-based Offline Policy Optimization) | 离线 RL | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](model-based.md#mopo) |
| Decision Transformer | 离线 RL | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](offline.md#decision-transformer) |

### 模型基础与世界模型算法（Model-Based & World Models）

| 算法名称 | 类型 | 动作空间 | 稳定性 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| PETS (Probabilistic Ensemble Trajectory Sampling) | 模型基础 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](model-based.md#pets) |
| MBPO (Model-Based Policy Optimization) | 模型基础 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](model-based.md#mbpo) |
| Dreamer | 世界模型 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](model-based.md#dreamer) |
| DreamerV3 | 世界模型 | Discrete + Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](model-based.md#dreamerv3) |
| MuZero | 树搜索 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](model-based.md#muzero) |
| EfficientZero | 树搜索 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](model-based.md#efficientzero) |
| ScaleZero | 树搜索 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](model-based.md#scalezero) |
| Gumbel MuZero | 树搜索 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](model-based.md#gumbel-muzero) |
| DIAMOND | 世界模型 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](model-based.md#diamond) |
| EADream | 世界模型 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](model-based.md#eadream) |
| PO-Dreamer | 世界模型 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](model-based.md#po-dreamer) |
| Horizon Imagination | 世界模型 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](model-based.md#horizon-imagination) |
| JOWA | 世界模型 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](model-based.md#jowa) |
| MOW | 世界模型 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](model-based.md#mow) |
| Twisted | 世界模型 | Discrete | <span class="badge badge-experimental">Experimental</span> | [详情](model-based.md#twisted) |

### 目标条件算法（Goal-Conditioned）

| 算法名称 | 类型 | 动作空间 | 稳定性 | 链接 |
| :--- | :--- | :--- | :--- | :--- |
| HER (Hindsight Experience Replay) | 目标条件 | Continuous | <span class="badge badge-experimental">Experimental</span> | [详情](model-based.md#her) |

---

## 分类导航

<div class="grid cards" markdown>

- :material-trending-up:{ .lg .middle } **[策略梯度算法](on-policy.md)**

    ---

    PPO、A2C、TRPO 等在线策略梯度方法，直接通过与环境交互来优化策略。

    **12 种算法** | 3 Core + 8 Experimental + 1 Contrib

- :material-database-arrow-left:{ .lg .middle } **[离策略算法](off-policy.md)**

    ---

    SAC、TD3、DQN 等离策略方法，能够高效利用历史经验进行学习。

    **28 种算法** | 4 Core + 24 Experimental

- :material-harddisk:{ .lg .middle } **[离线 RL 算法](offline.md)**

    ---

    IQL、CQL、BC 等从固定数据集中学习策略的方法，无需与环境在线交互。

    **14 种算法** | 3 Core + 11 Experimental

- :material-earth:{ .lg .middle } **[模型基础算法](model-based.md)**

    ---

    Dreamer、MuZero、MBPO 等基于环境模型的方法，通过学习环境动态来提升采样效率。

    **16 种算法** | 全部 Experimental

</div>

---

## 快速选择指南

不确定该用哪种算法？以下指南帮助你快速定位：

### 按任务场景

| 你的需求 | 推荐算法 | 理由 |
| :--- | :--- | :--- |
| 连续控制入门，快速获得好结果 | **PPO** | 超参数鲁棒，易于调试，Core 级别稳定 |
| 连续控制追求最优采样效率 | **SAC** | 离策略+自动温度调节，采样效率高 |
| 离散动作空间（如 Atari） | **DQN** 或 **Rainbow DQN** | DQN 简单可靠，Rainbow 集成多种改进 |
| 离散动作空间+策略梯度 | **PPO**（离散模式） | 策略梯度方法中最稳定的选择 |
| 只有固定数据集、无法与环境交互 | **IQL** 或 **CQL** | 离线 RL 核心算法，避免分布外动作 |
| 模仿学习（有专家演示数据） | **BC** 或 **GAIL** | BC 简单直接，GAIL 可超越专家 |
| 需要极高的采样效率 | **DreamerV3** 或 **MBPO** | 模型基础方法，通过想象产生经验 |
| 高维像素观测 | **DrQ-v2** 或 **CURL** | 内置图像增强和表示学习 |
| 分布式大规模训练 | **IMPALA** 或 **APPO** | 支持多 Actor 异步采样架构 |
| 稀疏奖励+目标达成 | **HER** + SAC/TD3 | HER 通过事后经验回放解决稀疏奖励 |
| 部分可观测环境 | **RecurrentPPO** 或 **R2D2** | 内置循环网络处理序列信息 |
| 追求 SOTA 离线 RL 表现 | **Cal-QL** 或 **ReBRAC** | 最新离线 RL 研究成果 |

### 按算法特性

| 特性 | 可选算法 |
| :--- | :--- |
| :material-shield-check: 核心稳定 | PPO, A2C, TRPO, SAC, TD3, DQN, DiscreteSAC, IQL, CQL, BC |
| :material-lightning-bolt: 采样效率最高 | SAC, TD3, DreamerV3, MBPO |
| :material-scale-balance: 超参数鲁棒 | PPO, DreamerV3, CrossQ |
| :material-server-network: 支持分布式 | IMPALA, APPO, D4PG, R2D2 |
| :material-image: 支持像素输入 | DrQ, DrQ-v2, CURL, SPR, DreamerV3 |
| :material-memory: 支持循环网络 | RecurrentPPO, R2D2, DRQN, Dreamer |

---

## 版本兼容性

所有算法均兼容 AxiomRL **v1.0.0**。关于具体版本的 API 变更记录，请参阅 [兼容性说明](../compatibility.md)。
