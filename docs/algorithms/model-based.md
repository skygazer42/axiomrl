---
title: 模型基础算法
icon: material/earth
---

# 模型基础算法（Model-Based）

模型基础强化学习通过学习环境的动态模型（转移函数和奖励函数）来辅助策略学习。智能体可以在学到的模型中"想象"经验，从而大幅提升采样效率。本页涵盖三大方向：基于模型的策略优化、世界模型方法和树搜索与规划方法。

---

## 基于模型的策略优化

这类方法学习一个环境模型，并使用该模型生成模拟数据来辅助无模型算法的训练。

---

### PETS

**Probabilistic Ensemble Trajectory Sampling（概率集成轨迹采样）**

使用集成的概率神经网络来建模环境动态，通过 CEM（交叉熵方法）在模型中进行在线规划来选择动作。

> Chua et al., "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models", 2018. [arXiv:1805.12114](https://arxiv.org/abs/1805.12114)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-3` | 模型学习率 |
| `n_ensemble` | `5` | 集成模型数量 |
| `planning_horizon` | `25` | CEM 规划步数 |
| `n_candidates` | `500` | CEM 候选动作序列数 |
| `n_elites` | `50` | CEM 精英数 |
| `cem_iterations` | `5` | CEM 迭代次数 |
| `model_hidden_size` | `200` | 模型隐藏层大小 |
| `model_train_freq` | `250` | 模型训练频率（环境步数） |

#### YAML 配置示例

```yaml
algorithm: pets
algo_kwargs:
  learning_rate: 1e-3
  n_ensemble: 5
  planning_horizon: 25
  n_candidates: 500
  n_elites: 50
  cem_iterations: 5
  model_hidden_size: 200
  model_train_freq: 250
```

#### Python API 示例

```python
from rl_training.experimental import PETS

model = PETS(
    env_id="Pusher-v4",
    n_ensemble=5,
    planning_horizon=25,
    n_candidates=500,
)
model.train(total_timesteps=50_000)
```

!!! tip "最佳实践"
    - PETS 在极低采样量（数千步）下即可学到有效策略。
    - 概率集成模型的不确定性估计是其鲁棒性的关键。
    - 规划时间较长，不适合实时控制场景。

---

### MBPO

**Model-Based Policy Optimization（基于模型的策略优化）**

在真实数据上训练环境模型，然后用模型生成短轨迹来扩充 SAC 的回放缓冲区，以可控的方式利用模型数据。

> Janner et al., "When to Trust Your Model: Model-Based Policy Optimization", 2019. [arXiv:1906.08253](https://arxiv.org/abs/1906.08253)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | SAC 策略学习率 |
| `model_learning_rate` | `1e-3` | 环境模型学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `n_ensemble` | `7` | 集成模型数量 |
| `n_elites` | `5` | 精英模型数量 |
| `rollout_length` | `1` | 模型生成轨迹的长度 |
| `real_ratio` | `0.05` | 真实数据在训练批中的比例 |
| `model_train_freq` | `250` | 模型训练频率 |

#### YAML 配置示例

```yaml
algorithm: mbpo
algo_kwargs:
  learning_rate: 3e-4
  model_learning_rate: 1e-3
  buffer_size: 1_000_000
  batch_size: 256
  n_ensemble: 7
  n_elites: 5
  rollout_length: 1
  real_ratio: 0.05
  model_train_freq: 250
```

#### Python API 示例

```python
from rl_training.experimental import MBPO

model = MBPO(
    env_id="HalfCheetah-v4",
    n_ensemble=7,
    rollout_length=1,
    real_ratio=0.05,
)
model.train(total_timesteps=200_000)
```

!!! tip "最佳实践"
    - `rollout_length` 是最敏感的超参数，过长会积累模型误差。通常从 1 开始逐步增加。
    - `real_ratio` 控制真实数据和模型数据的混合比例。
    - MBPO 在 MuJoCo 基准上仅用 1/10 的采样量即可达到 SAC 的性能。

---

### MOPO

**Model-based Offline Policy Optimization（基于模型的离线策略优化）**

在离线 RL 中使用学到的模型，通过在奖励中减去模型不确定性来惩罚不确定的状态-动作对，实现保守的模型利用。

> Yu et al., "MOPO: Model-based Offline Policy Optimization", 2020. [arXiv:2005.13239](https://arxiv.org/abs/2005.13239)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | 学习率 |
| `model_learning_rate` | `1e-3` | 模型学习率 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `n_ensemble` | `7` | 集成模型数量 |
| `penalty_coef` | `1.0` | 不确定性惩罚系数 |
| `rollout_length` | `5` | 模型展开长度 |
| `dataset_path` | 必填 | 离线数据集路径 |

#### YAML 配置示例

```yaml
algorithm: mopo
algo_kwargs:
  learning_rate: 3e-4
  model_learning_rate: 1e-3
  batch_size: 256
  n_ensemble: 7
  penalty_coef: 1.0
  rollout_length: 5
  dataset_path: "data/hopper_medium.npz"
```

#### Python API 示例

```python
from rl_training.experimental import MOPO

model = MOPO(
    env_id="Hopper-v4",
    dataset_path="data/hopper_medium.npz",
    penalty_coef=1.0,
    rollout_length=5,
)
model.train(total_timesteps=500_000)
```

!!! tip "最佳实践"
    - `penalty_coef` 控制对模型不确定区域的惩罚力度，需要根据数据集质量调整。
    - MOPO 同时属于离线 RL 和模型基础方法，也收录在[离线 RL 算法](offline.md)中。
    - 在数据覆盖范围有限的场景中，模型可以有效扩展可用数据。

---

## World Models（世界模型）

世界模型方法学习一个紧凑的环境动态模型，在潜在空间中进行想象和规划，能够处理高维观测（如像素输入）。

---

### Dreamer

**Dreamer**

在学到的潜在空间世界模型中通过想象（imagination）进行策略优化，实现了高维像素观测下的高效学习。

> Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination", 2019. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `6e-4` | 模型学习率 |
| `actor_lr` | `8e-5` | Actor 学习率 |
| `critic_lr` | `8e-5` | Critic 学习率 |
| `batch_size` | `50` | 序列批大小 |
| `sequence_length` | `50` | 训练序列长度 |
| `gamma` | `0.99` | 折扣因子 |
| `imagination_horizon` | `15` | 想象规划步数 |
| `stoch_size` | `30` | 随机状态维度 |
| `deter_size` | `200` | 确定性状态维度 |
| `free_nats` | `3.0` | KL 散度自由比特数 |

#### YAML 配置示例

```yaml
algorithm: dreamer
algo_kwargs:
  learning_rate: 6e-4
  actor_lr: 8e-5
  critic_lr: 8e-5
  batch_size: 50
  sequence_length: 50
  imagination_horizon: 15
  stoch_size: 30
  deter_size: 200
  free_nats: 3.0
```

#### Python API 示例

```python
from rl_training.experimental import Dreamer

model = Dreamer(
    env_id="dm_control/walker-walk",
    imagination_horizon=15,
    stoch_size=30,
    deter_size=200,
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - Dreamer 通过在潜在空间想象来避免高维像素的重构开销。
    - `imagination_horizon` 决定了想象轨迹的长度，影响值估计的准确性。
    - 在 DMControl 基准上采样效率显著优于无模型方法。

---

### DreamerV3

**DreamerV3**

Dreamer 的第三个版本，通过 symlog 预测、离散化潜在空间和多项工程改进，实现了无需超参数调优的通用世界模型。

> Hafner et al., "Mastering Diverse Domains through World Models", 2023. [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete + Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `model_lr` | `1e-4` | 世界模型学习率 |
| `actor_lr` | `3e-5` | Actor 学习率 |
| `critic_lr` | `3e-5` | Critic 学习率 |
| `batch_size` | `16` | 序列批大小 |
| `sequence_length` | `64` | 训练序列长度 |
| `gamma` | `0.997` | 折扣因子 |
| `imagination_horizon` | `15` | 想象步数 |
| `stoch_classes` | `32` | 离散潜在类别数 |
| `stoch_dims` | `32` | 每个类别的维度 |
| `deter_size` | `512` | 确定性状态维度 |
| `model_size` | `"medium"` | 模型规模（`small`/`medium`/`large`/`xlarge`） |

#### YAML 配置示例

```yaml
algorithm: dreamerv3
algo_kwargs:
  model_lr: 1e-4
  actor_lr: 3e-5
  critic_lr: 3e-5
  batch_size: 16
  sequence_length: 64
  imagination_horizon: 15
  model_size: "medium"
```

#### Python API 示例

```python
from rl_training.experimental import DreamerV3

model = DreamerV3(
    env_id="Minecraft-diamond",
    model_size="large",
    imagination_horizon=15,
)
model.train(total_timesteps=100_000_000)
```

!!! tip "最佳实践"
    - DreamerV3 是目前最通用的世界模型方法，在 Atari、DMControl、Minecraft 等多种领域均表现出色。
    - `model_size` 可以根据任务复杂度调整，简单任务用 `small`，复杂任务用 `large`。
    - 超参数鲁棒性是其最大优势，通常使用默认参数即可。
    - 同时支持离散和连续动作空间。

---

### EADream

**EADream（Energy-Aware Dreamer）**

在 Dreamer 框架中引入能量模型来改善潜在空间的表示质量，增强世界模型的预测准确性。

> 基于 Dreamer 架构与能量模型的集成。

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `model_lr` | `3e-4` | 世界模型学习率 |
| `actor_lr` | `8e-5` | Actor 学习率 |
| `critic_lr` | `8e-5` | Critic 学习率 |
| `batch_size` | `50` | 序列批大小 |
| `imagination_horizon` | `15` | 想象步数 |
| `energy_coef` | `0.1` | 能量模型正则化系数 |
| `ebm_hidden_size` | `256` | 能量模型隐藏层大小 |

#### YAML 配置示例

```yaml
algorithm: eadream
algo_kwargs:
  model_lr: 3e-4
  actor_lr: 8e-5
  critic_lr: 8e-5
  batch_size: 50
  imagination_horizon: 15
  energy_coef: 0.1
  ebm_hidden_size: 256
```

#### Python API 示例

```python
from rl_training.experimental import EADream

model = EADream(
    env_id="dm_control/cheetah-run",
    imagination_horizon=15,
    energy_coef=0.1,
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - EADream 通过能量模型提升了潜在空间的表示质量。
    - `energy_coef` 需要平衡能量约束和世界模型训练。
    - 在具有复杂动态的环境中可能优于标准 Dreamer。

---

### PO-Dreamer

**PO-Dreamer（Partially Observable Dreamer）**

针对部分可观测环境优化的 Dreamer 变体，增强了对历史信息的编码能力和对观测缺失的鲁棒性。

> 基于 Dreamer 的部分可观测扩展。

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `model_lr` | `6e-4` | 世界模型学习率 |
| `actor_lr` | `8e-5` | Actor 学习率 |
| `critic_lr` | `8e-5` | Critic 学习率 |
| `batch_size` | `50` | 序列批大小 |
| `sequence_length` | `50` | 训练序列长度 |
| `imagination_horizon` | `15` | 想象步数 |
| `belief_hidden_size` | `400` | 信念状态隐藏层大小 |
| `obs_dropout` | `0.1` | 观测 Dropout 率 |

#### YAML 配置示例

```yaml
algorithm: po_dreamer
algo_kwargs:
  model_lr: 6e-4
  actor_lr: 8e-5
  critic_lr: 8e-5
  batch_size: 50
  sequence_length: 50
  imagination_horizon: 15
  belief_hidden_size: 400
  obs_dropout: 0.1
```

#### Python API 示例

```python
from rl_training.experimental import PODreamer

model = PODreamer(
    env_id="FlickeringCheetah-v0",
    imagination_horizon=15,
    belief_hidden_size=400,
)
model.train(total_timesteps=2_000_000)
```

!!! tip "最佳实践"
    - PO-Dreamer 针对观测噪声和部分缺失的场景做了专门优化。
    - `obs_dropout` 在训练时模拟观测缺失，提升鲁棒性。
    - 对于标准 MDP 任务，使用普通 Dreamer 或 DreamerV3 即可。

---

## 树搜索与规划

结合学到的模型与树搜索（如 MCTS）进行规划，在离散决策空间中表现出色。

---

### MuZero

**MuZero**

在无需已知环境规则的情况下，通过学习的隐式模型结合 MCTS 进行规划。在 Atari 和棋类游戏上取得超人表现。

> Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model", 2019. [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-3` | 学习率（含调度） |
| `batch_size` | `1024` | 小批量大小 |
| `gamma` | `0.997` | 折扣因子 |
| `num_simulations` | `50` | MCTS 模拟次数 |
| `unroll_steps` | `5` | 模型展开步数 |
| `td_steps` | `5` | TD 回报步数 |
| `num_actors` | `128` | 并行 Actor 数量 |
| `replay_buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `dirichlet_alpha` | `0.25` | 根节点探索噪声 |

#### YAML 配置示例

```yaml
algorithm: muzero
algo_kwargs:
  learning_rate: 3e-3
  batch_size: 1024
  gamma: 0.997
  num_simulations: 50
  unroll_steps: 5
  td_steps: 5
  num_actors: 128
  dirichlet_alpha: 0.25
```

#### Python API 示例

```python
from rl_training.experimental import MuZero

model = MuZero(
    env_id="Breakout-v5",
    num_simulations=50,
    unroll_steps=5,
    num_actors=128,
)
model.train(total_timesteps=100_000_000)
```

!!! tip "最佳实践"
    - `num_simulations` 直接决定规划质量，越大性能越好但推理越慢。
    - MuZero 的计算需求极高，建议使用分布式训练。
    - 对于棋类任务，增大 `num_simulations` 效果显著。

---

### EfficientZero

**EfficientZero**

在 MuZero 基础上引入自监督一致性损失和模型的临时差分值预测，在 Atari 100k 基准上以极少的数据量达到超人水平。

> Ye et al., "Mastering Atari Games with Limited Data", 2021. [arXiv:2111.00210](https://arxiv.org/abs/2111.00210)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-3` | 学习率 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.997` | 折扣因子 |
| `num_simulations` | `50` | MCTS 模拟次数 |
| `unroll_steps` | `5` | 模型展开步数 |
| `consistency_coef` | `2.0` | 自监督一致性损失权重 |
| `ssl_predictor_dim` | `512` | SSL 预测器维度 |
| `num_actors` | `8` | 并行 Actor 数量 |

#### YAML 配置示例

```yaml
algorithm: efficientzero
algo_kwargs:
  learning_rate: 3e-3
  batch_size: 256
  gamma: 0.997
  num_simulations: 50
  unroll_steps: 5
  consistency_coef: 2.0
  num_actors: 8
```

#### Python API 示例

```python
from rl_training.experimental import EfficientZero

model = EfficientZero(
    env_id="Pong-v5",
    num_simulations=50,
    consistency_coef=2.0,
)
model.train(total_timesteps=100_000)
```

!!! tip "最佳实践"
    - EfficientZero 在 Atari 100k 基准上是数据效率最高的方法之一。
    - 自监督一致性损失是其采样高效的关键创新。
    - 计算需求低于 MuZero，更适合资源受限场景。

---

### ScaleZero

**ScaleZero**

对 MuZero 进行规模化改进，通过更大的网络和更多的计算资源来提升模型的表示和规划能力。

> 基于 MuZero 的大规模扩展实现。

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-3` | 学习率 |
| `batch_size` | `2048` | 小批量大小 |
| `gamma` | `0.997` | 折扣因子 |
| `num_simulations` | `100` | MCTS 模拟次数 |
| `model_size` | `"large"` | 模型规模 |
| `num_actors` | `256` | 并行 Actor 数量 |

#### YAML 配置示例

```yaml
algorithm: scalezero
algo_kwargs:
  learning_rate: 1e-3
  batch_size: 2048
  gamma: 0.997
  num_simulations: 100
  model_size: "large"
  num_actors: 256
```

#### Python API 示例

```python
from rl_training.experimental import ScaleZero

model = ScaleZero(
    env_id="Breakout-v5",
    num_simulations=100,
    model_size="large",
    num_actors=256,
)
model.train(total_timesteps=200_000_000)
```

!!! tip "最佳实践"
    - ScaleZero 需要大量计算资源（多 GPU/TPU）。
    - 在充足资源下，扩大模型和 MCTS 模拟次数可以持续提升性能。
    - 适合需要极致性能的研究场景。

---

### Gumbel MuZero

**Gumbel MuZero**

使用 Gumbel 技巧替代传统的 PUCT 搜索，在 MCTS 中实现更高效的策略改进，尤其在低模拟次数下表现更优。

> Danihelka et al., "Policy improvement by planning with Gumbel", 2021. [arXiv:2104.06303](https://arxiv.org/abs/2104.06303)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `3e-3` | 学习率 |
| `batch_size` | `1024` | 小批量大小 |
| `gamma` | `0.997` | 折扣因子 |
| `num_simulations` | `16` | MCTS 模拟次数（可以更少） |
| `unroll_steps` | `5` | 模型展开步数 |
| `max_num_considered_actions` | `16` | 最大考虑动作数 |
| `c_visit` | `50` | 访问次数温度 |
| `c_scale` | `1.0` | Q 值缩放系数 |

#### YAML 配置示例

```yaml
algorithm: gumbel_muzero
algo_kwargs:
  learning_rate: 3e-3
  batch_size: 1024
  gamma: 0.997
  num_simulations: 16
  unroll_steps: 5
  max_num_considered_actions: 16
```

#### Python API 示例

```python
from rl_training.experimental import GumbelMuZero

model = GumbelMuZero(
    env_id="Go-9x9",
    num_simulations=16,
    max_num_considered_actions=16,
)
model.train(total_timesteps=50_000_000)
```

!!! tip "最佳实践"
    - Gumbel MuZero 的核心优势在于低模拟次数下仍能保证策略改进。
    - `num_simulations=16` 即可获得不错性能，远低于标准 MuZero 的 50+。
    - 在动作空间较大的棋类游戏中优势明显。

---

## 其他模型基础方法

---

### DIAMOND

**DIAMOND（Diffusion for World Modeling）**

使用扩散模型作为世界模型来生成逼真的环境模拟，能够产生高质量的像素级环境预测。

> Alonso et al., "Diffusion for World Modeling: Visual Details Matter in Atari", 2024. [arXiv:2405.12399](https://arxiv.org/abs/2405.12399)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` | 扩散模型学习率 |
| `actor_lr` | `3e-4` | Actor 学习率 |
| `batch_size` | `64` | 小批量大小 |
| `diffusion_steps` | `10` | 扩散推理步数 |
| `noise_schedule` | `"cosine"` | 噪声调度方式 |
| `imagination_horizon` | `50` | 想象步数 |
| `context_frames` | `4` | 上下文帧数 |

#### YAML 配置示例

```yaml
algorithm: diamond
algo_kwargs:
  learning_rate: 1e-4
  actor_lr: 3e-4
  batch_size: 64
  diffusion_steps: 10
  noise_schedule: "cosine"
  imagination_horizon: 50
  context_frames: 4
```

#### Python API 示例

```python
from rl_training.experimental import DIAMOND

model = DIAMOND(
    env_id="Breakout-v5",
    diffusion_steps=10,
    imagination_horizon=50,
)
model.train(total_timesteps=10_000_000)
```

!!! tip "最佳实践"
    - DIAMOND 的扩散世界模型能生成极其逼真的环境预测。
    - `diffusion_steps` 影响生成质量和速度的平衡，10 步是合理的起点。
    - 在 Atari 游戏中能够完全在想象中训练策略。

---

### Horizon Imagination

**Horizon Imagination（水平想象）**

通过可控的想象水平（horizon）来平衡世界模型的使用程度，在不同复杂度的任务中自适应调整。

> 基于自适应水平想象的世界模型方法。

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `model_lr` | `3e-4` | 模型学习率 |
| `actor_lr` | `1e-4` | Actor 学习率 |
| `batch_size` | `128` | 小批量大小 |
| `min_horizon` | `1` | 最小想象步数 |
| `max_horizon` | `20` | 最大想象步数 |
| `horizon_schedule` | `"adaptive"` | 水平调度策略 |

#### YAML 配置示例

```yaml
algorithm: horizon_imagination
algo_kwargs:
  model_lr: 3e-4
  actor_lr: 1e-4
  batch_size: 128
  min_horizon: 1
  max_horizon: 20
  horizon_schedule: "adaptive"
```

#### Python API 示例

```python
from rl_training.experimental import HorizonImagination

model = HorizonImagination(
    env_id="Seaquest-v5",
    min_horizon=1,
    max_horizon=20,
)
model.train(total_timesteps=5_000_000)
```

!!! tip "最佳实践"
    - 自适应水平调度能根据模型质量动态调整想象步数。
    - 在模型初期不准时使用短水平，模型成熟后增加水平以利用更多想象数据。

---

### JOWA

**JOWA（Joint World-Action Model）**

联合学习世界模型和动作模型，通过共享表示来提升两者的学习效率。

> 基于联合世界-动作建模的方法。

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `model_lr` | `3e-4` | 联合模型学习率 |
| `actor_lr` | `1e-4` | Actor 学习率 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `imagination_horizon` | `10` | 想象步数 |
| `joint_coef` | `0.5` | 联合训练损失权重 |

#### YAML 配置示例

```yaml
algorithm: jowa
algo_kwargs:
  model_lr: 3e-4
  actor_lr: 1e-4
  batch_size: 256
  imagination_horizon: 10
  joint_coef: 0.5
```

#### Python API 示例

```python
from rl_training.experimental import JOWA

model = JOWA(
    env_id="Walker2d-v4",
    imagination_horizon=10,
    joint_coef=0.5,
)
model.train(total_timesteps=500_000)
```

!!! tip "最佳实践"
    - `joint_coef` 平衡世界模型和动作模型的训练，需要根据任务调整。
    - 共享表示可以加速学习，但也可能导致优化冲突。

---

### MOW

**MOW（Model of the World）**

使用大规模神经网络来构建通用世界模型，旨在学习可迁移的环境理解能力。

> 基于大规模通用世界模型的方法。

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `model_lr` | `1e-4` | 模型学习率 |
| `actor_lr` | `3e-5` | Actor 学习率 |
| `batch_size` | `128` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `model_size` | `"medium"` | 模型规模 |
| `imagination_horizon` | `15` | 想象步数 |
| `pretrained_model` | `null` | 预训练模型路径 |

#### YAML 配置示例

```yaml
algorithm: mow
algo_kwargs:
  model_lr: 1e-4
  actor_lr: 3e-5
  batch_size: 128
  model_size: "medium"
  imagination_horizon: 15
  pretrained_model: null
```

#### Python API 示例

```python
from rl_training.experimental import MOW

model = MOW(
    env_id="dm_control/humanoid-walk",
    model_size="medium",
    imagination_horizon=15,
)
model.train(total_timesteps=2_000_000)
```

!!! tip "最佳实践"
    - MOW 支持在多个环境上预训练世界模型并迁移到目标环境。
    - 预训练模型可以通过 `pretrained_model` 参数加载。
    - 在目标环境数据有限时，迁移学习的优势尤为明显。

---

### Twisted

**Twisted**

使用扭曲（twisted）目标函数来改善世界模型中策略学习的质量，通过修正想象轨迹的分布来减少模型偏差。

> 基于扭曲目标函数的世界模型方法。

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Discrete

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `model_lr` | `3e-4` | 模型学习率 |
| `actor_lr` | `1e-4` | Actor 学习率 |
| `batch_size` | `128` | 小批量大小 |
| `gamma` | `0.99` | 折扣因子 |
| `twist_coef` | `1.0` | 扭曲修正系数 |
| `imagination_horizon` | `10` | 想象步数 |
| `n_particles` | `16` | 粒子采样数 |

#### YAML 配置示例

```yaml
algorithm: twisted
algo_kwargs:
  model_lr: 3e-4
  actor_lr: 1e-4
  batch_size: 128
  twist_coef: 1.0
  imagination_horizon: 10
  n_particles: 16
```

#### Python API 示例

```python
from rl_training.experimental import Twisted

model = Twisted(
    env_id="MsPacman-v5",
    twist_coef=1.0,
    imagination_horizon=10,
)
model.train(total_timesteps=5_000_000)
```

!!! tip "最佳实践"
    - `twist_coef` 控制分布修正的强度，过大可能引入额外方差。
    - 在模型偏差较大的场景中，扭曲修正可以有效改善策略学习。

---

## 目标条件方法

---

### HER

**Hindsight Experience Replay（事后经验回放）**

通过在回放时将失败轨迹的实际到达状态替换为目标，将失败经验转化为成功经验。是解决稀疏奖励目标条件任务的关键技术。

> Andrychowicz et al., "Hindsight Experience Replay", 2017. [arXiv:1707.01495](https://arxiv.org/abs/1707.01495)

**稳定性：** <span class="badge badge-experimental">Experimental</span> &nbsp; **动作空间：** Continuous

#### 关键超参数

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-3` | 学习率 |
| `buffer_size` | `1_000_000` | 回放缓冲区大小 |
| `batch_size` | `256` | 小批量大小 |
| `gamma` | `0.98` | 折扣因子 |
| `strategy` | `"future"` | 目标替换策略 (`future`/`final`/`episode`/`random`) |
| `n_sampled_goal` | `4` | 每个转换采样的替代目标数 |
| `base_algorithm` | `"sac"` | 底层 RL 算法 |

#### YAML 配置示例

```yaml
algorithm: her
algo_kwargs:
  learning_rate: 1e-3
  buffer_size: 1_000_000
  batch_size: 256
  gamma: 0.98
  strategy: "future"
  n_sampled_goal: 4
  base_algorithm: "sac"
```

#### Python API 示例

```python
from rl_training.experimental import HER

model = HER(
    env_id="FetchReach-v2",
    strategy="future",
    n_sampled_goal=4,
    base_algorithm="sac",
)
model.train(total_timesteps=1_000_000)
```

!!! tip "最佳实践"
    - `strategy="future"` 是最有效的目标替换策略，在大多数任务上推荐使用。
    - HER 是一种通用的包装器，可以与 SAC、TD3、DDPG 等离策略算法结合。
    - 环境必须支持目标条件接口（`GoalEnv`），提供 `achieved_goal` 和 `desired_goal`。
    - `n_sampled_goal=4` 在实践中是一个好的默认值。
