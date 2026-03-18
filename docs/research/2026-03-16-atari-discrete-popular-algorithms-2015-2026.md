# Atari / 离散动作（2015–2026）常见“年度热门”算法速记

> 说明：
> - 这份清单更接近“每年在 Atari / 离散控制圈子里最常被引用、最常出现在开源复现/基准里的方法/家族”，不是严格的学术统计排名。
> - 2015–2024 相对稳定；2025 基本稳定。2026 已补充到 2026-03-17 可核验的早期趋势项，但仍可能随着全年进展变化。
> - AxiomRL 对应的 `algo` 名称会标注在括号里，未实现会标注 `TODO`.

---

## 2015
- Double DQN（`double_dqn`）
- Dueling DQN（`dueling_dqn`）
- Prioritized Experience Replay / PER（`prioritized_dqn`）

## 2016
- A3C / A2C（AxiomRL: `a2c` 更接近 A2C；A3C 方向可作为 TODO）
- N-step returns（`n_step_dqn`；也常作为组合组件）
-（可选）TRPO（`trpo`，更偏 on-policy 基准）

## 2017
- Rainbow（`rainbow_dqn`）
- Distributional C51（`c51_dqn`）
- Noisy Networks（`noisy_dqn`）

## 2018
- IMPALA（`impala`）
- IQN（`iqn`）
- Ape-X（`apex_dqn`，单机近似：multi-actor epsilon + prioritized replay + n-step）

## 2019
- MuZero（`muzero`，MVP：PUCT MCTS + dynamics/reward/value；暂不含 reanalysis）
- R2D2（`r2d2`）
- FQF（`fqf`）

## 2020
- PPG（`ppg`）
- Agent57（`agent57`：MVP/lite，当前实现为 `R2D2 + RND intrinsic bonus`，不含完整 NGU / episodic memory / meta-controller）
- DreamerV2 系列（AxiomRL: `dreamer` 是像素离散 MVP，非完整 DreamerV2/V3）

## 2021
- Decision Transformer（`decision_transformer`，更偏离线/序列建模）
- EfficientZero（`efficientzero`：MVP/lite，当前实现为 `MuZero + latent consistency loss`，不含完整 reanalysis / value prefix / full paper stack）
- SPR（`spr`：MVP/lite，当前实现为 `pixel DQN + 1-step latent self-prediction`，不含完整 multi-step Rainbow SPR）

## 2022
- Gumbel MuZero（`gumbel_muzero`：MVP/lite，当前实现为 `MuZero + root Gumbel action selection`，不含完整 sequential halving / full policy improvement）
- Offline-to-Online / 数据混合范式（AxiomRL 方向：`rlpd`、`iql`、`cql` 等更偏连续/离线，但可作为策略层面参考）
-（可选）更强的 MuZero 工程化变体（TODO）

## 2023
- DreamerV3（`dreamerv3`：MVP/lite；当前实现为 `Dreamer + symlog/symexp targets + unimix actor sampling`，不含完整 DreamerV3 paper stack）
- 更强的通用表征/数据规模化训练路线（趋势项，TODO）

## 2024
- DIAMOND（`diamond`：MVP/lite；当前实现为 `Dreamer + denoising world-model auxiliary loss`，不含完整 diffusion world model / train-in-imagination stack）
- 强化学习 + 大模型/Transformer（DART 方向，TODO）
- 更强的 model-based / planning 混合（趋势项，TODO）

## 2025
- JOWA（`jowa`：MVP/lite；当前实现为 `pixel Q-learning + shared world-model regularization`，不含完整 offline transformer tokenizer / planning stack）
- 更强的离线到在线迁移、数据回放策略（趋势项，TODO）

## 2026（截至 2026-03-17）
- EAWM / EADream（`eadream`：MVP/lite；当前实现为 `Dreamer-family + auxiliary event-boundary prediction + event-aware latent modulation`，不含完整 event harmonizer / full EAWM stack）
- MoW（`mow`：MVP/lite；当前实现为 `Dreamer-family + mixture-of-experts latent dynamics/value/policy`，不含完整 modular VAE / transformer / task clustering stack）
- ScaleZero（`scalezero`：MVP/lite；当前实现为 `MuZero-family + mixture-of-experts latent dynamics/prediction`，不含完整 multi-task MoE / LoRA / DPS stack）
- Horizon Imagination（`horizon_imagination`：MVP/lite；当前实现为 `DIAMOND-lite + horizon-weighted imagination schedule + rollout stabilization`，不含完整 parallel denoising / sub-frame budget decoupling / full train-in-imagination stack）
- PO-Dreamer（`po_dreamer`：MVP/lite；当前实现为 `Dreamer + memory-guided latent fusion + memory-prediction auxiliary loss`，不含完整 partially observable recurrent world-model paper stack；对应 2026-03-17 可见的 ICLR 2026 under-review 趋势项）
- TWISTED（`twisted`：MVP/lite；当前实现为 `Dreamer + frame-persistence / token-reuse auxiliary loss`，不含完整 transformer token world model / spatio-temporal encoding / graph-based optimal decoding stack；对应 2026-03-17 可见的 ICLR 2026 under-review 趋势项）

---

## AxiomRL（Atari 像素）当前建议优先跑通的组合
- 2015 经典增强：`double_dqn` / `dueling_dqn` / `prioritized_dqn`
- 2017–2019 分布式价值分布：`c51_dqn` / `qr_dqn` / `iqn` / `fqf`
- 2017 组合基线：`rainbow_dqn`
- 2019 复现/稳定性常用：`r2d2`
- On-policy 对照：`a2c` / `impala` / `ppg` / `ppo` / `recurrent_ppo`
