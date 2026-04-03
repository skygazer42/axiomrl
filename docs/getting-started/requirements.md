---
title: 环境要求
icon: material/clipboard-check
---

# 环境要求

在安装 AxiomRL 之前，请确认你的系统满足以下基本要求。

---

## Python 版本

AxiomRL 要求 **Python 3.10 或更高版本**。以下是各版本的支持状态：

| Python 版本 | 支持状态 | 说明 |
|:-----------:|:--------:|------|
| 3.10 | :material-check-circle:{ .green } 完全支持 | 最低支持版本 |
| 3.11 | :material-check-circle:{ .green } 完全支持 | 推荐版本 |
| 3.12 | :material-check-circle:{ .green } 完全支持 | 最新稳定版 |
| < 3.10 | :material-close-circle:{ .red } 不支持 | 缺少必要的语言特性 |

检查当前 Python 版本：

```bash
python --version
```

!!! warning "版本要求"
    AxiomRL 使用了 Python 3.10 引入的结构化模式匹配（`match/case`）和改进的类型注解等特性，因此不兼容更早的 Python 版本。

---

## 核心依赖

以下依赖会在安装 AxiomRL 时自动拉取，无需手动安装：

| 依赖 | 用途 | 说明 |
|------|------|------|
| **PyTorch** | 深度学习框架 | 推荐使用最新稳定版 |
| **Gymnasium** | 环境接口 | 标准强化学习环境 API |
| **NumPy** | 数值计算 | 数组运算与数据处理 |
| **PyYAML** | 配置解析 | 读取 YAML 配置文件 |
| **TensorBoard** | 训练监控 | 可视化训练曲线与指标 |

!!! tip "PyTorch 安装建议"
    如果你需要 GPU 支持，建议先根据 [PyTorch 官方指引](https://pytorch.org/get-started/locally/) 安装对应 CUDA 版本的 PyTorch，再安装 AxiomRL。这样可以确保 GPU 加速正常工作。

---

## 操作系统

| 操作系统 | 支持状态 | 说明 |
|:--------:|:--------:|------|
| **Linux** (Ubuntu 20.04+) | :material-check-circle:{ .green } 推荐 | 最佳性能，完整功能支持 |
| **macOS** (12+) | :material-check-circle:{ .green } 支持 | Apple Silicon (M1/M2) 与 Intel 均可 |
| **Windows** (10/11) | :material-alert-circle:{ .yellow } 实验性 | 基本功能可用，部分环境可能不兼容 |

!!! note "Linux 推荐"
    强化学习训练通常涉及大量环境交互和 GPU 计算。Linux 在这些场景下具有最佳的性能和兼容性，建议在生产环境中使用 Linux 系统。

---

## GPU 支持（可选）

GPU 加速可以显著提升训练速度，但并非必需。AxiomRL 在 CPU 上同样可以完整运行。

| 要求 | 说明 |
|------|------|
| **NVIDIA GPU** | 支持 CUDA 的 NVIDIA 显卡 |
| **CUDA Toolkit** | 11.8 或 12.x，与 PyTorch 版本匹配 |
| **cuDNN** | 与 CUDA 版本对应的 cuDNN 库 |

检查 CUDA 是否可用：

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, 设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"
```

---

## 环境验证

安装完成后，使用内置诊断命令一键检查所有依赖：

```bash
axiomrl doctor
```

该命令将检查以下项目：

- [x] Python 版本是否满足要求
- [x] 核心依赖是否已正确安装
- [x] PyTorch 是否可用及 CUDA 状态
- [x] Gymnasium 环境是否正常
- [x] 可选依赖的安装状态

!!! example "预期输出示例"
    ```
    AxiomRL 环境诊断
    ──────────────────────────────
    Python        3.11.5          ✓
    PyTorch       2.2.0+cu121     ✓
    CUDA          可用 (RTX 4090) ✓
    Gymnasium     0.29.1          ✓
    NumPy         1.26.2          ✓
    PyYAML        6.0.1           ✓
    TensorBoard   2.15.1          ✓
    ale-py        0.8.1           ✓ (可选)
    minari        0.4.3           ✓ (可选)
    ──────────────────────────────
    全部检查通过！
    ```

---

## 下一步

环境就绪后，继续进行安装：

[:octicons-arrow-right-24: 安装指南](installation.md)
