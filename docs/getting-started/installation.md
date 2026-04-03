---
title: 安装指南
icon: material/download
---

# 安装指南

AxiomRL 已发布至 PyPI，可通过 `pip` 快速安装。根据你的使用场景，选择合适的安装方式。

---

## 安装方式

=== "pip 安装（推荐）"

    安装核心包，包含全部 80+ 算法和 CLI 工具：

    ```bash
    pip install axiomrl
    ```

    !!! tip "虚拟环境"
        建议在虚拟环境中安装，避免依赖冲突：

        ```bash
        python -m venv .venv
        source .venv/bin/activate  # Linux / macOS
        # .venv\Scripts\activate   # Windows
        pip install axiomrl
        ```

=== "完整安装"

    安装核心包及所有可选依赖（Atari 环境、离线 RL 数据集）：

    ```bash
    pip install axiomrl[atari,offline]
    ```

    !!! note "Atari 环境"
        Atari 环境需要额外下载 ROM 文件。安装 `ale-py` 后，按照其文档完成 ROM 配置。

=== "开发模式"

    适用于需要修改源码或参与项目开发的场景：

    ```bash
    git clone https://github.com/axiomrl/axiomrl.git
    cd axiomrl
    pip install -e ".[dev]"
    ```

    !!! info "开发依赖"
        开发模式会额外安装测试框架、代码格式化工具、文档构建工具等开发依赖。

---

## 可选依赖

AxiomRL 采用模块化设计，可选依赖根据使用场景按需安装：

| 扩展名 | 安装命令 | 依赖包 | 用途 |
|:------:|---------|:------:|------|
| `atari` | `pip install axiomrl[atari]` | ale-py | Atari 游戏环境支持 |
| `offline` | `pip install axiomrl[offline]` | minari | 离线 RL 数据集加载 |
| `dev` | `pip install axiomrl[dev]` | 构建工具集 | 开发、测试、文档构建 |

也可以组合安装多个扩展：

```bash
pip install axiomrl[atari,offline]
```

---

## 版本信息

当前最新版本为 **1.0.0**。查看已安装的版本：

```bash
pip show axiomrl
```

或在 Python 中获取：

```python
import rl_training
print(rl_training.__version__)
```

---

## 验证安装

### 方法一：导入检查

```bash
python -c "import rl_training; print(f'AxiomRL v{rl_training.__version__} 安装成功！')"
```

预期输出：

```
AxiomRL v1.0.0 安装成功！
```

### 方法二：环境诊断

使用内置诊断命令全面检查安装状态：

```bash
axiomrl doctor
```

!!! success "安装成功"
    如果 `axiomrl doctor` 显示所有核心依赖检查通过，说明安装已完成，可以开始使用了。

### 方法三：运行测试训练

快速启动一个简单的训练任务，验证完整流程：

=== "Python API"

    ```python
    from rl_training.core import PPO, TrainConfig

    config = TrainConfig(
        algo="PPO",
        env_id="CartPole-v1",
        seed=42,
        total_timesteps=1_000,
        output_dir="runs/test_install",
    )
    ppo = PPO(config)
    ppo.learn()
    print("安装验证通过！")
    ```

=== "CLI"

    ```bash
    axiomrl train --config configs/ppo/cartpole.yaml \
        --total-timesteps 1000 \
        --output-dir runs/test_install
    ```

---

## 常见问题

!!! question "安装时 PyTorch 版本冲突怎么办？"
    建议先单独安装 PyTorch（按照 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择对应 CUDA 版本），再安装 AxiomRL：

    ```bash
    # 以 CUDA 12.1 为例
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install axiomrl
    ```

!!! question "`axiomrl` 命令无法找到？"
    确保安装目录在系统 `PATH` 中。如果使用虚拟环境，确认已激活环境：

    ```bash
    which axiomrl        # 检查命令位置
    source .venv/bin/activate  # 激活虚拟环境
    ```

!!! question "Apple Silicon (M1/M2) 如何安装？"
    AxiomRL 完全兼容 Apple Silicon。PyTorch 在 macOS 上支持 MPS 加速：

    ```bash
    pip install axiomrl
    python -c "import torch; print(torch.backends.mps.is_available())"
    ```

---

## 下一步

安装完成后，跟随快速上手教程运行第一个实验：

[:octicons-arrow-right-24: 5 分钟上手](quickstart.md)
