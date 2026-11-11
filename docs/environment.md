# Quoridor 运行环境

本项目已迁移到当前用户的系统默认 `/usr/bin/python3`，依赖安装在
`~/.local/lib/python3.10/site-packages`。重新打开终端后即可启动训练或网页服务：

```bash
python3 train.py
```

```bash
python3 app.py
```

以上命令在项目根目录执行。网页地址为 `http://127.0.0.1:8000`。
训练参数在 `train.py` 顶部手动修改，默认权重位于 `models/`。

## 环境配置

- Python 3.10。
- TensorFlow 2.21.0、Keras 3.12.4。
- NumPy 1.26.4、h5py 3.14.0。
- CUDA 12.9、cuDNN 9.27，NVIDIA 组件安装在当前用户的 Python 包目录内。

这里的 CUDA 12.9 是实际加载的运行库版本；TensorFlow 安装包的编译信息显示 CUDA 12.5.1。
该安装包未列出 `sm_120`，部分 TensorFlow 核心算子仍可能需要首次启动时进行 PTX 编译。
本机已通过真实 GPU 计算验证；不能把首次编译提示与 cuDNN 越界错误等同处理。

`~/.config/tensorflow-gpu.sh` 配置 NVIDIA 库和 `ptxas` 路径，并清除旧版 cuDNN 的临时绕过设置。
`.bashrc` 和 `.profile` 自动加载此配置，系统原有 TensorFlow 2.14 已被新版替换。
无须激活虚拟环境。
旧终端若正在使用虚拟环境，先执行 `deactivate`；然后执行以下命令更新当前终端：

```bash
source ~/.config/tensorflow-gpu.sh
hash -r
```

## 依赖与检查

主要依赖见 `requirements.txt`，完整依赖版本见 `requirements-lock.txt`。
`environment-check.json` 记录当前机器的 GPU、实际加载的 NVIDIA 库路径和验证结果。

已完成的检查包括：

- 批量 64 的 CPU/GPU 前向计算与全部梯度对比。
- 40 次真实 GPU 更新，批量在 1、4、16、64 之间变化，训练和推理各只追踪一次。
- 临时权重保存与载入，以及 TensorFlow 2.14/Keras 2 权重的兼容性。
- 两轮多进程短局训练，每轮 3 局、2 个工作进程，分别验证从零训练和接续训练。
- 实际 HTTP 接口的模型载入与上传、人机先后手、双人对战和撤回。
- 已核对训练依赖的版本；测试期间未出现 cuDNN 越界写入或重复追踪警告。

系统在迁移前已有一项与本项目无关的依赖问题：`pygobject 3.42.1` 缺少 `pycairo`。
因此系统整体的 `pip check` 仍会报告这一项，TensorFlow 和上述项目功能检查已通过。
该问题也记录在 `environment-check.json` 中。

在另一台 Linux/WSL2 机器重建同样的 Python 3.10 依赖时执行：

```bash
python3 -m pip install --user -r requirements-lock.txt
```

还需配置 NVIDIA 库路径；本机配置内容见 `~/.config/tensorflow-gpu.sh`。
当前机器可直接检查依赖和 GPU：

```bash
python3 -m pip check
python3 -c 'import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices("GPU"))'
```

## 权重

`RESUME = True` 时，程序在 `MODEL_PATH` 存在的情况下加载权重；文件不存在则从随机权重开始。
网络结构参数与权重对应的残差块数和通道数必须一致。
续训会恢复网络权重和 BatchNorm 统计量，优化器状态和经验池会重新初始化。
