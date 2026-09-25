# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库性质

《深度学习入门：基于Python的理论与实现》（斋藤康毅）的学习仓库：按书逐章手写/抄录代码，以可独立运行的脚本形式组织。无 git、无依赖清单、无测试框架、无 lint。

## 运行方式

从仓库根目录直接运行脚本（无需配置 PYTHONPATH，脚本里的 `sys.path.append(os.pardir)` 是沿袭书本的写法，实际并不需要）：

```bash
python3 test01-3_mnist.py
```

环境为系统 Python 3.12 + numpy 2.x + matplotlib。注意：

- 带 `plt.show()` 的脚本（test01-1、test01-4、test01-8）会阻塞等待关闭窗口。
- 训练脚本（test01-7、test01-8）用**数值微分**算梯度（`network.numerical_gradient`）：每个参数要正反扰动各跑一次完整前向传播，hidden=20 时约 1.6 万个参数，每个 mini-batch 迭代约 3.2 万次前向，CPU 上极慢。书上高速版 `network.gradient()`（误差反向传播法）尚未实现，脚本里以注释形式预留（`# 高速版!`）。

## 结构约定

- 根目录 `test01-N_名称.py`：学习进度脚本，`01` 是学习阶段编号，每个脚本自包含，用于演示/验证当节内容（print 输出或 matplotlib 绘图）。
- `common/`：书中公共库代码，`__init__.py` 统一再导出。
  - `functions.py`：激活函数 step/sigmoid/relu/identity、损失函数 MSE/交叉熵、含溢出对策且支持 batch 的 softmax、梯度演示用的 function_1/function_2。
  - `gradients.py`（文件名带 s）：numerical_diff、numerical_gradient（np.nditer 多维版）、gradient_descent。
  - `two_layer_net.py`：`TwoLayerNet` 类，权重存 `params` dict。`input_size`/`output_size` 由数据和标签决定，`hidden_size` 是唯一可自由调整的超参数。
- `dataset/`：`mnist.py` 提供 `load_mnist(normalize, flatten, one_hot_label)`。原始 gz 文件在 `dataset/mnist/` 子目录（缺失时从 ossci-datasets S3 镜像下载），解析后缓存为 `mnist.pkl`；缓存已存在，`load_mnist()` 直接读 pickle，离线可用。

## 已知坑

- 本书代码基于旧版 numpy 编写，运行环境是 numpy 2.x，改写或新增书后代码时留意已废弃 API。
- `output_size` 不是提速旋钮：必须等于标签列数，否则 `cross_entropy_error` 里 `t * np.log(y + 1e-7)` 形状广播失败（详见 docs/01）。想缩小网络提速只动 `hidden_size` 和 `iters_num`。
- `numerical_gradient` 会临时**原地修改**传入的数组（扰动后还原）；模块级函数与 `TwoLayerNet.numerical_gradient` 方法同名，方法内部调用的是模块级函数，阅读时注意区分。

## 文档约定

`docs/` 下的文档按"序号-名称"格式命名（如 `00-fix_mnist_path.md`、`01-two_layer_net_dims_and_speed.md`），序号从 00 递增。
文档为中文，写给人读（用户借此自行改代码学习），生成方案文档时不直接改动代码。

## 风格

代码注释与 docstring 均为中文（对齐书中译本），新增代码保持一致。
