# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库性质

《深度学习入门：基于Python的理论与实现》（斋藤康毅，"鱼书"）的学习仓库：按书逐章手写/抄录代码，以可独立运行的脚本形式组织。无依赖清单、无测试框架、无 lint。

## 运行方式

从仓库根目录直接运行脚本（无需配置 PYTHONPATH，脚本里的 `sys.path.append(os.pardir)` 是沿袭书本的写法，实际并不需要）：

```bash
python3 test03-3_gradient-check.py
```

环境为系统 Python 3.12 + numpy 2.x + matplotlib。注意：

- 带 `plt.show()` 的脚本（test01-1、test01-4、test02-3）会阻塞等待关闭窗口。
- 训练脚本分两类：test02 系列用**数值微分**求梯度（每个参数正反扰动各跑一次完整前向传播，CPU 上慢，只适合小网络少量迭代）；test03-2 起改用 `network.gradient()`（误差反向传播法，common/net.py），可跑上万次迭代。
- test03-3 是梯度检验（gradient check）：对比数值梯度与反向传播梯度，各参数平均绝对误差应在 1e-10 量级。

## 结构约定

- 根目录 `testNN-M_名称.py`：学习进度脚本，`NN` 是学习阶段编号（01：数值微分与简单网络；02：TwoLayerNet 与数值梯度训练；03：计算图与误差反向传播），`M` 是阶段内序号。每个脚本自包含，用于演示/验证当节内容（print 输出或 matplotlib 绘图）。
- `common/`：书中公共库代码，`__init__.py` 统一再导出。
  - `functions.py`：激活函数 step/sigmoid/relu/identity、损失函数 MSE/交叉熵、含溢出对策且支持 batch 的 softmax、梯度演示用的 function_1/function_2。
  - `gradients.py`（文件名带 s）：numerical_diff、numerical_gradient（np.nditer 多维版）、gradient_descent。
  - `layer.py`：计算图与网络层。MulLayer/AddLayer 是书第 5 章的演示层（乘法节点/加法节点）；Affine、Relu、Sigmoid、SoftmaxWithLoss 是带 forward/backward 的网络层，梯度存在实例属性上（如 `Affine.dW`/`.db`）。
  - `net.py`：`TwoLayerNet` 类。权重存 `params` dict；网络层按前向顺序存 `OrderedDict` 的 `self.layers`，损失层单独放 `self.lastLayer`。求梯度有两条路径：`numerical_gradient()`（数值微分，每参数两次完整前向）与 `gradient()`（反向传播：先 loss 前向，再从 lastLayer 倒序 backward，最后从各 Affine 读 dW/db）。`input_size`/`output_size` 由数据和标签决定，`hidden_size` 是唯一可自由调整的超参数。
- `dataset/`：`mnist.py` 提供 `load_mnist(normalize, flatten, one_hot_label)`。原始 gz 文件在 `dataset/mnist/` 子目录（缺失时从 ossci-datasets S3 镜像下载），解析后缓存为 `mnist.pkl`；缓存已存在，`load_mnist()` 直接读 pickle，离线可用。

## 已知坑

- 本书代码基于旧版 numpy 编写，运行环境是 numpy 2.x，改写或新增书后代码时留意已废弃 API。
- `output_size` 不是提速旋钮：必须等于标签列数，否则 `cross_entropy_error` 里 `t * np.log(y + 1e-7)` 形状广播失败（详见 docs/01）。想缩小网络提速只动 `hidden_size` 和 `iters_num`。
- `numerical_gradient` 会临时**原地修改**传入的数组（扰动后还原）；模块级函数与 `TwoLayerNet.numerical_gradient` 方法同名，方法内部调用的是模块级函数，阅读时注意区分。
- `Relu.backward` 会原地置零 dout，`SoftmaxWithLoss.backward` 假定 t 是 one-hot（`self.y - self.t`）——都是书上的原始写法，组合或复用层时留意，不要当 bug "修" 掉。
- `common/` 子模块互相引用时不要 `from common import X`：包 `__init__.py` 按顺序导入各子模块，导入进行到一半时包处于部分初始化状态，会触发循环导入（整个 `import common` 直接失败）。子模块应写 `from common.layer import Affine` 这类完整模块路径。

## 文档约定

`docs/` 下的文档按"序号-名称"格式命名（如 `00-fix_mnist_path.md`、`01-two_layer_net_dims_and_speed.md`），序号从 00 递增。
文档为中文，写给人读（用户借此自行改代码学习），生成方案文档时不直接改动代码。

## 风格

代码注释与 docstring 均为中文（对齐书中译本），新增代码保持一致。
