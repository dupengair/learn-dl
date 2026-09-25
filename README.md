# 深度学习入门：基于 Python 的理论与实现 —— 学习仓库

按《深度学习入门：基于Python的理论与实现》（斋藤康毅，"鱼书"）逐章手写/抄录书中代码的学习仓库。每个脚本自包含、可独立运行，用于演示/验证当节内容。

## 环境

- Python 3.12
- numpy 2.x（注意：书基于旧版 numpy 编写，部分 API 已废弃）
- matplotlib

```bash
pip install numpy matplotlib
```

## 运行方式

从仓库根目录直接运行（无需配置 PYTHONPATH）：

```bash
python3 test01-3_mnist.py
```

注意：

- 带 `plt.show()` 的脚本（test01-1、test01-4、test01-8）会阻塞等待关闭窗口。
- 训练脚本（test01-7、test01-8）用**数值微分**计算梯度：每个参数正反扰动各跑一次完整前向传播，CPU 上非常慢。书中高速版 `gradient()`（误差反向传播法）尚未实现。

## 目录结构

```
.
├── test01-N_*.py          # 学习进度脚本（01 为学习阶段编号）
├── common/                # 书中公共库
│   ├── functions.py       # 激活函数 step/sigmoid/relu、损失函数 MSE/交叉熵、softmax
│   ├── gradients.py       # numerical_diff / numerical_gradient / gradient_descent
│   └── two_layer_net.py   # TwoLayerNet 两层全连接神经网络
├── dataset/
│   ├── mnist.py           # load_mnist()：下载、解析、缓存 MNIST
│   ├── mnist/             # 原始 .gz（不入库，脚本会自动下载）
│   └── lena.png           # 书中图像示例用图
└── docs/                  # 学习笔记 / 问题分析文档
```

## 学习进度

| 脚本 | 内容 |
|---|---|
| test01-1_func1.py | 绘制 f(x)=0.01x²+0.1x（数值微分示例函数） |
| test01-2_simplenet.py | simpleNet 类演示数值梯度求法 |
| test01-3_mnist.py | 加载 MNIST 并打印数据形状 |
| test01-4_stepfunc.py | 绘制阶跃函数 |
| test01-5_network.py | 手写三层网络前向传播（sigmoid + 恒等函数） |
| test01-6_neuralnet-base.py | TwoLayerNet：确认参数与梯度的形状 |
| test01-7_neuralnet-minibatch.py | mini-batch 训练（数值微分梯度），观察 loss 下降 |
| test01-8_neuralnet-eval.py | 训练 + 每轮评估训练/测试识别精度并绘图 |

## MNIST 数据说明

`dataset/mnist.py` 提供 `load_mnist(normalize, flatten, one_hot_label)`：

- 首次调用时若 `dataset/mnist.pkl` 缓存不存在，自动从 ossci-datasets S3 镜像下载 4 个 `.gz` 到 `dataset/mnist/`，解析后生成 `mnist.pkl`；
- 之后直接读缓存，离线可用。

## 学习笔记

- [docs/00-fix_mnist_path.md](docs/00-fix_mnist_path.md) —— MNIST 数据集路径问题修复方案
- [docs/01-two_layer_net_dims_and_speed.md](docs/01-two_layer_net_dims_and_speed.md) —— TwoLayerNet 维度报错分析与网络改小提速方案
