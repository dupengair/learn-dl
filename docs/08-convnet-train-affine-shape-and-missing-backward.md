# 08 - SimpleConvNet 训练报错分析：Affine 形状不匹配与缺失的 backward

对应脚本：`test05-2_train-convnet.py`（书第 7 章 simple_convnet 训练部分）。

## 一、报错现象

```
File "common/net.py", line 436, in predict
    x = layer.forward(x)
File "common/layer.py", line 45, in forward
    out = np.dot(x, self.W) + self.b
ValueError: shapes (100,30,12,12) and (4320,100) not aligned: 12 (dim 3) != 4320 (dim 0)
```

发生在 `Trainer.train()` → `SimpleConvNet.gradient()` → `loss()` → `predict()` → `Affine1.forward`。

## 二、形状推演

按脚本参数 `input_dim=(1,28,28)`、`filter_num=30, filter_size=5, pad=0, stride=1`、`Pooling(2x2, stride=2)`、`hidden_size=100`，前向传播的形状：

| 层 | 输出形状 | 说明 |
|---|---|---|
| 输入 | (100, 1, 28, 28) | batch=100，flatten=False 的 4 维输入 |
| Conv1 | (100, 30, 24, 24) | 28-5+1=24 |
| Relu1 | (100, 30, 24, 24) | |
| Pool1 | (100, 30, 12, 12) | 2x2 池化减半 |
| **Affine1** | **报错** | `np.dot((100,30,12,12), (4320,100))` |
| （预期）Affine1 | (100, 100) | 需要先展平为 (100, 4320) |
| Relu2 / Affine2 | (100, 100) → (100, 10) | |

关键观察：`4320 = 30 × 12 × 12`，说明 `net.py` 里 `pool_output_size` 的计算和 W2 的形状**都是对的**，错的不是网络结构参数，而是进入 Affine1 的数据仍是 4 维张量，没有展平成 2 维 `(N, 4320)`。

## 三、根因：两代书的代码混用了

书第 7 章 `simple_convnet.py` 的层序列（`Conv1 → Relu1 → Pool1 → Affine1 → Relu2 → Affine2`）里确实**没有显式的 Flatten 层**，展平动作藏在当时配套的 `Affine` 层里。书仓库第 7 章用的 `Affine` 是这个版本（`deep-learning-from-scratch/common/layers.py`）：

```python
def forward(self, x):
    self.original_x_shape = x.shape      # 记录原形状
    x = x.reshape(x.shape[0], -1)        # 4维输入展平为2维 (N, D)
    self.x = x
    out = np.dot(self.x, self.W) + self.b
    return out
```

而本仓库 `common/layer.py` 的 `Affine`（layer.py:35-52）是第 3/4 章时代的纯 2 维版本，`forward` 直接 `np.dot(x, self.W)`，没有 `reshape`。`SimpleConvNet` 把书第 7 章的网络结构抄进来，却复用了这个旧版 `Affine`，两代代码混用，于是在第一次前向就撞上形状不匹配。

这不是孤立问题——同一个原因还埋着三个连锁问题，见下节。

## 四、连锁问题（修完报错后会按顺序撞上）

按修复第一个报错后的触发顺序排列：

### 问题 2：`Convolution` / `Pooling` 没有 backward

`common/layer.py` 里 `Convolution`（layer.py:206-225）和 `Pooling`（layer.py:227-248）**只实现了 forward**。`SimpleConvNet.gradient()`（net.py:443-465）倒序调用 `layer.backward(dout)`，dout 传到 `Pool1` 时就会抛出：

```
AttributeError: 'Pooling' object has no attribute 'backward'
```

而且 `grads['W1'] = self.layers['Conv1'].dW`（net.py:458）依赖 `Convolution.backward` 计算并缓存 `dW`/`db`。

### 问题 3：`common/utils.py` 缺 `col2im`

补 backward 的前置依赖：`Convolution.backward` 和 `Pooling.backward` 都需要 `col2im`（im2col 的逆变换）把 2 维列矩阵还原成 4 维图像。本仓库 `utils.py` 只有 `im2col`。

### 问题 4：`SimpleConvNet` 没有 `accuracy` 方法

`Trainer` 在每个 epoch 末尾（trainer.py:61-62）和训练结束（trainer.py:73）调用 `network.accuracy(...)`。第一个 epoch（600 iter）跑完的瞬间就会抛：

```
AttributeError: 'SimpleConvNet' object has no attribute 'accuracy'
```

### 问题 5：`SimpleConvNet` 没有 `save_params` / `load_params`

脚本第 29 行 `network.save_params("params.pkl")` 会抛 AttributeError。书中原版这两个方法用于保存/加载权重（加载时还要把参数重新绑定回各层实例，因为层里持有的是初始化时的数组引用）。

### 问题 6（可选但强烈建议）：`SimpleConvNet` 没有 `numerical_gradient`

书第 7 章原版有这个方法，配套 `gradient_check.py` 做梯度检验。CNN 的手写 backward（尤其 conv/pool 的列展开逆变换）非常容易出错，先检验再训练是书里的标准流程。

## 五、修复方案

全部改动集中在 `common/utils.py`、`common/layer.py`、`common/net.py` 三个文件，均为**对齐书第 7 章原版配套实现**，不是发明新写法。按文件列出。

### 5.1 `common/utils.py`：补 `col2im`

```python
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """im2col 的逆变换

    Parameters
    ----------
    col : 2维数组（im2col 的输出）
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    img : 还原后的4维数组
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w)\
             .transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            # 与 im2col 中的写入相反，这里用 += 累加（滤波器重叠区域的梯度要累加）
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
```

注意两点：
- 用 `+=` 而不是 im2col 里的赋值——同一输入像素被多个卷积窗口覆盖时，梯度要累加。
- `img` 的尺寸多加了 `stride - 1` 的余量，最后切片裁掉，避免步幅不能整除时越界。

写完后记得在 `common/__init__.py` 的 utils 导入块和 `__all__` 里加上 `col2im`。

### 5.2 `common/layer.py`：升级 `Affine`

替换现有 `Affine` 类（本仓库唯一会受影响的调用方是 `TwoLayerNet`/`MultiLayerNet*`，它们只传 2 维输入，`reshape(N, -1)` 和 `reshape(*original_x_shape)` 对 2 维都是恒等操作，**向后兼容**）：

```python
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx
```

`backward` 末尾的 `reshape` 不能省：`Affine1.backward` 返回的 dx 要交给 `Pool1.backward`，后者按 4 维 `(N, C, H, W)` 处理。

### 5.3 `common/layer.py`：`Convolution` 补缓存与 backward

```python
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中间数据（backward时使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 权重参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T  # 滤波器的展开
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
```

### 5.4 `common/layer.py`：`Pooling` 补缓存与 backward

```python
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 展开(1)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        # 最大值(2)
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        # 转换(3)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        # 只把梯度传给 forward 时取过最大值的那个元素，其余置0
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
```

### 5.5 `common/net.py`：`SimpleConvNet` 补三个方法

`net.py` 顶部需要加 `import pickle`（`save_params`/`load_params` 用）。

```python
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        # 层实例里持有的是初始化时的数组引用，必须重新绑定
        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
```

（可选，用于梯度检验）：

```python
    def numerical_gradient(self, X, T):
        loss_W = lambda W: self.loss(X, T)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads
```

## 六、验证建议

### 6.1 先梯度检验，再训练

仿照 `test03-3_gradient-check.py` 写一个 CNN 版（或临时脚本）：

```python
from common import SimpleConvNet, numerical_gradient
import numpy as np

network = SimpleConvNet(input_dim=(1, 10, 10),
                        conv_param={'filter_num': 2, 'filter_size': 3,
                                    'pad': 1, 'stride': 1},
                        hidden_size=5, output_size=3,
                        weight_init_std=0.01)

x = np.random.rand(2, 1, 10, 10)
t = np.array([[1,0,0], [0,1,0]])  # one-hot

grad_backprop = network.gradient(x, t)
grad_num = network.numerical_gradient(x, t)

for key in grad_backprop.keys():
    diff = np.mean(np.abs(grad_backprop[key] - grad_num[key]))
    print(key + ":" + str(diff))
# 各参数平均绝对误差应在 1e-10 ~ 1e-8 量级，明显偏大（如 1e-2）说明 backward 写错了
```

务必用小网络（小 input_dim、少量滤波器）：`numerical_gradient` 对每个参数要做两次完整前向，卷积参数 W1 有几千个元素，全尺寸网络会算到天荒地老。

### 6.2 先小数据集试跑

训练通过梯度检验后，先把脚本第 13-14 行的子集注释解开（5000 张训练 + 1 epoch 左右）确认整条流水线能走通、准确率明显爬升，再放开全量 20 epochs。全量 12000 次迭代、每次迭代含 im2col 展开和两次 (57600×25) 级别的矩阵乘，纯 CPU 上预计要跑一两个小时。

## 七、附注

1. **与"不要把书上原始写法当 bug 修"的约定不冲突**：`CLAUDE.md` 提醒的是不要乱改书中有意为之的写法（如 `Relu.backward` 原地置零）。本次改动是把本仓库的 `Affine`/`Convolution`/`Pooling` **对齐到书第 7 章的配套版本**——书中 `simple_convnet.py` 从一开始就是搭配带 reshape 的 `Affine` 和带 backward 的 conv/pool 层使用的，本仓库此前只抄到了第 6 章对应的层实现。
2. 优化器 `Adam` 用的是 `params[key] -= ...` 原地更新（optimiser.py:92），`SimpleConvNet.layers` 各层持有的 `W`/`b` 引用与 `self.params` 是同一数组，训练中会同步更新，无引用断开问题，不需要额外处理。
3. `numpy 2.x` 兼容性：`col2im`/`im2col`/backward 里用到的 `np.pad`、切片步长赋值、`transpose`、`reshape` 均无废弃问题。
4. 已有的坑依然适用：`Relu.backward` 原地置零 dout。本 CNN 的链路上各 backward 传入的 dout 都是独立新数组（`SoftmaxWithLoss.backward` 的 `y - t`、`Affine.backward` 的 `np.dot` 结果、`col2im` 的返回值），不会互相踩踏，无需处理。
