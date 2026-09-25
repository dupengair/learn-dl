# TwoLayerNet 维度报错分析 与 网络改小提速方案

> 现象：`test01-6_neuralnet-base.py` 中把网络改为
> `TwoLayerNet(input_size=784, hidden_size=10, output_size=2)` 后，
> `net.numerical_gradient(x, t)` 抛出
> `ValueError: operands could not be broadcast together with shapes (100,10) (100,2)`。
> 本文分析根因，并回答"想把网络改小、让 CPU 跑得更快应该怎么改"。

## 一、怎么读这份 traceback

Python 的 traceback **从最后一行往上读**，最快定位到出错点：

```
ValueError: operands could not be broadcast together with shapes (100,10) (100,2)   ← 错误本身
  File common/functions.py, line 16, in cross_entropy_error
      return -np.sum(t * np.log(y + 1e-7))          ← 出错的表达式：t 和 y 逐元素相乘
  File common/two_layer_net.py, line 35, in loss     ← y 来自 predict(x)
  File common/two_layer_net.py, line 47, in <lambda> ← loss_W = lambda W: self.loss(x, t)
  File common/gradients.py, line 37, in numerical_gradient
      fxh1 = f(x)                                    ← 数值微分在逐参数调用 loss
  File test01-6_neuralnet-base.py, line 20
      grads = net.numerical_gradient(x, t)           ← 入口
```

结论已经写在错误信息里：**`t` 的形状是 (100, 10)，`y` 的形状是 (100, 2)**，
两者做逐元素乘法 `t * np.log(y + 1e-7)` 时对不上。

## 二、根因分析

### 2.1 网络内部的维度链路

`TwoLayerNet.__init__` 中各参数的形状完全由三个 size 决定：

```python
W1: (input_size, hidden_size)   # (784, 10)
b1: (hidden_size,)              # (10,)
W2: (hidden_size, output_size)  # (10, 2)   ← output_size 被你改成了 2
b2: (output_size,)              # (2,)
```

`predict` 的前向计算中，形状是这样流动的（batch_size = 100）：

```
x: (100, 784)
  → np.dot(x, W1) + b1 → a1: (100, hidden_size)  = (100, 10)
  → sigmoid(a1)        → z1: (100, 10)
  → np.dot(z1, W2) + b2 → a2: (100, output_size) = (100, 2)
  → softmax(a2)        → y:  (100, 2)            ← 网络输出 2 类的概率
```

注意：**`predict` 全程不报错**。`softmax` 对 2 列输入照样正常工作。
报错发生在下一步 `cross_entropy_error(y, t)`。

### 2.2 交叉熵要求 y 和 t 逐元素对齐

`cross_entropy_error` 的核心计算（functions.py 第 16 行）：

```python
return -np.sum(t * np.log(y + 1e-7)) / batch_size
```

数学上是 −Σ tₖ·log(yₖ)：`t` 和 `y` 必须**形状完全相同**，逐元素相乘。
`t[i][k]` 表示"样本 i 的真值是否是类别 k"，`y[i][k]` 表示"模型认为样本 i
是类别 k 的概率"——两者必须在同一个类别轴上对齐。

NumPy 广播规则从**尾部维度**对齐：`(100,10)` 与 `(100,2)` 的末维
`10 ≠ 2` 且都不为 1，无法广播 → 直接抛 ValueError。

### 2.3 三个 size 各自由谁决定（本次报错的本质）

| 参数 | 值 | 由谁决定 | 能否随意改 |
|---|---|---|---|
| `input_size` | 784 | 输入数据维度（28×28 展平） | ❌ 改了 `np.dot(x, W1)` 就崩 |
| `output_size` | 10 | **标签的类别数**（MNIST 10 类） | ❌ **必须等于 t 的列数** |
| `hidden_size` | 100 | 你自己（网络容量超参数） | ✅ 唯一自由参数 |

本脚本第 18 行 `t = np.random.rand(100, 10)` 是 **10 列**伪标签，
而 `output_size=2` 让网络只输出 2 类 → (100,10) 对 (100,2)，必崩。

**一句话：`output_size` 不是"想改小就改小"的提速旋钮，它由任务/标签决定；
想缩小网络，唯一该动的是 `hidden_size`。**

## 三、修复方案

只改一处——把 `output_size` 恢复为 10，`hidden_size=10` 保留（这正是"改小"
的正确姿势）：

```python
# test01-6_neuralnet-base.py 第 6 行
# ---- 修改前 ----
net = TwoLayerNet(input_size=784, hidden_size=10, output_size=2)
# ---- 修改后 ----
net = TwoLayerNet(input_size=784, hidden_size=10, output_size=10)
```

同时**同步更新脚本里的形状注释**（它们还写着书上的 hidden=100 版本，
改完参数后实际输出会和注释对不上，容易误导自己）：

```python
# ---- 修改前（第 7-10 行）----          # ---- 修改后 ----
print(net.params['W1'].shape) # (784, 100)   →  # (784, 10)
print(net.params['b1'].shape) # (100,)        →  # (10,)
print(net.params['W2'].shape) # (100, 10)     →  # (10, 10)
print(net.params['b2'].shape) # (10,)         →  # (10,)
# 第 22-25 行 grads 的四处注释同理更新
```

> 顺带发现一处小笔误：`common/two_layer_net.py` 第 11 行注释
> `# 初始化权重pyt` 末尾多了 "pyt"，可顺手改成 `# 初始化权重`。

## 四、想把网络改小、CPU 跑更快——完整分析

### 4.1 先算清楚：耗时到底花在哪

数值微分（`common/gradients.py` 的 `numerical_gradient`）用 `np.nditer`
**逐个标量参数**扰动：每个参数算 `f(x+h)` 和 `f(x−h)` 两次 loss，
而每次 loss 就是一次**整个 batch 的前向传播**。

参数总量（`input_size=784, output_size=10` 时）：

```
P = 784·H + H + H·10 + 10 ≈ 795·H + 10
```

| hidden_size H | 参数量 P | 一次梯度的前向次数 ≈ 2P | 相对耗时（数值梯度） |
|---:|---:|---:|---:|
| 100（书上） | 79,510 | ~159,020 | 1×（很慢，分钟级） |
| 50 | 39,760 | ~79,520 | ~1/4 |
| 20 | 15,910 | ~31,820 | ~1/25 |
| **10** | **7,960** | **~15,920** | **~1/100（秒级）** |

关键观察——**k² 效应**：把 H 缩小 k 倍，参数量（∝H）缩小 k 倍，
单次前向的计算量（100×784×H + 100×H×10，也 ∝H）再缩小 k 倍，
总计算量缩小 **k²** 倍。H: 100→10 即约 **100 倍**提速。

（注：书 4.5.2 原例 H=100 在纯 CPU 上本来就要跑几分钟，这正是书里说
数值微分"慢到不适合真正训练"的原因，见下文 4.4。）

### 4.2 batch_size 是第二个提速旋钮

脚本第 13/17 行 `x = np.random.rand(100, 784)` 的 100 是 batch 大小。
数值梯度下每个参数的两次前向都要处理**整个 batch**，所以 batch 缩小 k 倍，
总耗时也近似缩小 k 倍：

```python
# 伪数据只是验证流程，batch 可以随意缩小
x = np.random.rand(10, 784)   # 100 → 10
t = np.random.rand(10, 10)    # 列数 10 永远 = output_size，不能动
```

注意 `x` 的列数 784（= input_size）和 `t` 的列数 10（= output_size）
是**维度约束**，不能随"改小"而动——能改小的只有行数（样本数）。

### 4.3 推荐组合（test01-6 验证场景）

```python
net = TwoLayerNet(input_size=784, hidden_size=10, output_size=10)
x = np.random.rand(10, 784)
t = np.random.rand(10, 10)
```

相比书上原版（H=100、batch=100）约快 100×10 = 1000 倍，几秒内出结果。

### 4.4 更大的图景：真正的训练要靠反向传播（预告第 5 章）

数值微分对**每个参数**都要跑两次完整前向，参数一多必然慢。
第 5 章的误差反向传播法：一次前向 + 一次反向就得到**全部**参数的梯度，
复杂度从"2P 次前向"降为"1 次前向+反向"，快几个数量级——
那才是 CPU（以及 GPU）上训练网络的正确姿势。
等学到第 5 章，`TwoLayerNet` 会有一个基于 layers 的 `gradient()` 方法，
训练脚本里把 `net.numerical_gradient(x_batch, t_batch)` 换成
`net.gradient(x_batch, t_batch)` 即可，接口不变。

另外，到了真正训练时（后续 minibatch 脚本），还有这些常规提速/降耗手段：
- 减小 `iters_num`（迭代次数）、加大 `learning_rate` 配合调参；
- 只用训练集的一个子集先跑通流程；
- batch_size 在反向传播下才是真正的吞吐旋钮（太大占内存、太小梯度噪声大）。

## 五、修改清单（自己动手）

| # | 文件:行 | 改动 | 性质 |
|---|---|---|---|
| 1 | test01-6: 6 | `output_size=2` → `output_size=10` | **必修（修 bug）** |
| 2 | test01-6: 7-10, 22-25 | 注释形状同步为 (784,10)/(10,)/(10,10)/(10,) | 建议 |
| 3 | test01-6: 13, 17, 18 | batch 100 → 10（三处 rand 的行数） | 可选（提速） |
| 4 | two_layer_net.py: 11 | 注释笔误 `初始化权重pyt` → `初始化权重` | 可选 |

hidden_size=10 保留不动——它已经是缩小后的值。

## 六、验证

```bash
python3 test01-6_neuralnet-base.py
```

预期输出（改完 1、2 后；几秒内结束）：

```
create model:
(784, 10)
(10,)
(10, 10)
(10,)
predict:
numerical_gradient:
(784, 10)
(10,)
(10, 10)
(10,)
```

## 七、留给自己的思考题

1. 为什么书上用 `np.random.rand` 生成的假数据也能"算出梯度"？
   （提示：交叉熵和数值微分只要求**形状对齐**，数值无意义也能跑通流程；
   这类脚本验证的是实现正确性，不是训练效果。）
2. 如果真想做 2 分类（output_size=2），`t` 应该长什么样？
   （提示：t 的列数必须跟着变 2；MNIST 上等价于只取 0/1 两类样本做子任务。）
3. `numerical_gradient` 里 `x[idx] = tmp_val` 这行"还原值"删掉会怎样？
   （提示：扰动会累积到后续参数的 f(x+h)/f(x−h) 里，梯度全错。）
