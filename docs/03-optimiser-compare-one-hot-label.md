# 03 — test04-1 优化器比较脚本：交叉熵广播报错

日期：2026-10-04
涉及文件：`test04-1_optimiser-compare-minist.py`、`common/functions.py`、`common/layer.py`、`dataset/mnist.py`

## 现象

运行 `test04-1_optimiser-compare-minist.py`，第一次调用 `networks[key].gradient(x_batch, t_batch)` 即崩溃：

```
ValueError: operands could not be broadcast together with shapes (128,) (128,10)
```

调用链：

```
test04-1:45  networks[key].gradient(x_batch, t_batch)
  net.py:215   MultiLayerNet.gradient → self.loss(x, t)
    net.py:167   self.last_layer.forward(y, t) + weight_decay
      layer.py:62  SoftmaxWithLoss.forward → cross_entropy_error(self.y, self.t)
        functions.py:16  t * np.log(y + 1e-7)   ← 爆炸点
```

## 根因

一句话：**脚本按原书抄的是 `load_mnist(normalize=True)`，标签以整数形式返回（形状 `(batch,)`），而本仓库的 `cross_entropy_error` 是书早期简化版，只支持 one-hot 标签，于是 `(128,)` 与 `(128,10)` 相乘广播失败。**

分三步看：

### 1. 标签不是 one-hot

`dataset/mnist.py:96` 的默认签名是 `load_mnist(normalize=True, flatten=True, one_hot_label=False)`。
脚本 `test04-1_optimiser-compare-minist.py:14` 写的是：

```python
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
```

没传 `one_hot_label=True`，所以 `t_train` 是形状 `(60000,)` 的整数标签（0~9），minibatch 后 `t_batch` 形状为 `(128,)`。

### 2. 本仓库的 cross_entropy_error 只支持 one-hot

`common/functions.py:10`：

```python
def cross_entropy_error(y, t):
    if y.ndim == 1:               # 只处理“单样本”的情况
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size   # 要求 t 与 y 同形
```

这里 `y` 是 softmax 输出 `(128, 10)`，`t` 是 `(128,)`。`y.ndim == 1` 不成立，reshape 分支不触发；接着 `t * np.log(y + 1e-7)` 触发广播：NumPy 把 `(128,)` 右对齐成 `(1, 128)`，与 `(128, 10)` 逐维比较——最后一维 `128` 对 `10`，既不相等也没有一个是 1，广播失败。这就是报错信息里 `(128,) (128,10)` 的来历。

`SoftmaxWithLoss.backward`（`common/layer.py:65`）同样假定 one-hot（`dx = (self.y - self.t) / batch_size`），`(128,10) - (128,)` 在这里一样会炸。

### 3. 为什么原书这段脚本"能跑"

原书仓库（ch04 的 `optimizer_compare_mnist.py`）确实也是 `load_mnist(normalize=True)`，但原书**新版** `common/functions.py` 的 `cross_entropy_error` 支持两种标签形式：

```python
# 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
if t.size == y.size:
    t = t.argmax(axis=1)
batch_size = y.shape[0]
return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```

它用花式索引 `y[np.arange(batch_size), t]` 直接取正确类别的概率，one-hot 与整数标签都能吃。本仓库的 `functions.py` / `layer.py` 抄的是书**前半部分**的早期简化版，没有这段兼容逻辑。所以这不是抄错了书，而是"书的脚本"与"书的公共库"版本不配套——仓库既有脚本都通过 `one_hot_label=True` 规避了这一点（见 `test01-3`、`test02-2`、`test03-2`，全部显式传了 `one_hot_label=True`）。

## 修复方案

### 方案 A（推荐）：脚本加 `one_hot_label=True`

`test04-1_optimiser-compare-minist.py:14` 改为：

```python
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
```

一行改动，不动公共库，与仓库前三个阶段脚本的既有惯例完全一致。缺点：`accuracy()` 里 `t.ndim != 1` 时会自动对 one-hot 做 `argmax`（`net.py:172`），所以后续评估也不受影响；唯独打印的 loss 数值与原书运行结果完全可比，无副作用。

### 方案 B（备选）：把公共库升级成书后新版双形式实现

如果希望以后抄书不再处处补 `one_hot_label=True`，可以给公共库加上书后版本的兼容逻辑，但**必须两处一起改**，缺一不可：

1. `common/functions.py` 的 `cross_entropy_error`：加 `if t.size == y.size: t = t.argmax(axis=1)`，求和改为花式索引写法（见上文原书代码）。
2. `common/layer.py` 的 `SoftmaxWithLoss.backward`：加分支——

   ```python
   def backward(self, dout=1):
       batch_size = self.t.shape[0]
       if self.t.size == self.y.size:      # one-hot
           dx = (self.y - self.t) / batch_size
       else:                                # 整数标签
           dx = self.y.copy()
           dx[np.arange(batch_size), self.t] -= 1
           dx = dx / batch_size
       return dx
   ```

只改 1 不改 2 的话，loss 算出来了，backward 里 `(128,10) - (128,)` 会在下一行立刻报同样的广播错。

代价与注意点：

- 影响面是全部公共库使用者（test02/test03 系列脚本），需回归验证梯度检验（`test03-3`）仍通过——两版实现在 one-hot 输入下数学等价，理论上无误，但值得跑一遍确认。
- 注意 CLAUDE.md 的提醒：`SoftmaxWithLoss.backward` 的 one-hot 假定是"书上的原始写法"。改成双形式等于偏离当前抄录进度、提前引入第 5~6 章的实现，学习阶段不建议；等抄到书中出现新版实现时再顺手替换更自然。

## 结论

采用方案 A：在 `test04-1_optimiser-compare-minist.py:14` 补上 `one_hot_label=True`。改完直接重跑脚本，第一个 iteration 的四个优化器 loss 应在 2.3 附近（ln 10 ≈ 2.3026），随后逐步下降。

## 顺带记录

- `load_mnist` 的报错根源不在 `net.py` 新加的 `weight_decay`：`weight_decay_lambda` 默认 0，该项在 forward 里贡献 0、backward 里 `dW + λ·W` 也为 `dW`，与本错误无关。
- codegraph 索引在 `net.py` 刚改动后短暂滞后（报 traceback 行号 215 与索引内旧版 82 行文件不符），排查"刚改过的文件"时以磁盘内容为准。

---

# 追加分析（第二轮，2026-10-04）：方案 B 只抄了一半

## 现象

按方案 B 修改后重跑，报错与第一轮**逐字相同**——只是爆炸行号从 `functions.py:16` 变成了 `functions.py:20`：

```
File "common/functions.py", line 20, in cross_entropy_error
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
ValueError: operands could not be broadcast together with shapes (128,) (128,10)
```

## 现状代码（common/functions.py:10-20）

```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size   # ← 这一行没换
```

`argmax` 分支加了（第一步），但求和行还是旧版逐元素乘（第二步漏了）。

## 为什么加了 argmax 还是同样的错

1. **argmax 分支根本不负责当前这条路径。** 它是给 one-hot 输入预备的：触发条件 `t.size == y.size` 要求 t 和 y 同形（MNIST batch 下是 1280 == 1280）。当前脚本传进来的 t 已经是 `(128,)` 整数索引，128 ≠ 1280，条件不成立，t 原样穿过这个分支——然后落进**一字未改的旧求和行**，报出与第一轮完全相同的广播错。整个修改对这条输入路径来说是空操作。
2. **`t * np.log(y + 1e-7)` 这行只对"t 与 y 同形"的 one-hot 语义成立。** 一旦 t 是 `(batch,)` 的类别索引，逐元素乘在数学上没有意义，必须换成花式索引 `y[np.arange(batch_size), t]`——按样本号取"正确类别"那一列的概率。这正是方案 B 里 functions.py 部分的**两个不可分割的半步**：argmax 分支负责把 one-hot **归一成索引形式**，花式索引负责**按索引形式求值**。只做前者，所有路径最终都汇入"索引形式 + 逐元素乘"这个矛盾组合。

## 更糟：这次修改把原本能跑的路径也弄坏了

修改前 one-hot 输入是可以正常工作的（test01-3 / test02 / test03 全系列都靠它）。修改后 one-hot 输入 `(128,10)` 会触发 argmax 变成 `(128,)` 索引，随后同样在旧求和行炸掉。**实测确认**（同形状随机数据）：

```
当前实现连 one-hot 输入也炸: operands could not be broadcast together
with shapes (128,) (128,10)
```

即：修改前整数标签路径崩、one-hot 路径正常；修改后**两条路径全崩**——比不改更糟。

另有一个更阴险的隐雷值得记下：若 batch 恰好等于类别数（如 batch_size=10），`(10,) * (10,10)` 按 NumPy 广播规则**能"成功"**（`(10,)` 右对齐补成 `(1,10)` 再逐行复制），不报任何错，但逐元素乘的语义完全错乱，loss 是一个毫无意义的数。所以"不报错"不等于"修对了"，修完必须做数值校验（见下）。

## 最终正确写法（整体替换 functions.py:10-20）

```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```

与现状代码唯一的差别就是最后一行。`layer.py` 那半步（`SoftmaxWithLoss.backward` 双形式分支）**已经改对，不要再动**。

## 已验证（2026-10-04，本机实测）

把上面的最终函数 patch 进 `common.layer` 后，配合磁盘上现有（已改对的）`SoftmaxWithLoss`：

- `cross_entropy_error`：整数标签与 one-hot 两种输入算出的 loss 完全一致，且等于手算期望值 `-np.mean(np.log(y[np.arange(128), t] + 1e-7))`；
- `SoftmaxWithLoss.forward`：两种输入 loss 一致（`np.isclose` → True）；
- `SoftmaxWithLoss.backward`：两种输入输出形状均为 `(128, 10)`，与手算 `(y - onehot(t)) / batch_size` 逐元素比对，最大偏差 **0.0**。

验证方法提示：`layer.py` 里是 `from common.functions import cross_entropy_error` 按名字绑定，patch 必须打在 `common.layer.cross_entropy_error` 上才生效（模块属性在调用时查找）；给 `forward` 喂的数据必须是**网络原始 logits**——喂已归一化的概率会被内部 softmax 再归一一次，得到看似"backward 不对"的假阴性。

## 修完后的回归建议

1. `python3 test03-3_gradient-check.py`——梯度检验走 one-hot 路径，确认新求和行没把原有路径改坏（两版实现在 one-hot 输入下数学等价，应仍是 1e-10 量级误差）。
2. `python3 test04-1_optimiser-compare-minist.py`——第一个 iteration 四个优化器 loss 应约 2.3（ln 10 ≈ 2.3026），随后下降。

## 复盘

第一轮文档的方案 B 把 functions.py 的改动拆散在两处叙述（argmax 分支写在方案 B 正文、花式索引求和行写在"根因"第 3 小节引用的原书代码里），照抄时极易漏掉后半步。本次教训：**给"替换函数"类修复建议时，应一次性给出完整函数体**，而不是让人把分散的片段自己拼起来。
