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

---

# 追加分析（第三轮，2026-10-05）：新增 RMSprop 后绘图 KeyError

## 现象

在 `test04-1` 中解禁 `optimizers['RMSprop'] = RMSprop()`（第 27 行，书上原本的注释行），自己实现了 `RMSprop` 类，训练全部跑完，最后在绘图处崩溃：

```
File "test04-1_optimiser-compare-minist.py", line 62, in <module>
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], ...)
                                                     ~~~~~~~^^^^^
KeyError: 'RMSprop'
```

## 排查：整条新增链路都是通的

| 环节 | 位置 | 状态 |
|---|---|---|
| RMSprop 实现 | `common/optimiser.py:44`（EMA 形式，与原书 6.1.4 一致） | ✓ 正确 |
| 包导出 | `common/__init__.py:37`（导入）、`:80`（`__all__`） | ✓ |
| 脚本导入 | `test04-1:10` `from common import ..., RMSprop` | ✓ |
| 优化器注册 | `test04-1:27` | ✓ |
| 训练循环 | 5 个优化器 × 2000 迭代**完整跑完**，无报错 | ✓ |
| 绘图样式 | `test04-1:59` `markers` 字典只有 4 项 | ✗ **唯一没跟上的一处** |

训练能完整跑完同时说明：第二轮的 `cross_entropy_error` 修复已生效（整数标签路径已通）。

## 根因：手工平行字典漏同步

`markers` 是一个**按优化器名字手工枚举**的绘图样式映射：

```python
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
```

脚本里的另外三个字典——`optimizers`、`networks`、`train_loss`——都是以 `optimizers` 为单一事实源、由 `for key in optimizers.keys()` 循环自动生成的，新增优化器自动跟上；**唯独 `markers` 需要手工同步**。新增第五个优化器时其余四个字典全部自动扩容，只有它忘了加条目，`markers['RMSprop']` 就 KeyError。

为什么原书脚本没有这个坑：原书 `common/optimizer.py` **根本没有 RMSprop 类**（6.1.4 节只讲了原理，没给实现），脚本里那行 `#optimizers['RMSprop'] = RMSprop()` 是书上留给读者的练习，注释状态恰好与 4 项 markers 对齐。解禁它意味着同时要做三件事：实现类、导出导入、同步 markers——前两件不做脚本跑不起来（会立刻报 ImportError/NameError），第三件不做则**训练全程跑完才在最后一行崩掉**，2000 次迭代白跑。这就是"静默滞后"类错误的典型形态：滞后点离工作点越远，浪费越大。

## 修复方案

### 方案 A（最小）：markers 补一项

`test04-1:59` 改为：

```python
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D", "RMSprop": "^"}
```

marker 样式任选，与已有的 o（圆）/ x（叉）/ s（方）/ D（菱）区分度好即可；常用备选：`^`（上三角）、`v`（下三角）、`P`（填充加号）、`*`（星号）、`p`（五边形）。注意小写 `x` 是线形叉、大写 `X` 才是填充叉，别混用。

### 方案 B（根治）：自动分配 marker，消除手工同步点

```python
marker_list = ["o", "x", "s", "D", "^", "v", "<", ">", "P", "*"]
x = np.arange(max_iterations)
for i, key in enumerate(optimizers.keys()):
    plt.plot(x, smooth_curve(train_loss[key]),
             marker=marker_list[i % len(marker_list)],
             markevery=100, label=key)
```

以后再加第 6、7 个优化器（如书 6 章之后的对比实验）都不会再炸；代价是与原书代码形态略有偏离。学习阶段建议至少用方案 A 改通，理解原因后可换方案 B。

## 顺带点评：RMSprop 实现本身

`common/optimiser.py:44-62` 的实现是对的——对比 `AdaGrad` 只差两点，正是 RMSprop 的核心思想：

- AdaGrad：`h += g²`（无衰减累加）→ 分母单调增大 → 学习率迟早衰减到接近 0，训练后期走不动；
- RMSprop：`h = 0.99·h + 0.01·g²`（指数移动平均）→ 分母反映**近期**梯度尺度 → 学习率保持自适应但不枯竭。

`lr=0.01, decay_rate=0.99` 是该实现的常规起点，与本脚本中 AdaGrad 的默认 lr 相同，对比是公平的。预期曲线：应明显快于 SGD，与 AdaGrad/Adam 同档（具体相对位置受批次随机性影响，不必强求书图复现）。

## 复盘

三轮报错其实是一条共同模式的三次显形：**"书上脚本"与"本仓库公共库/现场改动"之间的隐式契约被打破**——第一轮是损失函数版本不配套，第二轮是修复只落地一半，第三轮是新增项漏了手工同步的平行字典。排查时先确认"链路上每一环是否都在"（本轮的表格排查法），再找"哪一环是手工维护的"（它就是最可能的滞后点）。
