# 05 · test04-4 BatchNorm 实验：`W**2` overflow 与 softmax nan 的原因与修复

对应脚本：`test04-4_batch-norm.py`（书上 6.4.3 节 Batch Norm 实验，图 6-19 的复现）。

## 一、现象

运行时报出三类警告：

```text
common/net.py:327: RuntimeWarning: overflow encountered in square
  weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
common/net.py:327: RuntimeWarning: invalid value encountered in scalar multiply
  weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
test04-4_batch-norm.py:85: UserWarning: No artists with labels found to put in legend.
common/functions.py:35: RuntimeWarning: invalid value encountered in subtract
  x = x - np.max(x, axis=0)
```

同时训练日志里 accuracy 仍打印出 `0.117 - 0.09` 这类正常数字，看起来"没崩"，警告却刷个不停。

## 二、结论（一句话）

**不是 BatchNorm 实现有 bug**：是实验里 `np.logspace(0, -4, num=16)` 的**大初始值端（w=1.0、0.54）让对照组（无 BN 的普通网络）梯度爆炸**，权重在 ~20 次迭代内冲破 float64 上限（>1e154），nan/inf 随前向传播污染 softmax。三条警告都是这一件事的下游症状；legend 警告则是另一处独立的小问题。

实测验证（本仓库代码原样运行）：

| 组 | BN 网络 | 普通网络（无 BN） |
|---|---|---|
| 1/16 w=1.0 | 稳定收敛 | **第 0 个 epoch 内爆成 nan** |
| 2/16 w≈0.54 | 稳定收敛 | **第 0 个 epoch 内爆成 nan** |
| 3/16 ~ 16/16（w≤0.29） | 全部稳定 | 稳定（大部分学不动，停在 0.1 附近） |

普通网络 w=1.0 时的爆炸速度（同一个 batch 连续更新）：

| 迭代 i | max\|W\| | max\|梯度\| | 最后层输出 max\|logit\| |
|---|---|---|---|
| 0 | 4.3 | 4.7e4 | 6.1e5 |
| 1 | 4.7e2 | 8.8e11 | 8.3e14 |
| 2 | 8.8e9 | 2.7e18 | 4.3e22 |
| 4 | 8.2e18 | 1.5e40 | 7.5e51 |
| 5 | 1.5e38 | 2.8e54 | 5.2e87 |

每迭代权重放大约 100 倍，再过十几次迭代就会超过 float64 上限 ~1.8e308（平方溢出阈值约 1.3e154）。

## 三、三条警告的因果链

权重爆炸后，警告按下面顺序产生：

1. **`overflow encountered in square`**：`np.sum(W**2)` 中 `W**2` 超出 float64 上限，结果为 `inf`。
2. **`invalid value encountered in scalar multiply`**：本实验 `weight_decay_lambda=0`，于是 `0.5 * 0 * inf = nan`（0 乘无穷是未定义运算）。这一条纯粹是上一条的衍生。
3. **`invalid value encountered in subtract`**（`functions.py:35`）：普通网络的前向输出 logits 已含 ±inf/nan，softmax 的溢出对策 `x - np.max(x, axis=0)` 中出现 `inf - inf = nan`。
4. **为什么日志里 accuracy 还"正常"**：`accuracy()` 用 `np.argmax`，对含 nan 的输出也能返回某个类别索引，于是照常打印出 ~0.1 的准确率，训练循环对发散毫无感知，一直跑到 20 epoch 结束。

5. **legend 警告与上面无关**：脚本只给最后一个子图（`i == 15`）的两条曲线设了 `label`，但 `plt.legend(loc='lower right')` 对全部 16 个子图都调用了，前 15 个子图自然找不到带标签的曲线。

## 四、根因：为什么普通网络在大 w 端必爆、BN 网络却稳

- 本实验 `MultiLayerNetExtend` 未传 `activation`，默认 **ReLU**（书上也如此）。ReLU 无上界，w=1.0 时 784→100→…→10 的前向逐层放大约 10 倍/层，初始 logits 就达 1e5~1e6 量级。
- 反向传播的梯度也逐层放大（softmax 交叉熵回传后经 Affine `dW = x.T·dout`，`x` 本身就是 1e5 量级的激活），一次 `SGD(lr=0.01)` 更新就把 W 推大两个数量级；W 变大 → logits 更大 → 梯度更大，形成**平方级正反馈**（上表）。
- **BN 网络没事**，恰好印证了实验主题：Batch Norm 把每层激活强行拉回均值 0、方差 1，前向的复合放大被掐断，logits 始终是 O(1)，梯度也是 O(1e-3)，正反馈回路不存在。实测 BN 网络 16 组初始值（含 w=1.0 连跑 600 iter）全部稳定。
- 书上的原代码同样存在这个问题——这是实验设定（大 w 端 + ReLU 对照组）的固有现象，是否撞上取决于随机初始化/采样序列；社区复现该书代码时报告 nan 的帖子很多。图 6-19 上大 w 端普通网络那条"停在 0.1 的平线"，实际上往往就是爆掉后的 `argmax(nan)`。

## 五、修复方案（供自行改代码，本文档不改动代码）

### 方案 A（推荐）：训练循环加发散防护

发散本身就是这个实验的结论之一——"这组初始值训不动"。给每个网络加一个存活标记，权重一旦出现非有限值就冻结它，避免后续白跑 20 epoch、刷满警告：

```python
    networks = (bn_network, network)
    alive = [True, True]

    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for j, _network in enumerate(networks):
            if not alive[j]:
                continue
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)
            # 发散防护：任一权重出现 nan/inf 即冻结该网络
            if not all(np.all(np.isfinite(w)) for k, w in _network.params.items()
                       if k.startswith('W')):
                alive[j] = False
                print(f"epoch:{epoch_cnt} 一个网络发散，已冻结")
```

效果：曲线停在发散前的准确率（画图用 `plt.ylim(0, 1.0)` 不受影响），终端不再刷警告，每组实验照常在 20 epoch 结束。

### 方案 B：收窄初始值扫描范围

把 `weight_scale_list = np.logspace(0, -4, num=16)` 改成避开病态大 w 端，例如 `np.logspace(-2, -4, num=16)`（实测 w≤0.29 时普通网络稳定）。缺点：偏离书上的实验设定，"普通初始化在大 w 端训不动"这一对照信息就丢了，不如方案 A。

### 不推荐的做法

- **降低学习率或换 Adam**：能让普通网络在 w=1.0 端不爆，但改变了书上的实验条件，对照组就失去"朴素初始化有多糟"的说明力。
- **给 weight_decay_lambda 设非零值**：L2 正则确实能压住权重幅值，但同样改变了实验语义（该实验本就设 λ=0）。
- **忽略警告**：`np.errstate` 压警告只是眼不见，nan 照样在算，浪费 20 epoch 的算力。

### legend 警告的修复

把 legend 挪进有 label 的分支即可：

```python
    if i == 15:
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", label='Normal(without BatchNorm)', markevery=2)
        plt.legend(loc='lower right')
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)
```

### 不修复也是合理的选择

分析后的决定是**保持脚本原样**：发散本身就是这个实验对照结论的一部分——"朴素初始化在极端值下训不动"，普通网络在大 w 端不收敛，正好与 BN 网络的快速收敛形成对照。代价只有两点：终端刷满警告、爆掉后的迭代照跑（白算一些 CPU 时间），都可接受。

但讲解这张图时要分清两端失败机制的**不同**：

| | 大 w 端（w=1.0、0.54） | 小 w 端（w≤0.29） |
|---|---|---|
| 失败机制 | **梯度爆炸**，权重爆成 nan/inf | **梯度消失**：激活/梯度太小，学不动但**不爆** |
| 曲线表现 | 停在 ~0.1 的平线是**假象**：权重已是 nan，0.1 是 `np.argmax` 对 nan 输出凑出的垃圾值 | 真正的"学不动但没爆" |
| BN 网络 | 免疫（归一化掐断正反馈，16 组实测全稳） | 同样稳定 |

所以准确的说法是"大 w 端普通网络**直接爆炸**、小 w 端**学不动**"，而不是笼统的"普通网络停在 0.1"。

## 六、梯度爆炸是如何诊断出来的（复盘）

三步，从"读警告"到"亲眼看着它滚雪球"：

**第一步：警告本身就泄露了量级。** `overflow encountered in square` 出现在 `W**2` 上——float64 上限约 1.8e308，平方要溢出意味着 |W| > 1.3e154。正常训练的权重是 0.01~10 量级，要到 1e154 只有指数增长才可能在几十步内到达。第一眼就能断定：权重已经指数爆炸，nan 只是下游污染，不是 BatchNorm 算错。

**第二步：定位是哪个网络、哪组参数。** 诊断脚本把 16 组 w 都扫一遍，每个网络每步检查一次权重：

```python
wmax = max(np.abs(w).max() for k, w in net.params.items() if k.startswith('W'))
if not np.isfinite(wmax) or wmax > 1e100:   # 1e100 是提前报警线，不必等到溢出
    print(f"发散: w={w} i={i} max|W|={wmax:.3g}")
```

结果只有普通网络在 w=1.0、0.54 两组触发，BN 网络 16 组全稳。

**第三步：打印增长曲线，看清机制。** 每步记录 max|W|、max|梯度|、max|logit|（最后层输出）：

```python
gmax = max(np.abs(g).max() for g in grads.values())
print(f"i={i} max|W|={wmax:.3g} max|grad|={gmax:.3g}")
```

就得到第二节那张表：`4.3 → 4.7e2 → 8.8e9 → 1.5e38`，每迭代 ×100，且 logits 从 6e5 涨到 5e87。**"每步平方级放大"就是正反馈式梯度爆炸的指纹**：W 变大 → logits 更大 → 梯度更大 → W 更大。

### 以后排查梯度爆炸的信号清单

| 信号 | 说明 |
|---|---|
| `overflow` / `invalid value` 警告 | 出现在 `square`/`dot`/`subtract` 等处，警告位置提示谁先爆 |
| loss 变 nan 或陡增震荡 | 最直接的训练层信号 |
| max\|W\|、max\|grad\| 逐迭代暴涨 | 上面几行代码就能观测 |
| 准确率掉到 1/类别数并停住 | 可疑但**不充分**：本例的 0.1 就是 nan 凑的，要配合前面的信号确认 |

## 七、附注

- `layer.py` 里 `np.sqrt(var + 10e-7)` 的 `10e-7` 实为 1e-6（沿袭书本原文的写法），只是 epsilon 偏大一点点，与本问题无关，不必"修"。
- `CLAUDE.md` 已记录的坑在此不适用但值得重申：本实验的网络是 ReLU（`activation` 默认值），不是sigmoid——排查时容易误以为是 sigmoid 饱和的问题，方向就错了。
