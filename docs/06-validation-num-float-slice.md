# 06 - test04-7 验证集切片报错：validation_num 是浮点数

## 现象

运行 `test04-7_hyperparameter_optimization.py` 报错：

```
Traceback (most recent call last):
  File "test04-7_hyperparameter_optimization.py", line 21, in <module>
    x_val = x_train[:validation_num]
            ~~~~~~~^^^^^^^^^^^^^^^^^
TypeError: slice indices must be integers or None or have an __index__ method
```

## 原因分析

问题出在脚本第 19 行：

```python
validation_rate = 0.20
validation_num = x_train.shape[0] * validation_rate   # ← 少了 int()
```

`x_train.shape[0]` 是 Python `int`（此处为 500），`validation_rate` 是 `float`（0.20），两者相乘得到 **`float` 类型的 100.0**。而 Python 的切片协议要求索引实现 `__index__` 方法（即整型），`x_train[:100.0]` 因此直接抛出 `TypeError`。

两点说明：

- 这**不是 numpy 2.x 的兼容性问题**，而是纯 Python 语义：用浮点数切片任何序列都会失败，与 numpy 版本无关。最小复现：
  ```python
  >>> import numpy as np
  >>> a = np.zeros((500, 784))
  >>> a[:500 * 0.20]
  TypeError: slice indices must be integers or None or have an __index__ method
  ```
- 对照书本原版代码（第 6 章 `hyperparameter_optimization.py`），该行是 `validation_num = int(x_train.shape[0] * validation_rate)`，抄录时漏掉了 `int()`。

第 21–24 行共四处切片（`[:validation_num]` 两处、`[validation_num:]` 两处）用的都是同一个变量，修一处即可全部生效。

## 修复方案

推荐改回书本原版写法（第 19 行加 `int()`）：

```python
validation_num = int(x_train.shape[0] * validation_rate)
```

等价备选：样本数能被整除时可用整除写法 `validation_num = x_train.shape[0] // 5`（500 // 5 = 100）。`int()` 写法更通用，不能整除时向下取整，行为与原书一致。

## 备注

- 修复后脚本即可跑通，但默认 `optimization_trial = 100` 次试验 × 每次 50 epoch，CPU 上耗时很长。建议先把它调小（如 3~5 次）验证全流程（含最后的绘图），确认无误后再放大跑完整搜索。
- 脚本里 `__train()` 函数名以双下划线开头且定义在模块顶层，这只是命名习惯，无 mangle 副作用，不影响运行。
