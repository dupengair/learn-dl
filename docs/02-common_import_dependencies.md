# 02 — common 包的模块导入：现状分析与工业界做法

背景：第 5 章引入 `common/layer.py` 后，`common/` 内部出现了模块间引用（layer 依赖 functions，net 依赖 layer 和 gradients），而各子模块统一写 `from common import X`（从包自身导入），靠 `common/__init__.py` 的语句顺序维持正确性——当前顺序下 `import common` 直接失败。

本文回答三个问题：现在的做法是否优雅？模块多了会不会难维护？工业界的常规做法是什么？

---

## 结论（TL;DR）

1. **不优雅的点不是 `__init__.py` 做聚合导出**——包的 `__init__.py` 统一 re-export 是标准做法（numpy、torch、scikit-learn 都这样）。
2. **不优雅且脆弱的是子模块"回头导入包自身"**（`from common import X`）。它把 `__init__.py` 的**语句顺序**变成了程序正确性的一部分，而这个约束看不见、没有工具检查。
3. `common/` 的**依赖方向本身是健康的**（functions ← layer ← net，gradients ← net，单向无环），问题只出在导入写法上。
4. 工业界常规：**子模块只导入自己直接依赖的模块（完整路径），`__init__.py` 只做对外门面**。改完后 `__init__.py` 的顺序变得无关紧要。
5. 本仓库最小改法：`layer.py`、`net.py` 共改 3 行 import，其余不动。已在副本上验证通过（§5.3）。

---

## 1. 现状

### 1.1 common 内部依赖图

```
functions.py   （无内部依赖：激活函数、损失函数、softmax）
gradients.py   （无内部依赖：数值微分、梯度法）

layer.py   ──依赖──> functions.py
net.py     ──依赖──> layer.py
net.py     ──依赖──> gradients.py

dataset/   与 common 无关
```

依赖是单向分层的：`functions / gradients`（底层工具）→ `layer`（层）→ `net`（网络）。这是一张有向无环图（DAG），存在合法的拓扑序，例如：

```
functions, gradients, layer, net
```

### 1.2 各文件当前的导入写法

| 文件 | 导入语句 | 说明 |
|---|---|---|
| `common/__init__.py` | `from .functions import ...` → `from .gradients import ...` → `from .net import TwoLayerNet` → `from .layer import ...` | 聚合导出；顺序：functions → gradients → **net → layer** |
| `common/layer.py:2` | `from common import sigmoid, softmax, cross_entropy_error` | 回头导入包自身 |
| `common/net.py:4` | `from common import Affine, SoftmaxWithLoss, Relu` | 回头导入包自身 |
| `common/net.py:5` | `from common import numerical_gradient` | 回头导入包自身 |
| 测试脚本 test_* | `from common import TwoLayerNet` 等 | 门面用法，**没有问题** |

### 1.3 当前状态：`import common` 整体失败

```
ImportError: cannot import name 'Affine' from partially initialized
module 'common' (most likely due to a circular import)
```

两个因素叠加：

- **顺序违反拓扑序**：`__init__.py` 先导入 `net`（第 18 行）后导入 `layer`（第 22 行），而 net 依赖 layer。
- **子模块回头导入包**：`net.py` 执行时回头 `from common import Affine`，此刻 `common` 这个包对象正处于"初始化到一半"的状态。

只要这两个条件有一个不成立，问题就消失——这也是为什么"调整 `__init__.py` 顺序"看起来能修好它。但那只是把雷埋得更深（见 §2.3）。

---

## 2. 原理：为什么"靠 `__init__.py` 顺序"会炸

### 2.1 `from common import X` 到底做了什么

Python 执行 `from common import X` 分三步：

1. 在 `sys.modules` 缓存里找 `common`——**找到了**（正在初始化中的那个半成品对象也算找到，不会重新执行 `__init__.py`，否则会无限递归）；
2. 在这个包对象上做属性查找 `common.X`；
3. 找不到时再尝试把 `X` 当作 `common` 的子模块导入，仍失败才抛 `ImportError`。

关键在第 2 步：`common/__init__.py` 执行到第 18 行时，包对象上只绑定了前两个 import 语句导出的名字（functions、gradients 的内容），**`Affine` 还不存在**，于是 ImportError。报错信息里那句 "partially initialized module"（部分初始化的模块）说的就是这件事。

### 2.2 当前的依赖回路

```
common/__init__.py ──导入──> common/net.py ──导入──> common（包自身，初始化中）
                        └──实际需要──> common/layer.py（还没被导入）
```

回路的根源是"包 → 子模块 → 包"这条边。注意一个重要的不对称：

- `from common import Affine` 读的是**包对象的属性** → 撞上部分初始化；
- `from common.layer import Affine` 只是把 `common` 当作**定位子模块的路径容器**，真正读的是 `layer` 模块的属性 → 部分初始化的父包完全没问题。

### 2.3 所以"硬编码顺序"到底哪里不优雅

不是"有顺序"不优雅，而是**顺序成了正确性的一部分，却没有任何机制声明和保障这一点**：

1. **隐式契约不可见**：`layer.py` 能跑的前提是"`.functions` 必须排在 `.layer` 之前"，这个约束没有写在 `layer.py` 里，只存在于 `__init__.py` 的行序中。读代码的人无从得知。
2. **同一行代码，包内外语义不同**：`from common import X` 在脚本里（包已完整初始化）永远正确，在子模块里（包正在初始化）时对时错。"有时能跑"的 bug 最难排查。
3. **静态与运行时行为不一致**：IDE / 类型检查器看到 `from common import Affine` 觉得完全合法（重构成这样它们也不会报警），运行时才炸。
4. **当前顺序本身就已经错了**：`__init__.py` 把 `net` 排在 `layer` 前面，直接违反拓扑序。

---

## 3. 模块多了会怎样

按书的后半部分推演，`common/` 很快会长出 `optimizer.py`、`multi_layer_net.py`、`trainer.py`、`im2col.py`/`convolution` 相关层等七八个模块。在"子模块回头导入包"的写法下：

1. **每新增一个模块都要手工维护全局拓扑序**。新增文件若被别人（或 `__init__.py` 里的任何先导入者）传递依赖，插入位置错了就全包炸。
2. **报错信息不指根因**。`cannot import name 'X' from partially initialized module` 只告诉你"有个名字还没绑定"，是哪条回路、该调哪行顺序，要自己画依赖图推。
3. **排序不能动了**。`__init__.py` 既不能按字母序整理、也不能按功能分组——任何"美化"都可能炸，于是它逐渐变成没人敢碰的行序敏感区，diff 噪音大、code review 无法判断顺序改动是否安全。
4. **没有任何工具兜底**。顺序契约不在类型系统、不在 linter、不在测试里，只能靠运行时发现。

结论：两三个模块时还能靠脑子记住顺序，五六个以上后，这套隐式契约会成为每次改动的风险点。这正是"是否会造成维护困难"的肯定答案。

---

## 4. 工业界常规做法

### 4.1 原则一：依赖只声明在"直接依赖的模块"上，保持单向无环

子模块内部**永远不回头导入包自身**，只导入真正依赖的兄弟模块：

```python
# layer.py
from common.functions import sigmoid, softmax, cross_entropy_error

# net.py
from common.layer import Affine, SoftmaxWithLoss, Relu
from common.gradients import numerical_gradient
```

依赖关系从"`__init__.py` 的行序"这个全局隐式状态，变成"每个文件头部的 import"这个局部显式声明。pylint、mypy、IDE 都能据此直接画出依赖图、检测环。

### 4.2 原则二：`__init__.py` 只做对外门面（facade），配 `__all__`

成熟开源库的 `__init__.py` 都是聚合导出：`numpy/__init__.py` re-export 整个公共 API；scikit-learn 的 `sklearn/linear_model/__init__.py` 导出各种估计器；`torch.nn` 同理。调用方写 `from common import TwoLayerNet` 而不必知道文件布局，内部拆文件、改名都不影响调用方——这正是本仓库脚本已经在用的方式，**保留**。

可选加 `__all__`：声明公共 API，控制 `from common import *` 的范围，也给 linter 和文档工具看。

关键收益：当子模块都按 4.1 导入后，`__init__.py` 里的导入顺序**不再影响正确性**，想按字母序、按功能分组随意排。

### 4.3 原则三：包内导入用完整路径（绝对）或相对路径，二选一并保持一致

- PEP 8：推荐**绝对导入**（通常更可读）；**显式相对导入**（`from .layer import Affine`）是可接受的替代，适合可能改名/嵌套的包。requests 等库用相对导入，scikit-learn 用绝对导入，两者都是主流。
- 本仓库建议选**绝对路径**（`from common.layer import ...`）：与书原文一致，且文件被单独复制走时语义不变。
- 相对导入的坑：脚本不能直接运行一个含相对导入的模块（`python3 common/layer.py` 会报 `attempted relative import with no known parent package`），不过库模块本来也不该被直接运行，影响不大。

### 4.4 原则四：规模再大时的升级路径（认知储备，本仓库暂不需要）

- **子包分层**：`common/layers/`、`common/optimizers/` 各自带 `__init__.py` 聚合，顶层再聚合。
- **src 布局 + `pyproject.toml` + `pip install -e .`**：包变成可安装对象，`import common` 在任何目录都成立，从此不需要 `sys.path` 操作。本仓库刻意保持"无依赖清单、从根目录直接跑"，这条仅作认知储备。
- **循环实在解不开时的手术刀**：函数内延迟导入、`if TYPE_CHECKING:`（仅类型注解需要对方时）。属急救手段，不应成为常态——出现时先想想是不是分层错了。

### 4.5 原则五：把"导入是否健康"变成可检查项

- **冒烟检查**：`python3 -c "import common"`——一行命令跑遍 `__init__.py` 的整条导入链，本次的循环导入当场就能暴露。值得在每次改完 `common/` 后顺手跑一下。
- **更严格**：显式逐个 `import common.functions`、`common.layer`……，防止某子模块被从 `__init__.py` 漏掉后"看起来没人用、其实早坏了"。
- **静态工具**：pylint 的 cyclic-import 检查、mypy 都能提前抓到这类问题。
- 顺带一提：`sys.path.append(os.pardir)` 写在 `net.py` 这类**库模块**里属于书本沿袭——路径操作应该只出现在入口脚本，库代码不假设自己的安装方式。从根目录运行时它无害，但工业界不会这么做。

---

## 5. 本仓库的推荐改法（方案文档，按约定不直接改代码）

### 5.1 改动清单：共 3 行 import

```python
# common/layer.py 第 2 行
- from common import sigmoid, softmax, cross_entropy_error
+ from common.functions import sigmoid, softmax, cross_entropy_error

# common/net.py 第 4、5 行
- from common import Affine, SoftmaxWithLoss, Relu
- from common import numerical_gradient
+ from common.layer import Affine, SoftmaxWithLoss, Relu
+ from common.gradients import numerical_gradient
```

- `common/__init__.py` **可以原样保留**（改完后顺序不再重要）；可选补一个 `__all__` 列表，写法见 §5.2。
- 测试脚本**一行都不用改**——它们用的就是门面入口。
- 注意：`net.py:59` 的 `def gradient` 缩进顶格（跑到了类外）是**另一处独立笔误**，不修它 `network.gradient()` 不存在，下面的梯度检验跑不起来。修复方式：把这行（连带函数体）缩进 4 格放回 `TwoLayerNet` 类内。

### 5.2 可选：给 `__init__.py` 补 `__all__`

`__all__` 是一个**字符串列表**，逐项列出"这个包对外提供的公共名字"，惯例放在所有 import 之后（它实际在模块执行完之后才被查询，位置不影响行为，放顶部只是惯例）。它只影响一件事：`from common import *` 时哪些名字被引入。写它的收益：

- **声明公共 API**：读 `__init__.py` 的人一眼看清包的门面有什么，不用通读每条 import；
- **兜住 `import *`**：实测本包不带 `__all__` 时，`from common import *` 除了 18 个函数/类，还会混进 `functions`、`gradients`、`layer`、`net` 这 4 个**子模块对象本身**（Python 会把已导入的子模块自动绑定为包的属性）。加上 `__all__` 后这些子模块对象被挡在外面，`import *` 的结果精确等于声明的 API；
- **工具可检查**：ruff 的 F822（undefined-export）会检查 `__all__` 里每个名字都真实存在，拼写错误当场暴露。

写法只有一条铁律：**列表里每个字符串必须是 `__init__.py` 里真实绑定过的名字**（即上面各 import 引进来的名字）。`__all__` 里只写"名字"，不写模块路径：

```python
# common/__init__.py 末尾追加（分组与顺序和上方 import 一一对应，方便对照维护）

__all__ = [
    # functions.py：激活函数
    'step_function',
    'identity_function',
    'sigmoid',
    'relu',
    'softmax',
    # functions.py：损失函数
    'mean_squared_error',
    'cross_entropy_error',
    # functions.py：梯度演示用
    'function_1',
    'function_2',
    # gradients.py：数值微分与梯度法
    'numerical_gradient',
    'gradient_descent',
    # layer.py：计算图演示层与网络层
    'MulLayer',
    'AddLayer',
    'Affine',
    'Relu',
    'Sigmoid',
    'SoftmaxWithLoss',
    # net.py：网络
    'TwoLayerNet',
]
```

三点注意：

1. **只写已导入的名字**。`__all__` 里出现没绑定的名字（拼错、或想导出但忘了 import），`from common import *` 时直接 `AttributeError`。例如 `numerical_diff` 在 `gradients.py` 里有，但 `__init__.py` 没有 re-export 它，就不能进 `__all__`——想导出它，先在 import 列表加一行，再在 `__all__` 加一项。
2. **它不影响指名导入**。`from common import TwoLayerNet` 这类写法从不查 `__all__`，本仓库脚本一行都不用改。
3. **加新模块时同步两处**。将来加 `optimizer.py`：import 列表加一段、`__all__` 加一组，两处按相同顺序摆放就不容易漏。

在 §5.1 修补后的副本上实测：

```
import * 引入 18 个名字，__all__ 共 18 项
__all__ 中未绑定的名字: []
漏掉的已绑定公共名: ['functions', 'gradients', 'layer', 'net']   # 被挡住的正是 4 个子模块对象
```

### 5.3 为什么这就够了

依赖图一张没变，变的只是每条边的声明位置：从"`__init__.py` 的全局行序"下沉到"各文件自己的头部"。`common/__init__.py` → net → layer → functions 这条链依然存在，但每个子模块导入的都是**兄弟模块**而非半成品包对象，部分初始化问题不再出现。

### 5.4 验证（已在仓库外副本上实测）

```bash
python3 -c "import common"          # 冒烟：不再报 ImportError
python3 test03-3_gradient-check.py  # 业务正确性：梯度检验
```

按 5.1 修补的副本上，数值梯度 vs 反向传播梯度的平均绝对误差实测：

```
W1  1.4e-09
b1  2.8e-09
W2  1.2e-08
b2  1.8e-07
```

全部在 1e-7 以下（数值微分 h=1e-4 的截断误差量级；换用 MNIST 真实数据、书上那种 batch=3 的设置通常到 1e-10 量级）。说明修补后 `gradient()` 的反向传播链路工作正常。

---

## 6. 小结

`common/__init__.py` 统一导出本身是标准的门面模式，要消除的不是它，而是子模块对它的**反向依赖**。一句话规则：

> **包内各模块只 import 自己直接依赖的兄弟模块（完整路径），`__init__.py` 只负责对外 re-export；让 `__init__.py` 的顺序从"正确性的一部分"变回"纯风格问题"。**

这个规则成本极低（本仓库 3 行），收益随模块数量线性增长——第 6 章开始往 `common/` 加 optimizer、trainer 之前，值得先做。
