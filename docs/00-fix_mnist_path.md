# MNIST 数据集路径问题修复方案

> 目标：`dataset/mnist.py` 能识别已手动下载到 `dataset/mnist/` 子目录的 4 个 `.gz` 文件，
> 存在则跳过下载，并正常完成"解压 → 转 NumPy → 生成 mnist.pkl"流程，
> 最终让 `test01-3_mnist.py` 顺利运行。

## 一、问题分析

### 1.1 当前代码的执行流程

```
test01-3_mnist.py
  └─ load_mnist()
       ├─ dataset/mnist.pkl 存在？
       │    ├─ 是 → 直接读 pickle 返回（之后的运行都走这条快速路径）
       │    └─ 否 → init_mnist()
       │              ├─ download_mnist()   # 逐个检查 4 个 .gz，缺失才下载
       │              ├─ _convert_numpy()   # gzip.open 读取 .gz → NumPy 数组
       │              └─ pickle.dump()      # 生成 dataset/mnist.pkl
```

### 1.2 问题根源

`dataset/mnist.py` 第 21 行定义了基准目录：

```python
dataset_dir = os.path.dirname(os.path.abspath(__file__))   # 即 dataset/ 目录
```

随后所有路径都基于 `dataset_dir` 拼接：

| 函数 | 拼出的路径 | 实际情况 |
|---|---|---|
| `_download`（第 31 行） | `dataset/train-images-idx3-ubyte.gz` | 文件在 `dataset/mnist/` 下，检查不到 |
| `_load_img`（第 55 行） | `dataset/train-images-idx3-ubyte.gz` | 同上，读取时也会失败 |
| `_load_label`（第 45 行） | `dataset/train-labels-idx1-ubyte.gz` | 同上 |

于是 `_download` 里的 `os.path.exists()` 判定"文件不存在"，转而访问
`http://yann.lecun.com/exdb/mnist/` 下载——该源如今已不稳定，直接报错。

**关键认识**：
1. "已存在就不下载"的逻辑（`if os.path.exists(file_path): return`）代码里**本来就有**，
   只是它检查的是错误目录，所以永远判定"不存在"。
2. "解压"不需要额外步骤——`_load_img` / `_load_label` 用的是 `gzip.open()`，
   读取时即解压，直接得到 NumPy 数组。**问题只出在路径拼接上。**

## 二、背景知识：mnist.pkl 的作用与目录职责划分

### 2.1 mnist.pkl 起什么作用

`mnist.pkl` 是 `init_mnist()` 用 `pickle.dump(dataset, f, -1)` 生成的**序列化缓存文件**
（`-1` 表示使用最高二进制协议，比默认协议更快、文件更小）。

**里面存的是什么**：一个字典，含 4 个已从 IDX 二进制格式转换好的 NumPy 数组——

| 键 | 形状 | dtype | 说明 |
|---|---|---|---|
| `train_img` | (60000, 784) | uint8 | 训练图像，像素值 0~255 |
| `train_label` | (60000,) | uint8 | 训练标签 |
| `test_img` | (10000, 784) | uint8 | 测试图像 |
| `test_label` | (10000,) | uint8 | 测试标签 |

文件体积约 55 MB（(60000+10000)×784 字节的图像数据 + 标签）。

**为什么要它**：`.gz` 原始文件是压缩的 IDX 二进制格式，每次使用都要重复
"gzip 解压 → `np.frombuffer` 按字节解析 → reshape"。pickle 缓存把这套转换
**只做一次**，之后每次 `load_mnist()` 直接 `pickle.load` 拿到现成的 NumPy 数组。

**一个容易忽略的设计细节**：pickle 里存的是**未经任何加工的中间形态**
（uint8、0~255、一维标签）。`load_mnist` 的三个参数 `normalize` / `one_hot_label`
/ `flatten` 都是**读入 pickle 之后**才动态应用的。这样一份缓存就能服务所有
参数组合——如果缓存时就把数据正规化了，`normalize=False` 的需求就无法满足。
这是"缓存中间结果，最终形态按需计算"的典型做法。

### 2.2 为什么不直接把 dataset_dir 改成 dataset/mnist

先说结论：直接改**也能跑通**，只是 `mnist.pkl` 会跟着生成在 `dataset/mnist/` 里，
功能上没有对错之分。方案选择"新增 `mnist_dir`"基于以下几点考虑：

**理由 1：避免隐式连锁改动（最小影响面原则）**

`dataset_dir` 有两个使用者：`.gz` 的寻址（`_download` / `_load_*`）和
`save_file` 的寻址。本次问题只出在 `.gz` 寻址上，`save_file` 的行为是对的、
不需要变。如果直接改 `dataset_dir` 的值，`save_file` 会被"顺带"改掉——
一个不需要变的行为发生了隐式变化。这类"改 A 坏 B"的连锁效应正是 bug 的温床。
新增独立变量，等于显式声明：**只有 `.gz` 的路径变了，缓存位置不变**。

**理由 2：变量的名字与语义要一致**

`dataset_dir` 的语义是"本模块所在目录"（`os.path.dirname(os.path.abspath(__file__))`），
`dataset/` 下还有 `mnist.py` 本身。让它指向 `dataset/mnist/` 后，名字叫
"dataset 目录"、值却是"mnist 子目录"，后来读代码的人（包括几个月后的自己）
会被误导。

**理由 3：原始数据与派生数据分开存放**

`.gz` 和 `mnist.pkl` 是两种性质完全不同的文件：

```
dataset/
├── mnist.py          ← 模块代码
├── mnist.pkl         ← 派生数据：本地生成，可随时删除重建
├── lena.png
└── mnist/            ← 原始数据：从外部获取，只读，不应改动
    ├── train-images-idx3-ubyte.gz
    ├── train-labels-idx1-ubyte.gz
    ├── t10k-images-idx3-ubyte.gz
    └── t10k-labels-idx1-ubyte.gz
```

原始数据（下载来的，坏了得重新下）和派生缓存（本地算出来的，删了再生成就行）
分开放，目录结构本身就表达了这层区别。混在同一个目录里，就无法一眼分辨
"哪些文件动不得、哪些可以随手清理"。比如以后想清缓存，直接删
`dataset/mnist.pkl` 即可，不用担心误碰原始数据。

**理由 4：不弃用已有缓存**

如果一台机器上已经在 `dataset/mnist.pkl` 生成过缓存，直接改 `dataset_dir`
会让代码去 `dataset/mnist/mnist.pkl` 找缓存——旧缓存被无声遗弃，
还会多做一次全量转换。保留 `save_file` 原路径则完全向后兼容。

## 三、修改方案总览

思路：新增一个变量 `mnist_dir` 指向 `.gz` 实际所在子目录，让"检查/下载/读取"
三个环节都改用它；`mnist.pkl` 缓存位置不动，仍在 `dataset/` 下。

共 4 处改动（3 处改一行，1 处加两行），其余函数零改动：

| # | 位置 | 改动 |
|---|---|---|
| 1 | 第 21-22 行附近 | 新增 `mnist_dir` 变量 |
| 2 | `_download()` | 路径改用 `mnist_dir`；下载前确保目录存在 |
| 3 | `_load_label()` | 路径改用 `mnist_dir` |
| 4 | `_load_img()` | 路径改用 `mnist_dir` |

## 四、具体代码改动

### 改动 1：新增 gz 文件目录变量（第 21-22 行）

```python
# ---- 修改前 ----
dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

# ---- 修改后 ----
dataset_dir = os.path.dirname(os.path.abspath(__file__))
mnist_dir = dataset_dir + "/mnist"        # 新增：.gz 文件所在子目录
save_file = dataset_dir + "/mnist.pkl"    # pickle 缓存仍放在 dataset/ 下
```

**说明**：`dataset_dir` 保留原语义（模块所在目录），继续供 `save_file` 使用；
`.gz` 的寻址统一走新变量 `mnist_dir`。

### 改动 2：`_download()`（第 30-38 行）

```python
# ---- 修改前 ----
def _download(file_name):
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")

# ---- 修改后 ----
def _download(file_name):
    file_path = mnist_dir + "/" + file_name     # 改：用 mnist_dir 拼路径

    if os.path.exists(file_path):
        return                                   # 已存在 → 跳过下载（原本就有这逻辑）

    os.makedirs(mnist_dir, exist_ok=True)        # 新增：全新环境下目录可能不存在
    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
```

**说明**：
- 第一行改成 `mnist_dir` 后，`os.path.exists()` 就能命中你手动下载的文件，直接 `return`，不再访问网络。
- `os.makedirs(..., exist_ok=True)` 是防御性代码：仅在没有现成文件、真要下载时才会执行；
  目录已存在时不报错。你现在的环境里这行不会触发，但换机器时就用得上了。

### 改动 3：`_load_label()`（第 44-45 行）

```python
# ---- 修改前 ----
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

# ---- 修改后 ----
def _load_label(file_name):
    file_path = mnist_dir + "/" + file_name     # 只改这一行
```

### 改动 4：`_load_img()`（第 54-55 行）

```python
# ---- 修改前 ----
def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

# ---- 修改后 ----
def _load_img(file_name):
    file_path = mnist_dir + "/" + file_name     # 只改这一行
```

### 不需要改动的部分

- `download_mnist()`：只是循环调用 `_download`，不含路径逻辑。
- `_convert_numpy()`：只传文件名，路径在 `_load_img` / `_load_label` 内部拼。
- `init_mnist()` / `load_mnist()`：只管 pickle 的生成与读取，与 `.gz` 路径无关。
- `test01-3_mnist.py`：完全不用动，`load_mnist()` 会自动走"检查 → 转换 → 缓存"全流程。

## 五、验证步骤

改完后按顺序执行：

```bash
# 1.（可选）单独初始化，观察转换流程
python3 dataset/mnist.py
# 预期输出（不应出现任何 Downloading 字样）：
#   Converting train-images-idx3-ubyte.gz to NumPy Array ...
#   Done
#   Converting train-labels-idx1-ubyte.gz to NumPy Array ...
#   Done
#   Converting t10k-images-idx3-ubyte.gz to NumPy Array ...
#   Done
#   Converting t10k-labels-idx1-ubyte.gz to NumPy Array ...
#   Done
#   Creating pickle file ...
#   Done!
# 并在 dataset/ 下生成 mnist.pkl

# 2. 运行学习脚本
python3 test01-3_mnist.py
# 预期输出：
#   (60000, 784)
#   (60000, 10)

# 3. 再次运行（验证 pickle 缓存生效，应秒出结果且无 Converting 输出）
python3 test01-3_mnist.py
```

## 六、杂项说明

- `dataset/mnist/` 下的 `*.gz:Zone.Identifier` 文件是 Windows/WSL 下载时产生的元数据，
  对脚本无影响（代码按精确文件名检查），可随意删除。
- 书中原始代码基于旧版 numpy，本机为 numpy 2.x，`mnist.py` 用到的
  `np.frombuffer` / `pickle` 等接口在 2.x 下均正常，无需额外适配。

## 七、可选进阶（不影响本次目标，仅供参考）

1. **换下载源**：`url_base` 指向的 yann.lecun.com 已不稳定，可改为常用镜像：
   ```python
   url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
   ```
2. **文件完整性校验**：下载源不稳定可能得到截断的文件。可在 `_download` 中加文件大小校验
   （正确大小：train-images 9912422、train-labels 28881、t10k-images 1648877、
   t10k-labels 4542 字节），大小不符则重新下载。
3. **用 `os.path.join` 拼路径**：比字符串拼接 `+ "/" +` 更跨平台、更不易错，
   例如 `os.path.join(mnist_dir, file_name)`。本方案为保持与书中代码风格一致、
   便于对照 diff，未做此替换。
