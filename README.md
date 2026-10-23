# Python 与计算光刻学习记录

记录从科学计算基础到计算光刻仿真与优化的学习过程。仓库目前包含 **144 个 Python 学习脚本**，既有早期练习，也有第一阶段的成像、ILT 和 SMO 实验。

## 从这里开始

| 内容 | 入口 | 脚本数 |
| --- | --- | ---: |
| NumPy、SciPy、Matplotlib、SymPy 基础 | [科学计算工具](01_scientific_python/README.md) | 4 |
| 本科阶段：插值、拟合、成像、测量、工艺窗口、OPC | [本科学习记录](02_undergraduate/README.md) | 77 |
| 暑期复习 | [复习记录](03_summer_review/README.md) | 1 |
| 第一阶段：FFT → 部分相干 → ILT → 稳健优化 → SMO | [计算光刻学习索引](04_lithography_stage1/README.md) | 60 |
| 二维掩膜频谱、剂量扫描与结果导出 | [独立实验](05_experiments/README.md) | 2 |

想继续计算光刻学习，直接进入 **[计算光刻第一阶段](04_lithography_stage1/README.md)**，按 day 编号查找。相同 day 下是不同实验版本，具体顺序见该页建议。

## 安装与运行

在 VS Code 中打开仓库文件夹，再打开终端。使用已经装好依赖的 Python 环境即可；新环境可按下面操作。

```bash
conda create -n litho-learning python=3.11
conda activate litho-learning
python -m pip install -r requirements.txt
```

核心依赖是 NumPy、SciPy、Matplotlib。需要运行全部科学计算笔记和交互结果导出时，安装扩展依赖：

```bash
python -m pip install -r requirements-extra.txt
```

从仓库根目录运行一个简短实验：

```bash
python "04_lithography_stage1/day02.py"
```

再查看掩膜的二维频谱：

```bash
python "05_experiments/mask_fft_2d.py"
```

大多数脚本会弹出 Matplotlib 图窗；部分脚本需要关闭前一张图后才继续。代码中的参数一般直接在文件开头修改。VS Code 使用 `Python: Select Interpreter` 选择相同环境。更多说明见 [运行指南](docs/RUNNING.md)。

## 如何使用这份学习记录

- 每次只选一个脚本，先看掩膜、网格、光学参数，再对应成像与优化步骤。
- 修改一个参数，比较图像和打印指标，并记录原因。
- 文件中的 `day` 是原始学习编号，不代表固定日期或学习进度要求。
- 各阶段包含不同程度的简化模型；光学振幅、光强、归一化、空间单位和指标定义需要结合具体脚本判断。
- 这些文件保留学习时的实验写法，运行通过不等于物理模型或全部数值结论已经验证。

## 维护与历史

- [旧路径 → 新路径完整对照](docs/FILE_MAP.md)：查找整理前的文件。
- [新增实验与日常提交](docs/WORKFLOW.md)：后续文件放置、命名和结果记录方法。
- [整理验证记录](docs/VALIDATION.md)：本次实际检查的范围与限制。

本次整理保留全部原有 Python 文件的内容和学习编号，仅调整存放位置与文件名。原来的 `test.py` 是实验脚本，现归入独立实验，不作为自动化测试。
