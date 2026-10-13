# 运行指南

[返回首页](../README.md)

## 选择环境

推荐新建 Python 3.11 环境，也可以使用已有环境。仓库是独立脚本集合，无须安装成 Python 包。

```bash
conda create -n litho-learning python=3.11
conda activate litho-learning
python -m pip install -r requirements.txt
```

不使用 Conda 时，可以用 `python -m venv .venv` 建环境；Windows PowerShell 用 `.\.venv\Scripts\Activate.ps1` 激活，macOS/Linux 用 `source .venv/bin/activate` 激活，再安装依赖。

核心依赖为 numpy、scipy、matplotlib。扩展安装 `python -m pip install -r requirements-extra.txt` 包含 pandas、sympy、plotly，分别用于基础笔记、符号计算和 HTML 图表导出。依赖文件未锁定版本；实际检查环境见 [验证记录](VALIDATION.md)。

## 运行一个文件

在 VS Code 中选择 `Python: Select Interpreter`，选中装好依赖的环境。在仓库根目录执行：

```bash
python "04_lithography_stage1/day02.py"
```

也可以打开该文件，点击“运行 Python 文件”。大部分实验以绘图窗口呈现结果，多个 `plt.show()` 可能要求关闭当前图窗后才继续。

## 常见情况

| 现象 | 处理 |
| --- | --- |
| `ModuleNotFoundError` | 确认 VS Code 与终端使用同一个 Python，再安装对应依赖 |
| 找不到旧文件名 | 在 [路径对照](FILE_MAP.md) 查找新位置 |
| 运行后暂时没有图 | 查看终端是否仍在计算；部分优化在完成所有迭代后才绘图 |
| 中文显示为方框 | 为 Matplotlib 选择本机已安装的中文字体 |
| 导出文件不见了 | 剂量扫描结果位于仓库的 `results/day7_results/` |

首次建议运行 `day02.py` 或 `05_experiments/mask_fft_2d.py`。后期二维优化、多条件扫描、TCC 特征分解的开销不同；调低网格或迭代次数会改变结果，实验时应一并记录。

## 数值结果的边界

不同历史文件的单位、归一化、胶模型和误差指标可能不同。不要仅凭曲线相似就横向比较。做定量比较前，核对网格、波长/NA、光源权重、阈值与剂量，以及 CD/EPE 的测量位置和符号定义。
