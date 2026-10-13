# 整理验证记录

[返回首页](../README.md)

整理日期：2026-10-13。基于 main 提交 `5519234e259fd90378e9b25450ef50e1a5e1b78b`。

## 文件与文档

- 144 个原始 Python 脚本全部保留，整理后的内容与原 Git 对象逐字节一致。
- 144 个脚本全部通过 Python 语法解析。
- 已核对文档中的本地文件链接；完整旧/新路径见 [文件对照](FILE_MAP.md)。
- 根目录 VS Code 环境设置保留原样。
- 依赖按实际 import 整理为核心与扩展两组。

## 抽样运行

使用 Matplotlib 的 Agg 后端（无图形窗口），以原始参数运行以下 5 个脚本，均正常退出：

| 脚本 | 检查内容 |
| --- | --- |
| `04_lithography_stage1/day02.py` | 离散卷积与信号绘图流程 |
| `04_lithography_stage1/day06_2d.py` | 二维 PSF 成像流程 |
| `04_lithography_stage1/day08_gradient_compare.py` | 4 个像素的解析/有限差分梯度比较 |
| `04_lithography_stage1/day24_source_shift_gradient_check.py` | 离焦条件下的光源位置梯度比较 |
| `05_experiments/mask_fft_2d.py` | 二维掩膜 FFT 与绘图流程 |

运行时已检查输出：day08 的相对误差约为 `4.9e-10` 至 `7.2e-8`；day24 的相对误差约为 `6.4e-9`。

环境：Python 3.12.14，NumPy 2.3.5、SciPy 1.17.0、Matplotlib 3.10.8。为稳定运行时间，BLAS/OpenMP 线程数设为 1；这些耗时不代表个人电脑性能。

## 验证范围

未对所有历史脚本逐个执行，也未做完整物理正确性审查或桌面图窗交互检查。本环境缺少 SymPy 和 Plotly，因此相应笔记与剂量扫描 HTML 导出未执行；依赖已列在 `requirements-extra.txt`。

本次不改变数值实现。特别是独立剂量扫描实验的 CD/EPE 边缘选择和模型解释仍需单独审查，不能从文件整理推断其物理结论正确。
