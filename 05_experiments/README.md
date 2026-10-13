# 独立实验

[返回首页](../README.md)

| 文件 | 原文件名 | 用途 |
| --- | --- | --- |
| [mask_fft_2d.py](mask_fft_2d.py) | prac.py | 比较矩形与细线掩膜的二维 FFT 幅度谱 |
| [dose_sweep.py](dose_sweep.py) | test.py | 剂量扫描，导出 CD/EPE 图、交互 HTML 和数值表 |

从仓库根目录运行：

```bash
python "05_experiments/mask_fft_2d.py"
python -m pip install -r requirements-extra.txt
python "05_experiments/dose_sweep.py"
```

`dose_sweep.py` 按脚本位置定位输出；移动到本目录后，输出会落在仓库内的 `results/day7_results/`。该目录由脚本自动创建，默认不提交到 Git。原脚本中的输出目录名和数值实现保持原样。

剂量扫描是历史探索代码，CD/EPE 的边缘选择、目标边缘及光学/胶模型仍需另行核对；导出成功仅代表运行流程可用。
