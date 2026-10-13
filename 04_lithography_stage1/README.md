# 计算光刻 · 第一阶段

[返回首页](../README.md)

保留原来 day 1–24 的 60 个实验。每个文件都是可单独打开的学习脚本；相同 day 下保留不同模型与迭代版本。

## 按主题定位

| 原始编号 | 主要内容 |
| --- | --- |
| day01–05 | 卷积、PSF、FFT、pupil 与周期图形 |
| day06–10 | 二维成像、soft resist、ILT、TV 与二值化 |
| day11–15 | 离轴/双极照明、光源扫描、CD/EPE 与离焦 |
| day16–18 | NILS、工艺窗口、Hopkins/TCC 与部分相干 ILT |
| day19–22 | 二维部分相干成像、工艺窗口与 ILT |
| day23–24 | 稳健优化、光源优化与交替 SMO |

同一天有多份文件时，建议先看成像/前向计算，再看梯度检查，最后看优化和比较。day24 可按 `source_shift_gradient_check` → `continuous_source_shift` → `continuous_alternating_smo` → `compare` 阅读。文件名中的 `gradient` 不一定代表独立梯度测试，以代码实际内容为准。

首次运行可以选 `day02.py`。后期的参数扫描、特征分解和多条件优化可能耗时较长，不建议一开始批量运行。

## 完整脚本索引

### day01 · PSF 成像与 dose–width 扫描

- [day01.py](day01.py)

### day02 · 离散卷积与正弦信号

- [day02.py](day02.py)

### day03 · 矩形掩膜与 FFT 频谱

- [day03_fft.py](day03_fft.py)

### day04 · pupil 截止频率与重建

- [day04.py](day04.py)

### day05 · 周期图形、频谱与低通成像

- [day05.py](day05.py)
- [day05_gradient.py](day05_gradient.py)

### day06 · 二维 PSF 成像、serif 与参数优化

- [day06_2d.py](day06_2d.py)
- [day06_optimize.py](day06_optimize.py)
- [day06_serif.py](day06_serif.py)

### day07 · soft resist、连续松弛与迭代优化

- [day07_continuous_relaxation.py](day07_continuous_relaxation.py)
- [day07_iterative_optimization.py](day07_iterative_optimization.py)
- [day07_scan_and_optimization.py](day07_scan_and_optimization.py)
- [day07_soft_resist.py](day07_soft_resist.py)

### day08 · 像素化 ILT 与梯度对比

- [day08_gradient_compare.py](day08_gradient_compare.py)
- [day08_iterative_optimization.py](day08_iterative_optimization.py)
- [day08_pixel_based_ilt.py](day08_pixel_based_ilt.py)

### day09 · ILT 迭代与 TV 正则

- [day09_iterative.py](day09_iterative.py)
- [day09_tv_regularization.py](day09_tv_regularization.py)

### day10 · 二值化损失、TV 与物理 pupil

- [day10_binary_loss.py](day10_binary_loss.py)
- [day10_more_than_toy.py](day10_more_than_toy.py)
- [day10_with_tv.py](day10_with_tv.py)

### day11 · NA、梯度检查与离轴照明

- [day11_different_na_contrast.py](day11_different_na_contrast.py)
- [day11_gradient_check.py](day11_gradient_check.py)
- [day11_off_axis_illumination.py](day11_off_axis_illumination.py)
- [day11_shifted_pupil.py](day11_shifted_pupil.py)

### day12 · 双极照明

- [day12_dipole_illumination.py](day12_dipole_illumination.py)

### day13 · 光源位置扫描与 Pareto 比较

- [day13_pareto_front.py](day13_pareto_front.py)
- [day13_scan_for_best_source.py](day13_scan_for_best_source.py)

### day14 · CD、EPE 与离焦

- [day14_cd_and_epe.py](day14_cd_and_epe.py)
- [day14_defocus.py](day14_defocus.py)

### day15 · 离焦扫描

- [day15_scan_defocus.py](day15_scan_defocus.py)

### day16 · 对比度、NILS、工艺窗口与 Hopkins 入门

- [day16_contrast_and_nils.py](day16_contrast_and_nils.py)
- [day16_defocus_um.py](day16_defocus_um.py)
- [day16_dipole_and_normal_illumination.py](day16_dipole_and_normal_illumination.py)
- [day16_first_hopkins.py](day16_first_hopkins.py)
- [day16_pixel_nils.py](day16_pixel_nils.py)
- [day16_process_window.py](day16_process_window.py)

### day17 · TCC 特征分解、多光源与模式截断

- [day17_5_mode_resist.py](day17_5_mode_resist.py)
- [day17_more_light_sources.py](day17_more_light_sources.py)
- [day17_tcc_eigendecomposition.py](day17_tcc_eigendecomposition.py)

### day18 · 部分相干 ILT、二值化与正则化

- [day18_add_binarization_ilt.py](day18_add_binarization_ilt.py)
- [day18_more_real_ilt.py](day18_more_real_ilt.py)
- [day18_regularized_partial_coherent_ilt.py](day18_regularized_partial_coherent_ilt.py)

### day19 · 二维部分相干成像与光源扫描

- [day19_2d.py](day19_2d.py)
- [day19_periodic_line.py](day19_periodic_line.py)
- [day19_scan_source_position.py](day19_scan_source_position.py)

### day20 · 二维 CD、NILS、DOF 与工艺窗口

- [day20_cd_and_nils.py](day20_cd_and_nils.py)
- [day20_defocus_and_dof.py](day20_defocus_and_dof.py)
- [day20_process_window.py](day20_process_window.py)

### day21 · 二维 ILT 前向、梯度与迭代

- [day21_2d_ilt_one_gradient_model.py](day21_2d_ilt_one_gradient_model.py)
- [day21_2d_ilt_total.py](day21_2d_ilt_total.py)
- [day21_ilt_optimization.py](day21_ilt_optimization.py)

### day22 · 双光源部分相干二维 ILT

- [day22_2_source_partial_coherent_2d_ilt.py](day22_2_source_partial_coherent_2d_ilt.py)

### day23 · 标称/稳健 ILT、soft worst-case 与 SMO

- [day23_nominal_vs_robust_ilt.py](day23_nominal_vs_robust_ilt.py)
- [day23_smo.py](day23_smo.py)
- [day23_soft_worst_case_objective.py](day23_soft_worst_case_objective.py)

### day24 · 光源位置梯度、连续优化与交替 SMO

- [day24_compare.py](day24_compare.py)
- [day24_continuous_alternating_smo.py](day24_continuous_alternating_smo.py)
- [day24_continuous_source_shift.py](day24_continuous_source_shift.py)
- [day24_source_shift_gradient_check.py](day24_source_shift_gradient_check.py)
