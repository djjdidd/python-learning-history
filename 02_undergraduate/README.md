# 本科学习记录

[返回首页](../README.md)

保留 77 个历史脚本，包括逐日学习、额外练习和总结。编号可能不连续，未补造缺失天数。下表的内容提示取自原脚本注释，作为查找线索。

| 脚本 | 内容提示 |
| --- | --- |
| [day01_20_summary.py](day01_20_summary.py) | 前20天总结 |
| [day01_离散数据到连续曲线——scipy_插值入门.py](day01_%E7%A6%BB%E6%95%A3%E6%95%B0%E6%8D%AE%E5%88%B0%E8%BF%9E%E7%BB%AD%E6%9B%B2%E7%BA%BF%E2%80%94%E2%80%94scipy_%E6%8F%92%E5%80%BC%E5%85%A5%E9%97%A8.py) | 第一步：准备已知数据 |
| [day02_掩模如何变成成像——卷积与一维光刻模型.py](day02_%E6%8E%A9%E6%A8%A1%E5%A6%82%E4%BD%95%E5%8F%98%E6%88%90%E6%88%90%E5%83%8F%E2%80%94%E2%80%94%E5%8D%B7%E7%A7%AF%E4%B8%8E%E4%B8%80%E7%BB%B4%E5%85%89%E5%88%BB%E6%A8%A1%E5%9E%8B.py) | 1. 构造一维空间坐标 |
| [day03_从成像到打印——阈值判定与二值图形.py](day03_%E4%BB%8E%E6%88%90%E5%83%8F%E5%88%B0%E6%89%93%E5%8D%B0%E2%80%94%E2%80%94%E9%98%88%E5%80%BC%E5%88%A4%E5%AE%9A%E4%B8%8E%E4%BA%8C%E5%80%BC%E5%9B%BE%E5%BD%A2.py) | 基础练习（见代码） |
| [day04.py](day04.py) | 剂量如何改变线宽——打印宽度测量与 dose 分析 |
| [day05.py](day05.py) | 从数据反推模型——SciPy 曲线拟合与光刻参数标定 |
| [day06.py](day06.py) | 用误差来评价拟合效果——RMSE 与 R² 比较模型好坏 |
| [day07.py](day07.py) | 1. 空间网格 |
| [day08.py](day08.py) | 二维掩模与二维成像入门 |
| [day09.py](day09.py) | 二维阈值打印与桥连/断裂直觉 |
| [day10.py](day10.py) | 二维图形测量入门 |
| [day11.py](day11.py) | 曲线拟合入门——从离散数据到参数化模型 |
| [day12.py](day12.py) | least_squares 与“残差”概念 |
| [day13.py](day13.py) | 二维图形的噪声、阈值敏感性与简单后处理 |
| [day14.py](day14.py) | 二维插值与 focus-dose 工艺数据可视化 |
| [day15.py](day15.py) | 二维图形边缘提取与 CD 测量入门 |
| [day16.py](day16.py) | dose-threshold 参数扫描入门 |
| [day17.py](day17.py) | 参数扫描结果热图与二维工艺窗口入门 |
| [day18.py](day18.py) | 第18天：目标函数与自动找最优 dose |
| [day18优化版.py](day18%E4%BC%98%E5%8C%96%E7%89%88.py) | 第18天：目标函数与自动找最优 dose优化版 |
| [day19.py](day19.py) | 噪声下的鲁棒性分析 |
| [day20.py](day20.py) | 第20天：简化版可用工艺窗口提取 |
| [day22.py](day22.py) | 第22天：线端缩短（Line-End Shortening）与二维图形局部失真 |
| [day23.py](day23.py) | 第23天：line-space-line 结构的多指标测量 |
| [day24.py](day24.py) | 局部薄弱区分析 —— 找出图形最先失效的位置 |
| [day25.py](day25.py) | 局部失效图 —— 把每一行分成“安全 / 偏弱 / 失效” |
| [day26.py](day26.py) | 第26天：参数扫描下的局部工艺窗口分析 |
| [day27.py](day27.py) | 第27天：二维参数扫描下的局部工艺窗口热图 |
| [day28.py](day28.py) | 第28天：从比例热图到整体通过/失败窗口图 |
| [day29.py](day29.py) | 整体窗口规则从“硬判定”升级到“带容忍度的判定” |
| [day30.py](day30.py) | 在 pass 区里选最优点，不只看 safe_ratio |
| [day31.py](day31.py) | 从“比例指标”走向“最差值指标” |
| [day32.py](day32.py) | 多指标整体判定规则：把比例型指标和 worst-case 指标放在同一个整体规则里，才能同时考虑整体趋势和局部极端退化 |
| [day33.py](day33.py) | 目标函数设计 |
| [day34.py](day34.py) | 最简单的“迭代改参数”直觉 |
| [day35.py](day35.py) | 1. 建立二维网格 |
| [day36.py](day36.py) | 约束和目标，到底分别在干什么 |
| [day37.py](day37.py) | 目标函数权重敏感性 |
| [day38.py](day38.py) | 第38天：多指标整体判定再升级——到底是谁让它 fail |
| [day39.py](day39.py) | 如果某原因只集中在很小区域，说明它是局部热点问题 |
| [day40.py](day40.py) | 多原因诊断的“主显示优先级”——一个点有多个问题时，热图到底显示谁 |
| [day41.py](day41.py) | 瓶颈占比图——频次和占比到底有什么区别 |
| [day42.py](day42.py) | 从 CD / gap 过渡到 EPE——什么叫边缘位置误差 EPE = Edge Placement Error |
| [day43.py](day43.py) | 从逐行 EPE 走向 contour——target contour 和 printed contour 到底是什么 |
| [day44.py](day44.py) | worst-case EPE 与 hotspot——怎么从边缘误差里找最危险位置 |
| [day45.py](day45.py) | 为什么需要 OPC——从误差诊断走向图形修正 |
| [day46.py](day46.py) | 一周总结——从 EPE / contour / hotspot 到 toy OPC |
| [day47.py](day47.py) | 从一步 OPC 到迭代 OPC——为什么修一次通常不够 iterative迭代的 |
| [day48.py](day48.py) | 迭代 OPC 为什么会振荡oscillation，以及怎么让它更稳定 |
| [day49.py](day49.py) | 从单一 worst EPE 到综合目标函数 objective |
| [day50.py](day50.py) | 不同 objective 会导致不同 OPC 行为 |
| [day51.py](day51.py) | 从全局边缘平移到局部 hotspot 修正 |
| [day52.py](day52.py) | 第52天：分段 OPC——不同 y 段使用不同修正量 |
| [day53.py](day53.py) | 第53天：OPC 为什么需要约束 |
| [day54.py](day54.py) | 第54天：从 OPC 过渡到 ILT |
| [day56.py](day56.py) | 第56天：真实 OPC 为什么更复杂 |
| [第一天练习.py](%E7%AC%AC%E4%B8%80%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 基础练习（见代码） |
| [第三天练习.py](%E7%AC%AC%E4%B8%89%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 阈值较低时，中间较弱的成像区域也可能超过阈值，因此两条线更容易桥连。 |
| [第九天练习.py](%E7%AC%AC%E4%B9%9D%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 基础练习（见代码） |
| [第二十三天练习.py](%E7%AC%AC%E4%BA%8C%E5%8D%81%E4%B8%89%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 第23天：line-space-line 结构的多指标测量 |
| [第二十八天练习.py](%E7%AC%AC%E4%BA%8C%E5%8D%81%E5%85%AB%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 第28天练习：从比例热图到整体通过/失败窗口图 |
| [第二十六天练习.py](%E7%AC%AC%E4%BA%8C%E5%8D%81%E5%85%AD%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 第26天练习：把今天的“最佳 dose 标准”从：safe_ratio 最大改成一个新的综合评价 |
| [第二十四天练习.py](%E7%AC%AC%E4%BA%8C%E5%8D%81%E5%9B%9B%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 构造gap随y变化的mask |
| [第二十四天练习2.py](%E7%AC%AC%E4%BA%8C%E5%8D%81%E5%9B%9B%E5%A4%A9%E7%BB%83%E4%B9%A02.py) | 构造梯形mask |
| [第二十天练习.py](%E7%AC%AC%E4%BA%8C%E5%8D%81%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 线-空-线结构的可用工艺窗口分析 |
| [第二天练习.py](%E7%AC%AC%E4%BA%8C%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 1. 一维坐标 |
| [第五天练习.py](%E7%AC%AC%E4%BA%94%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 拟合得到的参数 |
| [第六天练习.py](%E7%AC%AC%E5%85%AD%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 基础练习（见代码） |
| [第十一天练习.py](%E7%AC%AC%E5%8D%81%E4%B8%80%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 基础练习（见代码） |
| [第十三天练习.py](%E7%AC%AC%E5%8D%81%E4%B8%89%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 1. 建立二维空间网格 |
| [第十九天练习.py](%E7%AC%AC%E5%8D%81%E4%B9%9D%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 基础练习（见代码） |
| [第十二天练习.py](%E7%AC%AC%E5%8D%81%E4%BA%8C%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 基础练习（见代码） |
| [第十五天练习.py](%E7%AC%AC%E5%8D%81%E4%BA%94%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 构造“上宽下窄”的 mask |
| [第十八天练习.py](%E7%AC%AC%E5%8D%81%E5%85%AB%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | ---- 图1：原始 mask ---- |
| [第十四天练习.py](%E7%AC%AC%E5%8D%81%E5%9B%9B%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 基础练习（见代码） |
| [第十天练习.py](%E7%AC%AC%E5%8D%81%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 找到二维打印图中间那一行 |
| [第四天练习.py](%E7%AC%AC%E5%9B%9B%E5%A4%A9%E7%BB%83%E4%B9%A0.py) | 两条线间距变小时，打印结果更容易粘连，不容易保持分开。 |
