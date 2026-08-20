import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

x = np.linspace(-2.0, 2.0, 2000)
dx = x[1] - x[0]

base_center = 0.50
sigma = 0.12
psf = np.exp(-x**2 / (2 * sigma**2))
psf = psf / np.sum(psf)

threshold = 0.30
dose = 1.0

mask_list = []#忘记
printed_list = []
center_list = []

for i in range(3):
    center = base_center - 0.15 * i
    maski = np.where((np.abs(x + center) <= 0.10) | (np.abs(x - center) <= 0.10),1.0,0.0)
    aerial_imagei = signal.fftconvolve(maski, psf, mode='same')
    image_with_dosei = dose * aerial_imagei
    printed_patterni = np.where(image_with_dosei >= threshold, 1.0, 0.0)
    mask_list.append(maski)
    printed_list.append(printed_patterni)
    center_list.append(center)

plt.figure(figsize=(10, 8))

for i in range(3):
    center = center_list[i]

    plt.subplot(3, 2, i * 2 + 1)
    plt.plot(x, mask_list[i])
    plt.title(f"Mask: centers = ({-center:.2f}, {center:.2f})")
    plt.xlabel("Position")
    plt.ylabel("Intensity")
    plt.grid(True)

    plt.subplot(3, 2, i * 2 + 2)
    plt.plot(x, printed_list[i])
    plt.title(f"Printed Pattern: centers = ({-center:.2f}, {center:.2f})")
    plt.xlabel("Position")
    plt.ylabel("Binary Value")
    plt.grid(True)

plt.tight_layout()
plt.show()

# 两条线间距变小时，打印结果更容易粘连，不容易保持分开。
# 在相同的 dose 和 threshold 下，间距越小，两条线的成像分布重叠越强，因此打印结果更容易连在一起。
# 这说明图形分辨能力有限，当图形间距过小时，系统更难把相邻图形清楚地区分开。

# ==============================
# 第四天作业：本次代码容易出错的地方
# ==============================
# 1. 如果在 for 循环里直接反复写 maski / printed_patterni，
#    但不保存到列表中，那么循环结束后只会保留最后一组结果。
#
# 2. 如果后面画图时还在直接使用 maski / printed_patterni，
#    那么画出来的很可能只是最后一组，而不是三组不同结果。
#
# 3. subplot(3, 2, 编号) 的编号必须按 1,2,3,4,5,6 排，
#    不能写成 i+1 和 i+2，否则后面的图会覆盖前面的图。
#
# 4. printed_pattern 的判定必须使用 image_with_dose，
#    不能错写成直接用 aerial_image，否则 dose 就没有真正起作用。
#
# 5. center_list 既然保存了每组中心位置，后面最好在标题中用出来，
#    否则很难看出三组图到底对应哪一组间距。
#
# 6. 变量名尽量表达物理意义。
#    比如 number 不如 base_center 更清楚。
# ==============================