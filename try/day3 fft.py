import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 建立空间坐标
# ============================================================

# 在 -2 到 2 之间建立 2000 个空间采样点
x = np.linspace(-2, 2, 2000)

# 相邻两个空间采样点之间的距离
# 后面 fftfreq() 需要 dx 来计算真正的空间频率坐标
dx = x[1] - x[0]


# ============================================================
# 2. 建立两个不同宽度的一维矩形 mask
# ============================================================

# 宽 mask：
# |x| <= 0.5 的地方为 1，其余为 0
# 因此空间宽度为 1.0
mask1 = np.where(np.abs(x) <= 0.5, 1.0, 0.0)

# 窄 mask：
# |x| <= 0.1 的地方为 1，其余为 0
# 因此空间宽度为 0.2
mask2 = np.where(np.abs(x) <= 0.1, 1.0, 0.0)


# ============================================================
# 3. FFT：从空间域转换到频域
# ============================================================

# Fourier Transform：
# 把空间域中的 mask 分解成不同空间频率的成分
#
# 可以暂时理解为：
# “这个 mask 是由哪些不同频率的波组成的？”
F1 = np.fft.fft(mask1)
F2 = np.fft.fft(mask2)


# ============================================================
# 4. fftshift：把零频率移动到频谱中央
# ============================================================

# FFT 默认的频率排列方式不方便观察，
# fftshift() 只是重新排列数据，把 0 频率移动到中间。
#
# 注意：
# fftshift 不会改变频谱的信息，只是改变排列顺序。
F1 = np.fft.fftshift(F1)
F2 = np.fft.fftshift(F2)


# ============================================================
# 5. 计算频谱的 magnitude
# ============================================================

# F1、F2 是复数，里面包含 magnitude（幅值）和 phase（相位）信息。
#
# np.abs() 只取频谱的 magnitude，
# 用来观察“每个空间频率有多强”。
#
# 注意：
# spectrum1 / spectrum2 已经丢失了 phase，
# 所以后面真正做 IFFT 恢复图像时不能只使用它们。
spectrum1 = np.abs(F1)
spectrum2 = np.abs(F2)


# ============================================================
# 6. 归一化频谱
# ============================================================

# 把两个频谱的最大值都变成 1。
# 这样可以更公平地比较“频谱宽度”，
# 而不是比较两个频谱谁的绝对幅值更大。
spectrum1 /= np.max(spectrum1)
spectrum2 /= np.max(spectrum2)


# ============================================================
# 7. 建立真正的空间频率坐标
# ============================================================

# FFT 得到的是一个数组。
# fftfreq() 根据：
#   1. 采样点数量 len(x)
#   2. 空间采样间隔 dx
# 计算每一个 FFT 数据点对应的真实空间频率。
freq = np.fft.fftfreq(len(x), d=dx)

# 因为前面的 F1、F2 使用了 fftshift，
# 所以频率坐标 freq 也必须做同样的 shift，
# 这样 freq[i] 才能和 F1[i] / F2[i] 正确对应。
freq = np.fft.fftshift(freq)


# ============================================================
# 8. 建立一个非常简化的 pupil（光学系统）
# ============================================================

# cutoff 表示这个简化光学系统能够通过的最大空间频率。
#
# |frequency| <= cutoff：允许通过
# |frequency| > cutoff ：完全挡掉
cutoff = 2

# pupil = 1：这个频率能够通过
# pupil = 0：这个频率不能通过
#
# 可以暂时把 pupil 理解成光学系统的“频率大门”。
pupil = np.where(np.abs(freq) <= cutoff, 1.0, 0.0)


# ============================================================
# 9. 为了画图观察：看看哪些频率被 pupil 保留下来
# ============================================================

# spectrum2 是窄 mask 的频谱 magnitude。
#
# pupil 内：
# spectrum2 * 1 = spectrum2  → 保留
#
# pupil 外：
# spectrum2 * 0 = 0          → 删除
#
# 这一变量主要用于画图观察。
filtered_spectrum = spectrum2 * pupil


# ============================================================
# 10. 真正过滤完整的 Fourier Transform
# ============================================================

# 真正恢复图像时不能使用 filtered_spectrum，
# 因为 spectrum2 = abs(F2) 已经丢失了 phase。
#
# 所以必须直接过滤完整的复数 Fourier Transform F2。
F2_filtered = F2 * pupil


# ============================================================
# 11. ifftshift：把频谱恢复成 IFFT 需要的排列方式
# ============================================================

# 前面为了方便观察，我们用 fftshift() 把零频移到了中央。
#
# 在进行 IFFT 之前，需要用 ifftshift()
# 把频率重新排列回 FFT 默认的顺序。
#
# fftshift   ：为了方便人观察
# ifftshift  ：做 IFFT 前恢复原来的排列
F2_filtered = np.fft.ifftshift(F2_filtered)


# ============================================================
# 12. IFFT：从频域重新回到空间域
# ============================================================

# Inverse Fourier Transform：
# 根据过滤后剩余的频率成分重新组成空间图形。
#
# 因为一部分高频已经被 pupil 删除，
# 所以恢复出来的图形不能完全等于原来的 mask2。
image_filtered = np.fft.ifft(F2_filtered)


# ============================================================
# 13. 只保留实数部分
# ============================================================

# 理论上这里应该得到实数图像。
# 但是计算机浮点计算可能产生非常小的虚部误差，
# 例如 0.5 + 1e-16j。
#
# np.real() 取出真正需要的实数部分。
image_filtered = np.real(image_filtered)


# ============================================================
# 14. 空间域比较：原始 mask vs 过滤后的图像
# ============================================================

plt.plot(x, mask2, label="Original mask")
plt.plot(x, image_filtered, label="Filtered image")

plt.xlabel("Position")
plt.ylabel("Amplitude")
plt.title("Original Mask vs Filtered Image")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 15. 频域比较
# ============================================================

plt.figure(figsize=(12, 5))

# 宽 mask 的频谱
plt.plot(freq, spectrum1, label="Wide mask")

# 窄 mask 的频谱
plt.plot(freq, spectrum2, label="Narrow mask")

# 窄 mask 经过 pupil 后能够保留下来的频谱
plt.plot(freq, filtered_spectrum, label="Filtered spectrum")

# pupil：表示光学系统允许哪些空间频率通过
plt.plot(
    freq,
    pupil,
    label="Pupil function",
    linestyle="--",
    color="black"
)

# 只观察中心附近的空间频率
plt.xlim(-20, 20)

plt.xlabel("Spatial frequency")
plt.ylabel("Normalized magnitude")
plt.title("Mask Spectrum and Pupil Filtering")

plt.legend()
plt.grid(True)
plt.show()