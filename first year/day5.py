import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 2000)
dx = x[1] - x[0]

period_list = [0.4, 0.8, 1.0, 2.0]

cutoff = 9


# ==========================================
# 图1：比较不同 period 的频谱
# ==========================================

plt.figure(figsize=(15, 8))

for i, period in enumerate(period_list):

    # 保持 duty cycle = 50%
    line_width = period / 2

    position_in_period = np.mod(x, period)

    period_mask = np.where(
        position_in_period < line_width,
        1.0,
        0.0
    )

    # 空间域 -> 频域
    F = np.fft.fft(period_mask)
    F_shifted = np.fft.fftshift(F)

    # magnitude spectrum
    F_abs = np.abs(F_shifted)
    F_abs = F_abs / np.max(F_abs)

    # frequency axis
    freq = np.fft.fftfreq(len(x), d=dx)
    freq_shifted = np.fft.fftshift(freq)

    plt.subplot(2, 2, i+1)

    plt.plot(freq_shifted, F_abs)

    plt.xlim(-10, 10)
    plt.ylim(0, 1.1)

    plt.title(f"Period = {period}")
    plt.xlabel("Spatial Frequency")
    plt.ylabel("Normalized Magnitude")

    plt.grid(True)

plt.tight_layout()
plt.show()


# ==========================================
# 图2：比较不同 period 经过 pupil 后的成像
# ==========================================

plt.figure(figsize=(15, 8))

for i, period in enumerate(period_list):

    line_width = period / 2

    position_in_period = np.mod(x, period)

    period_mask = np.where(
        position_in_period < line_width,
        1.0,
        0.0
    )

    F = np.fft.fft(period_mask)
    F_shifted = np.fft.fftshift(F)

    freq = np.fft.fftfreq(len(x), d=dx)
    freq_shifted = np.fft.fftshift(freq)

    # pupil
    pupil = np.where(
        np.abs(freq_shifted) <= cutoff,
        1.0,
        0.0
    )

    # frequency filtering
    F_filtered = F_shifted * pupil

    # frequency -> spatial domain
    F_filtered = np.fft.ifftshift(F_filtered)
    image_filtered = np.fft.ifft(F_filtered)
    image_filtered = np.real(image_filtered)

    plt.subplot(2, 2, i+1)

    plt.plot(x, period_mask, label="Mask")
    plt.plot(x, image_filtered, label="Filtered image")

    plt.xlim(-2, 2)
    plt.title(f"Period = {period}")

    plt.xlabel("Position")
    plt.ylabel("Amplitude")

    plt.legend()
    plt.grid(True)
    intensity = np.abs(image_filtered) ** 2
    

plt.tight_layout()
plt.show()