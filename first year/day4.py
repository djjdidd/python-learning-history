import numpy as np

import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 2000)

dx = x[1] - x[0]

mask1 = np.where(np.abs(x) <= 0.5, 1.0, 0.0)

mask2 = np.where(np.abs(x) <= 0.1, 1.0, 0.0)

F1 = np.fft.fft(mask1)

F2 = np.fft.fft(mask2)

F1 = np.fft.fftshift(F1)

F2 = np.fft.fftshift(F2)

spectrum1 = np.abs(F1)

spectrum2 = np.abs(F2)
freq = np.fft.fftfreq(len(x), d=dx)
freq = np.fft.fftshift(freq)
cutoff_list = [1, 5, 10, 20, 50]
plt.figure()

plt.plot(x, mask2, label="Original mask")

for cutoff in cutoff_list:

    pupil = np.where(np.abs(freq) <= cutoff, 1.0, 0.0)

    F2_filtered = F2 * pupil

    F2_filtered = np.fft.ifftshift(F2_filtered)

    image_filtered = np.fft.ifft(F2_filtered)

    image_filtered = np.real(image_filtered)

    plt.plot(x, image_filtered, label=f"cutoff={cutoff}")

plt.xlabel("Position")
plt.ylabel("Amplitude")
plt.title("Effect of cutoff on reconstructed image")
plt.legend()
plt.grid(True)
plt.show()
