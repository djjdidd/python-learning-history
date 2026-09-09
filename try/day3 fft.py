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
cutoff = 2

spectrum1 /= np.max(spectrum1)
spectrum2 /= np.max(spectrum2)

freq = np.fft.fftfreq(len(x), d=dx)
freq = np.fft.fftshift(freq)
pupil = np.where(np.abs(freq) <= cutoff, 1.0, 0.0)
filtered_spectrum = spectrum2 * pupil

F2_filtered = F2 * pupil
F2_filtered = np.fft.ifftshift(F2_filtered)
image_filtered = np.fft.ifft(F2_filtered)
image_filtered = np.real(image_filtered)
plt.plot(x, mask2, label="Original mask")
plt.plot(x, image_filtered, label="Filtered image")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12,5))
plt.plot(freq, spectrum1, label="Wide mask")
plt.plot(freq, spectrum2, label="Narrow mask")
plt.plot(freq, filtered_spectrum, label="Filtered spectrum")
plt.plot(freq, pupil, label="Pupil function", linestyle='--', color='black')
plt.xlim(-20, 20)
plt.xlabel("Spatial frequency")
plt.ylabel("Normalized magnitude")
plt.legend()
plt.grid(True)
plt.show()