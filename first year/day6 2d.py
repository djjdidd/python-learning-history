import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
x = np.linspace(-1, 1, 500)
y = np.linspace(-1, 1, 500)

X, Y = np.meshgrid(x, y)
width_x = 0.6
width_y = 0.4

mask_2d = np.where(
    (np.abs(X) <= width_x / 2) &
    (np.abs(Y) <= width_y / 2),
    1.0,
    0.0
)
sigma = 0.08

psf_2d = np.exp(
    -(X**2 + Y**2) / (2 * sigma**2)
)

psf_2d = psf_2d / np.sum(psf_2d)
aerial_2d = signal.fftconvolve(
    mask_2d,
    psf_2d,
    mode="same"
)
threshold = 0.5

printed_2d = np.where(
    aerial_2d >= threshold,
    1.0,
    0.0
)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(mask_2d, origin="lower")
plt.title("Mask")

plt.subplot(1, 3, 2)
plt.imshow(aerial_2d, origin="lower")
plt.title("Aerial image")

plt.subplot(1, 3, 3)
plt.imshow(printed_2d, origin="lower")
plt.title("Printed pattern")

plt.tight_layout()
plt.show()