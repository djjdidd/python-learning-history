import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
x = np.linspace(-1, 1, 500)
y = np.linspace(-1, 1, 500)

X, Y = np.meshgrid(x, y)
width_x = 0.6
width_y = 0.4
width_x = 0.6
width_y = 0.4

mask_2d = np.where(
    (np.abs(X) <= width_x / 2) &
    (np.abs(Y) <= width_y / 2),
    1.0,
    0.0
)
serif_size = 0.1
serif1 = (
    (np.abs(X - 0.3) <= serif_size / 2) &
    (np.abs(Y - 0.2) <= serif_size / 2)
)

serif2 = (
    (np.abs(X + 0.3) <= serif_size / 2) &
    (np.abs(Y - 0.2) <= serif_size / 2)
)

serif3 = (
    (np.abs(X - 0.3) <= serif_size / 2) &
    (np.abs(Y + 0.2) <= serif_size / 2)
)

serif4 = (
    (np.abs(X + 0.3) <= serif_size / 2) &
    (np.abs(Y + 0.2) <= serif_size / 2)
)
mask_2d = np.where(
    (np.abs(X) <= width_x / 2) &
    (np.abs(Y) <= width_y / 2),
    1.0,
    0.0
)
mask_serif = np.where(
    (mask_2d == 1) |
    serif1 |
    serif2 |
    serif3 |
    serif4,
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

printed_original = np.where(
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
plt.imshow(printed_original, origin="lower")
plt.title("Printed pattern")

plt.tight_layout()
plt.show()
aerial_serif = signal.fftconvolve(
    mask_serif,
    psf_2d,
    mode="same"
)

printed_serif = np.where(
    aerial_serif >= threshold,
    1.0,
    0.0
)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(mask_2d, origin="lower")
plt.title("Original mask")

plt.subplot(1, 3, 2)
plt.imshow(mask_serif, origin="lower")
plt.title("Mask with serif")

plt.subplot(1, 3, 3)
plt.imshow(printed_serif, origin="lower")
plt.title("Printed with serif")

plt.tight_layout()
plt.show()
target = mask_2d
error_original = np.sum(
    (printed_original - target) ** 2
)

error_serif = np.sum(
    (printed_serif - target) ** 2
)

print("Original error:", error_original)
print("Serif error:", error_serif)