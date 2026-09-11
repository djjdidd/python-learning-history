import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import optimize
x=np.linspace(-10,10,1000)
def make_mask(mask_width):
    mask=np.where((x>(-mask_width/2))&(x<(mask_width/2)),1,0)
    return mask
sigma = 0.12

psf = np.exp(-x**2 / (2 * sigma**2))
psf = psf / np.sum(psf)
dx = x[1] - x[0]

def measure_width(binary):
    return np.sum(binary) * dx
def optical_image(mask_width):

    mask = make_mask(mask_width)

    aerial_image = signal.fftconvolve(
        mask,
        psf,
        mode="same"
    )

    return aerial_image
threshold = 0.40

def print_pattern(mask_width):

    aerial_image = optical_image(mask_width)

    # 这一行你自己写
    printed = np.where(aerial_image > threshold, 1, 0)

    return printed
target_width = 0.40

def objective(mask_width_array):

    mask_width = mask_width_array[0]

    if mask_width <= 0:
        return 1e6

    printed = print_pattern(mask_width)

    printed_width = measure_width(printed)

    return (printed_width - target_width) ** 2
result = optimize.minimize(
    objective,
    x0=np.array([0.40]),
    method="Nelder-Mead"
)

best_mask_width = result.x[0]

best_pattern = print_pattern(best_mask_width)
best_printed_width = measure_width(best_pattern)

original_mask_width = 0.40
optimized_mask_width = best_mask_width
original_mask = make_mask(original_mask_width)
original_printed = print_pattern(original_mask_width)

optimized_mask = make_mask(optimized_mask_width)
optimized_printed = print_pattern(optimized_mask_width)
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(x, original_mask, label="Original mask")
plt.plot(x, original_printed, label="Original printed")
plt.xlim(-1, 1)
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(x, optimized_mask, label="Optimized mask")
plt.plot(x, optimized_printed, label="Optimized printed")
plt.xlim(-1, 1)
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()