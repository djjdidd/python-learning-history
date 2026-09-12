import matplotlib.pyplot as plt
from scipy import signal, optimize
import numpy as np
x = np.linspace(-1, 1, 500)
y = np.linspace(-1, 1, 500)

X, Y = np.meshgrid(x, y)
width_x = 0.6
width_y = 0.4
width_x = 0.6
width_y = 0.4

def soft_resist(aerial, threshold, beta=30):

    resist = 1 / (
        1 + np.exp(-beta * (aerial - threshold))
    )

    return resist


mask_2d = np.where(
    (np.abs(X) <= width_x / 2) &
    (np.abs(Y) <= width_y / 2),
    1.0,
    0.0
)
psf_2d = np.exp(
    -(X**2 + Y**2) / (2 * 0.08**2)
)
psf_2d = psf_2d / np.sum(psf_2d)
threshold = 0.5
target = mask_2d

serif_size_list = np.linspace(0.10, 0.20, 30)
hard_error_list = []
soft_error_list = []
def serif_objective(size_array):

    serif_size = size_array[0]

    if serif_size <= 0:
        return 1e6

    mask_serif = make_serif_mask(serif_size)

    aerial = signal.fftconvolve(
        mask_serif,
        psf_2d,
        mode="same"
    )

    resist = soft_resist(aerial, threshold,beta=30)

    error = np.sum(
        (resist - target) ** 2
    )

    return error


def make_serif_mask(serif_size):
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
    mask_serif = np.where(
        (mask_2d == 1) |
        serif1 |
        serif2 |
        serif3 |
        serif4,
        1.0,
        0.0
    )
    return mask_serif
for serif_size in serif_size_list:

    mask_serif = make_serif_mask(serif_size)

    aerial = signal.fftconvolve(
        mask_serif,
        psf_2d,
        mode="same"
    )

    printed = np.where(
        aerial >= threshold,
        1.0,
        0.0
    )
    resist = soft_resist(aerial, threshold, beta=30)

    hard_error = np.sum(
        (printed - target) ** 2
    )
    soft_error_list.append(np.sum((resist - target) ** 2))
    hard_error_list.append(hard_error)

error_array = np.array(soft_error_list)

best_index = np.argmin(error_array)

best_serif_size_scan = serif_size_list[best_index]
best_error_scan = error_array[best_index]


result = optimize.minimize(
    serif_objective,
    x0=np.array([0.08]),
    method="Nelder-Mead"
)

best_serif_size_opt = result.x[0]
best_error_opt = result.fun


print("Best serif size from scan:", best_serif_size_scan)
print("Best error from scan:", best_error_scan)
print("Best serif size from optimization:", best_serif_size_opt)
print("Best error from optimization:", best_error_opt)
