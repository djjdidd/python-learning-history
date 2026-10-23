import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# =========================================================
# 1. Build 2D grid
# =========================================================

N = 21

x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)

X, Y = np.meshgrid(x, y)


# =========================================================
# 2. Define target pattern
# =========================================================

target = (
    (np.abs(X) <= 0.3) &
    (np.abs(Y) <= 0.2)
).astype(float)


# =========================================================
# 3. Define optical PSF
# =========================================================

sigma = 0.12

psf = np.exp(
    -(X**2 + Y**2) / (2 * sigma**2)
)

psf = psf / np.sum(psf)


# =========================================================
# 4. Sigmoid
#    Z -> continuous mask
# =========================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# =========================================================
# 5. Soft resist model
# =========================================================

def soft_resist(aerial, threshold=0.5, beta=20):

    return 1 / (
        1 + np.exp(
            -beta * (aerial - threshold)
        )
    )


# =========================================================
# 6. Forward model
#
#    Z
#    ↓
#    Mask
#    ↓
#    Aerial image
#    ↓
#    Resist
# =========================================================

def forward(Z):

    mask = sigmoid(Z)

    aerial = signal.fftconvolve(
        mask,
        psf,
        mode="same"
    )

    resist = soft_resist(
        aerial,
        threshold=0.5,
        beta=20
    )

    return mask, aerial, resist


# =========================================================
# 7. Printing loss
# =========================================================

def print_loss_function(resist):

    return np.mean(
        (resist - target) ** 2
    )


# =========================================================
# 8. TV loss
#
#    Penalize abrupt changes between neighboring pixels
# =========================================================

def tv_loss(mask, eps=1e-6):

    dx = mask[:, 1:] - mask[:, :-1]
    dy = mask[1:, :] - mask[:-1, :]

    tv_x = np.mean(
        np.sqrt(dx**2 + eps)
    )

    tv_y = np.mean(
        np.sqrt(dy**2 + eps)
    )

    return tv_x + tv_y


# =========================================================
# 9. TV gradient with respect to mask
# =========================================================

def tv_gradient(mask, eps=1e-6):

    dx = mask[:, 1:] - mask[:, :-1]
    dy = mask[1:, :] - mask[:-1, :]

    grad = np.zeros_like(mask)

    gx = dx / np.sqrt(dx**2 + eps)
    gx = gx / dx.size

    grad[:, 1:] += gx
    grad[:, :-1] -= gx

    gy = dy / np.sqrt(dy**2 + eps)
    gy = gy / dy.size

    grad[1:, :] += gy
    grad[:-1, :] -= gy

    return grad


# =========================================================
# 10. Binarization loss
#
#     Small near 0 or 1
#     Large near 0.5
# =========================================================

def binarization_loss(mask):

    return np.mean(
        mask * (1 - mask)
    )


# =========================================================
# 11. Binarization gradient with respect to mask
#
#     d/dM [M(1-M)] = 1 - 2M
# =========================================================

def binarization_gradient(mask):

    grad_bin_mask = (
        1 - 2 * mask
    ) / mask.size

    return grad_bin_mask


# =========================================================
# 12. Printing gradient / backward
# =========================================================

def backward(
    Z,
    target,
    psf,
    threshold=0.5,
    beta=20
):

    # -----------------------------
    # Forward
    # -----------------------------

    mask = sigmoid(Z)

    aerial = signal.fftconvolve(
        mask,
        psf,
        mode="same"
    )

    resist = soft_resist(
        aerial,
        threshold=threshold,
        beta=beta
    )

    # -----------------------------
    # Backward
    # -----------------------------

    # dL / dR
    dL_dR = (
        2 * (resist - target)
        / target.size
    )

    # dR / dA
    dR_dA = (
        beta
        * resist
        * (1 - resist)
    )

    # dL / dA
    dL_dA = (
        dL_dR
        * dR_dA
    )

    # Aerial -> Mask
    #
    # Use correlation as adjoint
    # of convolution
    dL_dM = signal.correlate(
        dL_dA,
        psf,
        mode="same"
    )

    # dM / dZ
    dM_dZ = (
        mask
        * (1 - mask)
    )

    # dL / dZ
    dL_dZ = (
        dL_dM
        * dM_dZ
    )

    return dL_dZ


# =========================================================
# 13. Initialize Z
#
# target area -> positive
# outside      -> negative
#
# This gives a reasonable initial mask
# =========================================================

initial_Z = np.where(
    target == 1,
    2.0,
    -2.0
)

Z = initial_Z.copy()


# =========================================================
# 14. Optimization settings
# =========================================================

learning_rate = 5.0

lambda_tv = 0.1
lambda_bin_history = []
weighted_bin_loss_history = []
bin_loss_history = []



num_iterations = 200

for iteration in range(num_iterations):

    if iteration < 50:
        lambda_bin = 0.01

    elif iteration < 100:
        lambda_bin = 0.1

    else:
        lambda_bin = 0.5

    mask, aerial, resist = forward(Z)

    print_loss = print_loss_function(resist)
    reg_tv = tv_loss(mask)
    reg_bin = binarization_loss(mask)
    bin_loss_history.append(reg_bin)
    grad_print = backward(
        Z,
        target,
        psf,
        threshold=0.5,
        beta=20
    )

    grad_tv_mask = tv_gradient(mask)

    grad_tv_Z = (
        grad_tv_mask
        * mask
        * (1 - mask)
    )

    grad_bin_mask = binarization_gradient(mask)

    grad_bin_Z = (
        grad_bin_mask
        * mask
        * (1 - mask)
    )

    grad_total = (
        grad_print
        + lambda_tv * grad_tv_Z
        + lambda_bin * grad_bin_Z
    )

    Z = Z - learning_rate * grad_total
    lambda_bin_history.append(lambda_bin)

    weighted_bin_loss_history.append(
    lambda_bin * reg_bin
)
plt.plot(
    bin_loss_history,
    label="Raw Binarization Loss"
)

plt.plot(
    weighted_bin_loss_history,
    label="Weighted Binarization Loss"
)

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()