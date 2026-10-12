import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# =========================================================
# 1. Build a small 2D coordinate grid
# =========================================================

N = 20

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
# 4. Sigmoid function
#    Convert latent variable Z into continuous mask
# =========================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# =========================================================
# 5. Soft resist model
# =========================================================

def soft_resist(aerial, threshold=0.5, beta=20):

    resist = 1 / (
        1 + np.exp(
            -beta * (aerial - threshold)
        )
    )

    return resist


# =========================================================
# 6. Forward imaging model
#
#    Z
#    ↓
#    continuous mask
#    ↓
#    optical convolution
#    ↓
#    aerial image
#    ↓
#    soft resist
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
# 7. Loss function
#
#    Compare resist result with target
# =========================================================

def loss_function(Z):

    _, _, resist = forward(Z)

    loss = np.mean(
        (resist - target) ** 2
    )

    return loss


# =========================================================
# 8. Numerical gradient map
#
#    For every pixel:
#
#           L(Z+h) - L(Z-h)
#    grad = ----------------
#                  2h
# =========================================================

def numerical_gradient_map(Z, h=1e-3):

    grad = np.zeros_like(Z)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):

            Z_plus = Z.copy()
            Z_minus = Z.copy()

            Z_plus[i, j] += h
            Z_minus[i, j] -= h

            loss_plus = loss_function(Z_plus)
            loss_minus = loss_function(Z_minus)

            grad[i, j] = (
                loss_plus - loss_minus
            ) / (2 * h)

    return grad


# =========================================================
# 9. Initialize Z
#
#    Z = 0
#    sigmoid(0) = 0.5
#
#    Therefore initial mask is 0.5 everywhere
# =========================================================

Z = np.where(
    target == 1,
    2.0,
    -2.0
)


# =========================================================
# 10. Calculate initial result
# =========================================================

initial_mask, initial_aerial, initial_resist = forward(Z)

initial_loss = loss_function(Z)

print("Initial loss:", initial_loss)


# =========================================================
# 11. Calculate gradient map
# =========================================================

grad = numerical_gradient_map(
    Z,
    h=1e-3
)


# =========================================================
# 12. Perform one gradient descent update
#
#    Z_new = Z - learning_rate * gradient
# =========================================================

learning_rate = 5.0

Z_new = Z - learning_rate * grad


# =========================================================
# 13. Calculate result after one update
# =========================================================

new_mask, new_aerial, new_resist = forward(Z_new)

new_loss = loss_function(Z_new)

print("New loss:", new_loss)


# =========================================================
# 14. Show target, initial mask, gradient,
#     updated mask, and resist result
# =========================================================

plt.figure(figsize=(15, 8))


plt.subplot(2, 3, 1)
plt.imshow(
    target,
    origin="lower",
    cmap="gray"
)
plt.title("Target")
plt.colorbar()


plt.subplot(2, 3, 2)
plt.imshow(
    initial_mask,
    origin="lower",
    cmap="gray"
)
plt.title("Initial Continuous Mask")
plt.colorbar()


plt.subplot(2, 3, 3)
plt.imshow(
    grad,
    origin="lower",
    cmap="coolwarm"
)
plt.title("Gradient Map")
plt.colorbar()


plt.subplot(2, 3, 4)
plt.imshow(
    new_mask,
    origin="lower",
    cmap="gray"
)
plt.title("Mask After One Update")
plt.colorbar()


plt.subplot(2, 3, 5)
plt.imshow(
    initial_resist,
    origin="lower",
    cmap="gray"
)
plt.title("Initial Resist")
plt.colorbar()


plt.subplot(2, 3, 6)
plt.imshow(
    new_resist,
    origin="lower",
    cmap="gray"
)
plt.title("Resist After One Update")
plt.colorbar()


plt.tight_layout()
plt.show()