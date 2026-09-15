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
# 3. Define PSF
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
#    mask
#    ↓
#    aerial image
#    ↓
#    resist
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
# =========================================================

def loss_function(Z):

    _, _, resist = forward(Z)

    loss = np.mean(
        (resist - target) ** 2
    )

    return loss


# =========================================================
# 8. Backward / adjoint gradient
# =========================================================

def backward(Z, target, psf, threshold=0.5, beta=20):

    # ---------- Forward ----------
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

    # ---------- Backward ----------

    # Step 1:
    # dL / dR
    dL_dR = 2 * (resist - target) / target.size

    # Step 2:
    # dR / dA
    dR_dA = beta * resist * (1 - resist)

    # Chain rule:
    # dL / dA
    dL_dA = dL_dR * dR_dA

    # Step 3:
    # dA / dM
    # Backward through convolution
    dL_dM = signal.correlate(
    dL_dA,
    psf,
    mode="same"
)

    # Step 4:
    # dM / dZ
    dM_dZ = mask * (1 - mask)

    # Final gradient:
    # dL / dZ
    dL_dZ = dL_dM * dM_dZ

    return dL_dZ


# =========================================================
# 9. Initialize Z
#
# Better than all zeros:
# target area -> positive
# outside -> negative
# =========================================================

Z = np.where(
    target == 1,
    2.0,
    -2.0
)


# =========================================================
# 10. Calculate backward gradient map
# =========================================================

grad_backward = backward(
    Z,
    target,
    psf,
    threshold=0.5,
    beta=20
)


# =========================================================
# 11. Gradient check
#
# Compare:
# numerical finite-difference gradient
# VS
# backward gradient
# =========================================================

h = 1e-4

check_points = [
    (10, 10),
    (8, 10),
    (5, 5),
    (15, 15)
]


print("========== Gradient Check ==========")

for i, j in check_points:

    # Copy Z
    Z_plus = Z.copy()
    Z_minus = Z.copy()

    # Perturb only one pixel
    Z_plus[i, j] += h
    Z_minus[i, j] -= h

    # Calculate loss
    loss_plus = loss_function(Z_plus)
    loss_minus = loss_function(Z_minus)

    # Numerical gradient
    grad_numerical = (
        loss_plus - loss_minus
    ) / (2 * h)

    # Backward gradient
    grad_adjoint = grad_backward[i, j]

    # Relative error
    relative_error = abs(
        grad_numerical - grad_adjoint
    ) / (
        abs(grad_numerical)
        + abs(grad_adjoint)
        + 1e-12
    )

    print()
    print("Pixel:", (i, j))
    print("Numerical gradient:", grad_numerical)
    print("Backward gradient :", grad_adjoint)
    print("Relative error    :", relative_error)


# =========================================================
# 12. Visualize backward gradient map
# =========================================================

plt.figure(figsize=(6, 5))

plt.imshow(
    grad_backward,
    origin="lower",
    cmap="coolwarm"
)

plt.colorbar()
plt.title("Backward Gradient Map")
plt.xlabel("x pixel")
plt.ylabel("y pixel")

plt.tight_layout()
plt.show()