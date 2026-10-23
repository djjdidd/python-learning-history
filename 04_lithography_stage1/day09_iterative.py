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

initial_Z = Z.copy()

initial_mask, initial_aerial, initial_resist = forward(initial_Z)
loss_history = []

learning_rate = 5.0
num_iterations = 50

for iteration in range(num_iterations):

    # 1. Forward
    mask, aerial, resist = forward(Z)

    # 2. Calculate loss
    loss = np.mean(
        (resist - target) ** 2
    )

    loss_history.append(loss)

    # 3. Backward
    grad = backward(
        Z,
        target,
        psf,
        threshold=0.5,
        beta=20
    )

    # 4. Update Z
    Z = Z - learning_rate * grad

final_mask, final_aerial, final_resist = forward(Z)
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(target, origin="lower", cmap="gray")
plt.title("Target")
plt.colorbar()

plt.subplot(2, 3, 2)
plt.imshow(initial_mask, origin="lower", cmap="gray")
plt.title("Initial Mask")
plt.colorbar()

plt.subplot(2, 3, 3)
plt.imshow(final_mask, origin="lower", cmap="gray")
plt.title("Optimized Mask")
plt.colorbar()

plt.subplot(2, 3, 4)
plt.imshow(initial_resist, origin="lower", cmap="gray")
plt.title("Initial Resist")
plt.colorbar()

plt.subplot(2, 3, 5)
plt.imshow(final_resist, origin="lower", cmap="gray")
plt.title("Optimized Resist")
plt.colorbar()

plt.subplot(2, 3, 6)
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss History")
plt.grid()

plt.tight_layout()
plt.show()
mask_change = final_mask - initial_mask

plt.figure(figsize=(6, 5))

plt.imshow(
    mask_change,
    origin="lower",
    cmap="coolwarm"
)

plt.colorbar()
plt.title("Mask Change")

plt.show()