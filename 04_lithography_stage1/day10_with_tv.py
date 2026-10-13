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
print_loss_history = []
tv_loss_history = []

def tv_loss(mask, eps=1e-6):

    dx = mask[:, 1:] - mask[:, :-1]
    dy = mask[1:, :] - mask[:-1, :]

    tv_x = np.mean(np.sqrt(dx**2 + eps))
    tv_y = np.mean(np.sqrt(dy**2 + eps))

    return tv_x + tv_y

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

learning_rate = 5.0
lambda_tv = 0.01
num_iterations = 50

for iteration in range(num_iterations):

    # ==========================================
    # 1. Forward
    # ==========================================

    mask, aerial, resist = forward(Z)

    # ==========================================
    # 2. Printing loss
    # ==========================================

    print_loss = np.mean(
        (resist - target) ** 2
    )

    # ==========================================
    # 3. TV regularization loss
    # ==========================================

    reg_loss = tv_loss(mask)

    # ==========================================
    # 4. Total loss
    # ==========================================

    total_loss = (
        print_loss
        + lambda_tv * reg_loss
    )

    # Save history
    print_loss_history.append(print_loss)
    tv_loss_history.append(reg_loss)
    loss_history.append(total_loss)

    # ==========================================
    # 5. Printing gradient
    # ==========================================

    grad_print = backward(
        Z,
        target,
        psf,
        threshold=0.5,
        beta=20
    )

    # ==========================================
    # 6. TV gradient
    # ==========================================

    grad_tv_mask = tv_gradient(mask)

    grad_tv_Z = (
        grad_tv_mask
        * mask
        * (1 - mask)
    )

    # ==========================================
    # 7. Total gradient
    # ==========================================

    grad_total = (
        grad_print
        + lambda_tv * grad_tv_Z
    )

    # ==========================================
    # 8. Update Z
    # ==========================================

    Z = Z - learning_rate * grad_total

    print(
        iteration,
        "total =", total_loss,
        "print =", print_loss,
        "TV =", reg_loss
    )
plt.figure(figsize=(12, 8))
def run_optimization(Z_start, lambda_tv, num_iterations=50):

    Z = Z_start.copy()

    learning_rate = 5.0

    for iteration in range(num_iterations):

        # 1. Forward
        mask, aerial, resist = forward(Z)

        # 2. Print loss
        print_loss = np.mean(
            (resist - target) ** 2
        )

        # 3. TV loss
        reg_loss = tv_loss(mask)

        # 4. Print gradient
        grad_print = backward(
            Z,
            target,
            psf,
            threshold=0.5,
            beta=20
        )

        # 5. TV gradient
        grad_tv_mask = tv_gradient(mask)

        grad_tv_Z = (
            grad_tv_mask
            * mask
            * (1 - mask)
        )

        # 6. Total gradient
        grad_total = (
            grad_print
            + lambda_tv * grad_tv_Z
        )

        # 7. Update
        Z = Z - learning_rate * grad_total

    # Final result
    final_mask, _, final_resist = forward(Z)

    final_print_loss = np.mean(
        (final_resist - target) ** 2
    )

    final_tv_loss = tv_loss(final_mask)

    return (
        final_mask,
        final_resist,
        final_print_loss,
        final_tv_loss
    )
lambda_list = [0.0, 0.1, 0.5]

for lambda_tv in lambda_list:

    mask_result, resist_result, print_loss_result, tv_loss_result = run_optimization(
        initial_Z,
        lambda_tv=lambda_tv
    )

    print()
    print("lambda_tv =", lambda_tv)
    print("Print loss =", print_loss_result)
    print("TV loss =", tv_loss_result)
    plt.subplot(1, len(lambda_list), lambda_list.index(lambda_tv) + 1)
    plt.imshow(mask_result, cmap="gray", vmin=0, vmax=1)
    plt.title(f"λ = {lambda_tv}")
plt.suptitle("Optimized Masks for Different TV Regularization Strengths")
plt.tight_layout()
plt.show()