import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Basic functions
# =========================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def soft_resist(aerial, threshold=0.5, beta=20):
    return 1.0 / (
        1.0 + np.exp(
            -beta * (aerial - threshold)
        )
    )


# =========================================================
# 2. Spatial grid
# =========================================================

N = 21

x = np.linspace(-1.0, 1.0, N)   # unit: μm
y = np.linspace(-1.0, 1.0, N)

X, Y = np.meshgrid(x, y)

dx = x[1] - x[0]
dy = y[1] - y[0]


# =========================================================
# 3. Target pattern
# =========================================================

target = (
    (np.abs(X) <= 0.30) &
    (np.abs(Y) <= 0.20)
).astype(float)


# =========================================================
# 4. Initial latent variable Z
# =========================================================

# Target inside -> positive
# Target outside -> negative

Z = np.where(
    target == 1,
    2.0,
    -2.0
).astype(float)


# =========================================================
# 5. Frequency grid
# =========================================================

fx = np.fft.fftfreq(
    N,
    d=dx
)

fy = np.fft.fftfreq(
    N,
    d=dy
)

FX, FY = np.meshgrid(
    fx,
    fy
)


# =========================================================
# 6. Optical parameters
# =========================================================

wavelength = 0.193   # μm, 193 nm

NA = 0.5

cutoff = (
    NA / wavelength
)

print(
    "Cutoff spatial frequency =",
    cutoff,
    "1/μm"
)


# =========================================================
# 7. Pupil function
# =========================================================

frequency_radius = np.sqrt(
    FX**2 + FY**2
)

pupil = (
    frequency_radius <= cutoff
).astype(float)


# =========================================================
# 8. Fourier optical forward model
# =========================================================

def forward_fourier(
    Z,
    pupil,
    threshold=0.5,
    beta=20
):

    # -------------------------------------
    # Z -> continuous mask
    # -------------------------------------

    mask = sigmoid(Z)


    # -------------------------------------
    # mask -> frequency domain
    # -------------------------------------

    spectrum = np.fft.fft2(
        mask
    )


    # -------------------------------------
    # pupil filtering
    # -------------------------------------

    filtered_spectrum = (
        spectrum * pupil
    )


    # -------------------------------------
    # frequency domain -> complex field
    # -------------------------------------

    field = np.fft.ifft2(
        filtered_spectrum
    )


    # -------------------------------------
    # complex field -> intensity
    # -------------------------------------

    aerial = (
        np.abs(field) ** 2
    )


    # IMPORTANT:
    # Do NOT use:
    #
    # aerial = aerial / np.max(aerial)
    #
    # during gradient checking.
    #
    # Otherwise the derivative of max()
    # must also be included.


    # -------------------------------------
    # intensity -> soft resist
    # -------------------------------------

    resist = soft_resist(
        aerial,
        threshold=threshold,
        beta=beta
    )


    return (
        mask,
        spectrum,
        filtered_spectrum,
        field,
        aerial,
        resist
    )


# =========================================================
# 9. Loss function
# =========================================================

def loss_fourier(
    Z,
    target,
    pupil,
    threshold=0.5,
    beta=20
):

    (
        mask,
        spectrum,
        filtered_spectrum,
        field,
        aerial,
        resist
    ) = forward_fourier(
        Z,
        pupil,
        threshold,
        beta
    )

    loss = np.mean(
        (resist - target) ** 2
    )

    return loss


# =========================================================
# 10. Backward propagation
# =========================================================

def backward_fourier(
    Z,
    target,
    pupil,
    threshold=0.5,
    beta=20
):

    # =====================================
    # Forward
    # =====================================

    (
        mask,
        spectrum,
        filtered_spectrum,
        field,
        aerial,
        resist
    ) = forward_fourier(
        Z,
        pupil,
        threshold,
        beta
    )


    # =====================================
    # Backward
    # =====================================


    # -------------------------------------
    # 1. Loss -> Resist
    # -------------------------------------

    dL_dR = (
        2.0
        * (resist - target)
        / target.size
    )


    # -------------------------------------
    # 2. Resist -> Intensity
    # -------------------------------------

    dR_dI = (
        beta
        * resist
        * (1.0 - resist)
    )

    dL_dI = (
        dL_dR
        * dR_dI
    )


    # -------------------------------------
    # 3. Intensity -> Complex field
    # -------------------------------------

    dL_dE = (
        2.0
        * dL_dI
        * field
    )


    # -------------------------------------
    # 4. Field -> Mask
    #
    # Optical adjoint:
    #
    # gM =
    # Re[
    # IFFT(
    # conj(P)
    # *
    # FFT(gE)
    # )
    # ]
    # -------------------------------------

    dL_dM_complex = np.fft.ifft2(
        np.conj(pupil)
        *
        np.fft.fft2(
            dL_dE
        )
    )

    dL_dM = np.real(
        dL_dM_complex
    )


    # -------------------------------------
    # 5. Mask -> Z
    # -------------------------------------

    dM_dZ = (
        mask
        * (1.0 - mask)
    )

    dL_dZ = (
        dL_dM
        * dM_dZ
    )


    return dL_dZ


# =========================================================
# 11. Numerical gradient for one pixel
# =========================================================

def numerical_gradient_pixel(
    Z,
    i,
    j,
    target,
    pupil,
    h=1e-5,
    threshold=0.5,
    beta=20
):

    Z_plus = Z.copy()
    Z_minus = Z.copy()

    Z_plus[i, j] += h
    Z_minus[i, j] -= h


    loss_plus = loss_fourier(
        Z_plus,
        target,
        pupil,
        threshold,
        beta
    )

    loss_minus = loss_fourier(
        Z_minus,
        target,
        pupil,
        threshold,
        beta
    )


    grad = (
        loss_plus
        - loss_minus
    ) / (2.0 * h)


    return grad


# =========================================================
# 12. Run forward once
# =========================================================

(
    mask,
    spectrum,
    filtered_spectrum,
    field,
    aerial,
    resist
) = forward_fourier(
    Z,
    pupil
)


current_loss = loss_fourier(
    Z,
    target,
    pupil
)

print(
    "Initial loss =",
    current_loss
)


# =========================================================
# 13. Backward gradient
# =========================================================

grad_backward = backward_fourier(
    Z,
    target,
    pupil
)


# =========================================================
# 14. Gradient check
# =========================================================

test_pixels = [
    (10, 10),   # center
    (10, 7),    # near target edge
    (8, 10),    # near target edge
    (5, 5),     # outside
    (15, 15)    # outside
]


print("\nGradient check:")
print("=" * 60)


for i, j in test_pixels:

    grad_num = numerical_gradient_pixel(
        Z,
        i,
        j,
        target,
        pupil,
        h=1e-5
    )

    grad_back = (
        grad_backward[i, j]
    )

    relative_error = (
        np.abs(
            grad_num
            - grad_back
        )
        /
        (
            np.abs(grad_num)
            +
            np.abs(grad_back)
            +
            1e-12
        )
    )


    print(
        f"Pixel ({i}, {j})"
    )

    print(
        "numerical =",
        grad_num
    )

    print(
        "backward  =",
        grad_back
    )

    print(
        "relative error =",
        relative_error
    )

    print("-" * 60)



# =========================================================
# 16. Fourier ILT optimization
# =========================================================

# Save the initial state
Z_initial = Z.copy()

(
    mask_initial,
    spectrum_initial,
    filtered_initial,
    field_initial,
    aerial_initial,
    resist_initial
) = forward_fourier(
    Z_initial,
    pupil
)


# Optimization parameters
num_iterations = 100

learning_rate = 5.0

loss_history = []


# Start from the initial Z
Z_opt = Z_initial.copy()


for iteration in range(
    num_iterations
):

    # -------------------------------------
    # 1. Calculate current loss
    # -------------------------------------

    current_loss = loss_fourier(
        Z_opt,
        target,
        pupil
    )

    loss_history.append(
        current_loss
    )


    # -------------------------------------
    # 2. Calculate gradient
    # -------------------------------------

    grad = backward_fourier(
        Z_opt,
        target,
        pupil
    )


    # -------------------------------------
    # 3. Gradient descent update
    # -------------------------------------

    Z_opt = (
        Z_opt
        -
        learning_rate
        * grad
    )


    # -------------------------------------
    # 4. Print progress
    # -------------------------------------

    if (
        iteration % 10 == 0
        or
        iteration == num_iterations - 1
    ):

        print(
            f"Iteration {iteration:3d}, "
            f"Loss = {current_loss:.8f}"
        )


# =========================================================
# 17. Final result
# =========================================================

(
    mask_final,
    spectrum_final,
    filtered_final,
    field_final,
    aerial_final,
    resist_final
) = forward_fourier(
    Z_opt,
    pupil
)


final_loss = loss_fourier(
    Z_opt,
    target,
    pupil
)


print()
print(
    "Initial loss =",
    loss_history[0]
)

print(
    "Final loss   =",
    final_loss
)


# =========================================================
# 18. Plot optimization results
# =========================================================

plt.figure(
    figsize=(12, 10)
)


# -----------------------------------------
# Target
# -----------------------------------------

plt.subplot(
    3,
    3,
    1
)

plt.imshow(
    target,
    extent=[
        x.min(),
        x.max(),
        y.min(),
        y.max()
    ],
    origin="lower",
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "Target"
)

plt.colorbar()


# -----------------------------------------
# Initial mask
# -----------------------------------------

plt.subplot(
    3,
    3,
    2
)

plt.imshow(
    mask_initial,
    extent=[
        x.min(),
        x.max(),
        y.min(),
        y.max()
    ],
    origin="lower",
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "Initial Mask"
)

plt.colorbar()


# -----------------------------------------
# Optimized mask
# -----------------------------------------

plt.subplot(
    3,
    3,
    3
)

plt.imshow(
    mask_final,
    extent=[
        x.min(),
        x.max(),
        y.min(),
        y.max()
    ],
    origin="lower",
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "Optimized Mask"
)

plt.colorbar()


# -----------------------------------------
# Initial aerial image
# -----------------------------------------

plt.subplot(
    3,
    3,
    4
)

plt.imshow(
    aerial_initial,
    extent=[
        x.min(),
        x.max(),
        y.min(),
        y.max()
    ],
    origin="lower"
)

plt.title(
    "Initial Aerial"
)

plt.colorbar()


# -----------------------------------------
# Final aerial image
# -----------------------------------------

plt.subplot(
    3,
    3,
    5
)

plt.imshow(
    aerial_final,
    extent=[
        x.min(),
        x.max(),
        y.min(),
        y.max()
    ],
    origin="lower"
)

plt.title(
    "Optimized Aerial"
)

plt.colorbar()


# -----------------------------------------
# Mask change
# -----------------------------------------

plt.subplot(
    3,
    3,
    6
)

plt.imshow(
    mask_final
    -
    mask_initial,
    extent=[
        x.min(),
        x.max(),
        y.min(),
        y.max()
    ],
    origin="lower"
)

plt.title(
    "Mask Change"
)

plt.colorbar()


# -----------------------------------------
# Initial resist
# -----------------------------------------

plt.subplot(
    3,
    3,
    7
)

plt.imshow(
    resist_initial,
    extent=[
        x.min(),
        x.max(),
        y.min(),
        y.max()
    ],
    origin="lower",
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "Initial Resist"
)

plt.colorbar()


# -----------------------------------------
# Final resist
# -----------------------------------------

plt.subplot(
    3,
    3,
    8
)

plt.imshow(
    resist_final,
    extent=[
        x.min(),
        x.max(),
        y.min(),
        y.max()
    ],
    origin="lower",
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "Optimized Resist"
)

plt.colorbar()


# -----------------------------------------
# Loss curve
# -----------------------------------------

plt.subplot(
    3,
    3,
    9
)

plt.plot(
    loss_history
)

plt.xlabel(
    "Iteration"
)

plt.ylabel(
    "Print Loss"
)

plt.title(
    "Loss Curve"
)

plt.grid()


plt.tight_layout()

plt.show()