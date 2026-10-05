import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Basic grid
# =========================================================

N = 128
L = 2.0
dx = L / N

x = np.arange(N) * dx - L / 2
y = np.arange(N) * dx - L / 2

X, Y = np.meshgrid(x, y)


# =========================================================
# 2. Target pattern
# =========================================================

target_width_x = 0.60
target_width_y = 0.40

target = (
    (np.abs(X) <= target_width_x / 2)
    &
    (np.abs(Y) <= target_width_y / 2)
).astype(float)


# =========================================================
# 3. Initial mask -> Z
# =========================================================

initial_mask = target.copy()

# Avoid exact 0 and 1
eps = 0.10

mask_clipped = np.clip(
    initial_mask,
    eps,
    1.0 - eps
)

# Inverse sigmoid / logit
Z = np.log(
    mask_clipped
    /
    (1.0 - mask_clipped)
)


# =========================================================
# 4. Sigmoid
# =========================================================

def sigmoid(z):

    return (
        1.0
        /
        (
            1.0
            +
            np.exp(-z)
        )
    )


# =========================================================
# 5. Optical parameters
# =========================================================

wavelength = 0.193
NA = 0.85

cutoff = NA / wavelength


# =========================================================
# 6. Frequency grid
# =========================================================

freq = np.fft.fftfreq(
    N,
    d=dx
)

FX, FY = np.meshgrid(
    freq,
    freq
)


# =========================================================
# 7. Circular pupil
# =========================================================

pupil = (
    np.sqrt(
        FX**2
        +
        FY**2
    )
    <= cutoff
).astype(float)


# =========================================================
# 8. Optical forward model
# =========================================================

def optical_forward(mask_input):

    spectrum = np.fft.fft2(
        mask_input
    )

    filtered_spectrum = (
        spectrum
        *
        pupil
    )

    field = np.fft.ifft2(
        filtered_spectrum
    )

    intensity = (
        np.abs(field) ** 2
    )

    return (
        intensity,
        field
    )


# =========================================================
# 9. Fixed reference normalization
# =========================================================

reference_intensity, _ = optical_forward(
    target
)

reference_peak = np.max(
    reference_intensity
)


# =========================================================
# 10. Resist model
# =========================================================

threshold = 0.5
beta = 20.0
dose = 1.0


def soft_resist(intensity):

    return (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -beta
                *
                (
                    intensity
                    -
                    threshold
                )
            )
        )
    )


# =========================================================
# 11. Printing loss
# =========================================================

def printing_loss(
    resist,
    target_pattern
):

    return np.mean(
        (
            resist
            -
            target_pattern
        ) ** 2
    )


# =========================================================
# 12. Binarization loss
# =========================================================

lambda_bin = 0.01


def binarization_loss(mask):

    return np.mean(
        mask
        *
        (
            1.0
            -
            mask
        )
    )


# =========================================================
# 13. TV regularization
# =========================================================

lambda_tv = 0.001
tv_epsilon = 1e-6


def tv_loss(mask):

    # Horizontal difference
    dx_mask = (
        mask[:, 1:]
        -
        mask[:, :-1]
    )

    # Vertical difference
    dy_mask = (
        mask[1:, :]
        -
        mask[:-1, :]
    )

    loss_x = np.mean(
        np.sqrt(
            dx_mask**2
            +
            tv_epsilon**2
        )
    )

    loss_y = np.mean(
        np.sqrt(
            dy_mask**2
            +
            tv_epsilon**2
        )
    )

    return (
        loss_x
        +
        loss_y
    )


# =========================================================
# 14. Forward pass
# =========================================================

def forward_pass(Z_input):

    # Z -> mask
    mask = sigmoid(
        Z_input
    )

    # Optical imaging
    intensity, field = optical_forward(
        mask
    )

    # Fixed normalization
    intensity_norm = (
        intensity
        /
        reference_peak
    )

    # Dose
    effective_intensity = (
        dose
        *
        intensity_norm
    )

    # Resist
    resist = soft_resist(
        effective_intensity
    )

    # -----------------------------------------
    # Three loss terms
    # -----------------------------------------

    loss_print = printing_loss(
        resist,
        target
    )

    loss_bin = binarization_loss(
        mask
    )

    loss_tv = tv_loss(
        mask
    )

    # -----------------------------------------
    # Total loss
    # -----------------------------------------

    loss_total = (
        loss_print
        +
        lambda_bin
        *
        loss_bin
        +
        lambda_tv
        *
        loss_tv
    )

    return (
        loss_total,
        loss_print,
        loss_bin,
        loss_tv,
        mask,
        intensity,
        field,
        resist
    )


# =========================================================
# 15. Backward pass
# =========================================================

def backward_pass(Z_input):

    (
        loss_total,
        loss_print,
        loss_bin,
        loss_tv,
        mask,
        intensity,
        field,
        resist
    ) = forward_pass(
        Z_input
    )


    # =====================================================
    # A. Printing-loss gradient
    # =====================================================

    # dL_print / dR
    dL_dR = (
        2.0
        *
        (
            resist
            -
            target
        )
        /
        resist.size
    )


    # dR / dI_effective
    dR_dIeff = (
        beta
        *
        resist
        *
        (
            1.0
            -
            resist
        )
    )


    # dL / dI_effective
    dL_dIeff = (
        dL_dR
        *
        dR_dIeff
    )


    # I_effective = dose * I / reference_peak
    dL_dI = (
        dL_dIeff
        *
        dose
        /
        reference_peak
    )


    # I = |E|^2
    dL_dE = (
        2.0
        *
        dL_dI
        *
        field
    )


    # Optical adjoint
    dLprint_dM_complex = np.fft.ifft2(
        np.fft.fft2(
            dL_dE
        )
        *
        np.conj(
            pupil
        )
    )


    dLprint_dM = np.real(
        dLprint_dM_complex
    )


    # =====================================================
    # B. Binarization gradient
    # =====================================================

    dLbin_dM = (
        1.0
        -
        2.0
        *
        mask
    ) / mask.size


    # =====================================================
    # C. TV gradient
    # =====================================================

    dLtv_dM = np.zeros_like(
        mask
    )


    # -------------------------
    # Horizontal direction
    # -------------------------

    dx_mask = (
        mask[:, 1:]
        -
        mask[:, :-1]
    )


    grad_x = (
        dx_mask
        /
        np.sqrt(
            dx_mask**2
            +
            tv_epsilon**2
        )
        /
        dx_mask.size
    )


    dLtv_dM[:, 1:] += (
        grad_x
    )

    dLtv_dM[:, :-1] -= (
        grad_x
    )


    # -------------------------
    # Vertical direction
    # -------------------------

    dy_mask = (
        mask[1:, :]
        -
        mask[:-1, :]
    )


    grad_y = (
        dy_mask
        /
        np.sqrt(
            dy_mask**2
            +
            tv_epsilon**2
        )
        /
        dy_mask.size
    )


    dLtv_dM[1:, :] += (
        grad_y
    )

    dLtv_dM[:-1, :] -= (
        grad_y
    )


    # =====================================================
    # D. Total gradient with respect to mask
    # =====================================================

    dLtotal_dM = (
        dLprint_dM
        +
        lambda_bin
        *
        dLbin_dM
        +
        lambda_tv
        *
        dLtv_dM
    )


    # =====================================================
    # E. Mask -> Z
    # =====================================================

    dM_dZ = (
        mask
        *
        (
            1.0
            -
            mask
        )
    )


    dL_dZ = (
        dLtotal_dM
        *
        dM_dZ
    )


    return (
        loss_total,
        loss_print,
        loss_bin,
        loss_tv,
        dL_dZ,
        mask,
        resist
    )


# =========================================================
# 16. Save initial result
# =========================================================

(
    initial_total,
    initial_print,
    initial_bin,
    initial_tv,
    mask_initial,
    intensity_initial,
    field_initial,
    resist_initial
) = forward_pass(
    Z
)


# =========================================================
# 17. Optimization parameters
# =========================================================

learning_rate = 10.0
num_iters = 100


total_history = []
print_history = []
bin_history = []
tv_history = []


# =========================================================
# 18. Optimization loop
# =========================================================

for it in range(
    num_iters
):

    (
        loss_total,
        loss_print,
        loss_bin,
        loss_tv,
        dL_dZ,
        mask_now,
        resist_now
    ) = backward_pass(
        Z
    )


    # Save loss history
    total_history.append(
        loss_total
    )

    print_history.append(
        loss_print
    )

    bin_history.append(
        loss_bin
    )

    tv_history.append(
        loss_tv
    )


    # Gradient descent
    Z = (
        Z
        -
        learning_rate
        *
        dL_dZ
    )


    # Print every 10 iterations
    if (
        it % 10 == 0
        or
        it == num_iters - 1
    ):

        print(
            f"Iteration {it:3d} | "
            f"Total = {loss_total:.8f} | "
            f"Print = {loss_print:.8f} | "
            f"Bin = {loss_bin:.8f} | "
            f"TV = {loss_tv:.8f}"
        )


# =========================================================
# 19. Final result
# =========================================================

(
    final_total,
    final_print,
    final_bin,
    final_tv,
    mask_final,
    intensity_final,
    field_final,
    resist_final
) = forward_pass(
    Z
)


# =========================================================
# 20. Print final information
# =========================================================

print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(
    "Initial total loss =",
    initial_total
)

print(
    "Final total loss   =",
    final_total
)

print()

print(
    "Initial print loss =",
    initial_print
)

print(
    "Final print loss   =",
    final_print
)

print()

print(
    "Initial bin loss   =",
    initial_bin
)

print(
    "Final bin loss     =",
    final_bin
)

print()

print(
    "Initial TV loss    =",
    initial_tv
)

print(
    "Final TV loss      =",
    final_tv
)


# =========================================================
# 21. Final error map
# =========================================================

error_final = (
    resist_final
    -
    target
)


# =========================================================
# 22. One large comparison figure
# =========================================================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(18, 9)
)


extent = [
    x[0],
    x[-1],
    y[0],
    y[-1]
]


# ---------------------------------------------------------
# Target
# ---------------------------------------------------------

im0 = axes[0, 0].imshow(
    target,
    extent=extent,
    origin="lower"
)

axes[0, 0].set_title(
    "Target"
)

axes[0, 0].set_xlabel(
    "x (um)"
)

axes[0, 0].set_ylabel(
    "y (um)"
)

fig.colorbar(
    im0,
    ax=axes[0, 0]
)


# ---------------------------------------------------------
# Initial mask
# ---------------------------------------------------------

im1 = axes[0, 1].imshow(
    mask_initial,
    extent=extent,
    origin="lower"
)

axes[0, 1].set_title(
    "Initial Mask"
)

axes[0, 1].set_xlabel(
    "x (um)"
)

axes[0, 1].set_ylabel(
    "y (um)"
)

fig.colorbar(
    im1,
    ax=axes[0, 1]
)


# ---------------------------------------------------------
# Final mask
# ---------------------------------------------------------

im2 = axes[0, 2].imshow(
    mask_final,
    extent=extent,
    origin="lower"
)

axes[0, 2].set_title(
    "Final Mask"
)

axes[0, 2].set_xlabel(
    "x (um)"
)

axes[0, 2].set_ylabel(
    "y (um)"
)

fig.colorbar(
    im2,
    ax=axes[0, 2]
)


# ---------------------------------------------------------
# Final error
# ---------------------------------------------------------

im3 = axes[0, 3].imshow(
    error_final,
    extent=extent,
    origin="lower"
)

axes[0, 3].set_title(
    "Final Error: Resist - Target"
)

axes[0, 3].set_xlabel(
    "x (um)"
)

axes[0, 3].set_ylabel(
    "y (um)"
)

fig.colorbar(
    im3,
    ax=axes[0, 3]
)


# ---------------------------------------------------------
# Initial resist
# ---------------------------------------------------------

im4 = axes[1, 0].imshow(
    resist_initial,
    extent=extent,
    origin="lower"
)

axes[1, 0].set_title(
    "Initial Resist"
)

axes[1, 0].set_xlabel(
    "x (um)"
)

axes[1, 0].set_ylabel(
    "y (um)"
)

fig.colorbar(
    im4,
    ax=axes[1, 0]
)


# ---------------------------------------------------------
# Final resist
# ---------------------------------------------------------

im5 = axes[1, 1].imshow(
    resist_final,
    extent=extent,
    origin="lower"
)

axes[1, 1].set_title(
    "Final Resist"
)

axes[1, 1].set_xlabel(
    "x (um)"
)

axes[1, 1].set_ylabel(
    "y (um)"
)

fig.colorbar(
    im5,
    ax=axes[1, 1]
)


# ---------------------------------------------------------
# Total + printing loss
# ---------------------------------------------------------

axes[1, 2].plot(
    total_history,
    label="Total"
)

axes[1, 2].plot(
    print_history,
    label="Printing"
)

axes[1, 2].set_title(
    "Main Loss History"
)

axes[1, 2].set_xlabel(
    "Iteration"
)

axes[1, 2].set_ylabel(
    "Loss"
)

axes[1, 2].grid(
    True
)

axes[1, 2].legend()


# ---------------------------------------------------------
# Regularization losses
# ---------------------------------------------------------

axes[1, 3].plot(
    np.array(bin_history),
    label="Bin loss"
)

axes[1, 3].plot(
    np.array(tv_history),
    label="TV loss"
)

axes[1, 3].plot(
    lambda_bin
    *
    np.array(bin_history),
    label="Weighted Bin"
)

axes[1, 3].plot(
    lambda_tv
    *
    np.array(tv_history),
    label="Weighted TV"
)

axes[1, 3].set_title(
    "Regularization History"
)

axes[1, 3].set_xlabel(
    "Iteration"
)

axes[1, 3].set_ylabel(
    "Loss"
)

axes[1, 3].grid(
    True
)

axes[1, 3].legend()


plt.tight_layout()

plt.show()