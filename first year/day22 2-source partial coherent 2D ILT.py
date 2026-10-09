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

eps = 0.10

initial_mask = target.copy()

mask_clipped = np.clip(
    initial_mask,
    eps,
    1.0 - eps
)

Z = np.log(
    mask_clipped
    /
    (1.0 - mask_clipped)
)


# =========================================================
# 4. Sigmoid
# =========================================================

def sigmoid(z):

    return 1.0 / (
        1.0 + np.exp(-z)
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
# 7. Source points
# =========================================================

source_shift = 0.6

source_points = [
    (+source_shift, 0.0, 0.5),
    (-source_shift, 0.0, 0.5)
]


# =========================================================
# 8. Shifted pupil
# =========================================================

def make_shifted_pupil(
    source_x,
    source_y
):

    pupil = (
        np.sqrt(
            (FX - source_x) ** 2
            +
            (FY - source_y) ** 2
        )
        <= cutoff
    ).astype(float)

    return pupil


# =========================================================
# 9. Partial coherent forward model
# =========================================================

def optical_forward_partial(
    mask_input
):

    spectrum = np.fft.fft2(
        mask_input
    )

    total_intensity = np.zeros_like(
        mask_input,
        dtype=float
    )

    fields = []
    pupils = []


    for (
        source_x,
        source_y,
        weight
    ) in source_points:

        pupil_s = make_shifted_pupil(
            source_x,
            source_y
        )

        field_s = np.fft.ifft2(
            spectrum
            *
            pupil_s
        )

        intensity_s = (
            np.abs(field_s) ** 2
        )

        total_intensity += (
            weight
            *
            intensity_s
        )

        fields.append(
            field_s
        )

        pupils.append(
            pupil_s
        )


    return (
        total_intensity,
        fields,
        pupils
    )


# =========================================================
# 10. Fixed reference normalization
# =========================================================

reference_intensity, _, _ = optical_forward_partial(
    target
)

reference_peak = np.max(
    reference_intensity
)


# =========================================================
# 11. Resist model
# =========================================================

threshold = 0.5
beta = 20.0
dose = 1.0


def soft_resist(
    intensity
):

    return 1.0 / (
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


# =========================================================
# 12. Printing loss
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
# 13. Forward pass
# =========================================================

def forward_pass(
    Z_input
):

    mask = sigmoid(
        Z_input
    )


    (
        intensity,
        fields,
        pupils
    ) = optical_forward_partial(
        mask
    )


    intensity_norm = (
        intensity
        /
        reference_peak
    )


    effective_intensity = (
        dose
        *
        intensity_norm
    )


    resist = soft_resist(
        effective_intensity
    )


    loss = printing_loss(
        resist,
        target
    )


    return (
        loss,
        mask,
        intensity,
        fields,
        pupils,
        resist
    )


# =========================================================
# 14. Backward pass
# =========================================================

def backward_pass(
    Z_input
):

    (
        loss,
        mask,
        intensity,
        fields,
        pupils,
        resist
    ) = forward_pass(
        Z_input
    )


    # -----------------------------------------------------
    # dL / dR
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # dR / dIeff
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # dL / dIeff
    # -----------------------------------------------------

    dL_dIeff = (
        dL_dR
        *
        dR_dIeff
    )


    # -----------------------------------------------------
    # dL / dI_total
    # -----------------------------------------------------

    dL_dI = (
        dL_dIeff
        *
        dose
        /
        reference_peak
    )


    # -----------------------------------------------------
    # Partial coherent optical adjoint
    # -----------------------------------------------------

    dL_dM = np.zeros_like(
        mask,
        dtype=float
    )


    for (
        source,
        field_s,
        pupil_s
    ) in zip(
        source_points,
        fields,
        pupils
    ):

        source_x, source_y, weight = source


        # I_total = sum w_s |E_s|^2
        dL_dE_s = (
            2.0
            *
            weight
            *
            dL_dI
            *
            field_s
        )


        grad_mask_s = np.fft.ifft2(
            np.fft.fft2(
                dL_dE_s
            )
            *
            np.conj(
                pupil_s
            )
        )


        dL_dM += np.real(
            grad_mask_s
        )


    # -----------------------------------------------------
    # mask = sigmoid(Z)
    # -----------------------------------------------------

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
        dL_dM
        *
        dM_dZ
    )


    return (
        loss,
        dL_dZ,
        mask,
        resist
    )


# =========================================================
# 15. Initial result
# =========================================================

(
    loss_initial,
    mask_initial,
    intensity_initial,
    fields_initial,
    pupils_initial,
    resist_initial
) = forward_pass(
    Z
)


# =========================================================
# 16. Optimization
# =========================================================

learning_rate = 10.0
num_iters = 100

loss_history = []


for it in range(
    num_iters
):

    (
        loss,
        dL_dZ,
        mask_now,
        resist_now
    ) = backward_pass(
        Z
    )


    loss_history.append(
        loss
    )


    Z = (
        Z
        -
        learning_rate
        *
        dL_dZ
    )


    if (
        it % 10 == 0
        or
        it == num_iters - 1
    ):

        print(
            f"Iteration {it:3d}: "
            f"loss = {loss:.10f}"
        )


# =========================================================
# 17. Final result
# =========================================================

(
    loss_final,
    mask_final,
    intensity_final,
    fields_final,
    pupils_final,
    resist_final
) = forward_pass(
    Z
)


print()
print("=" * 60)
print("PARTIAL COHERENT 2D ILT")
print("=" * 60)

print(
    "Initial loss =",
    loss_initial
)

print(
    "Final loss   =",
    loss_final
)


# =========================================================
# 18. Error map
# =========================================================

error_final = (
    resist_final
    -
    target
)


# =========================================================
# 19. Comparison figure
# =========================================================

extent = [
    x[0],
    x[-1],
    y[0],
    y[-1]
]


fig, axes = plt.subplots(
    2,
    3,
    figsize=(14, 9)
)


# Target
im = axes[0, 0].imshow(
    target,
    extent=extent,
    origin="lower",
    vmin=0,
    vmax=1
)

axes[0, 0].set_title(
    "Target"
)

fig.colorbar(
    im,
    ax=axes[0, 0]
)


# Initial mask
im = axes[0, 1].imshow(
    mask_initial,
    extent=extent,
    origin="lower",
    vmin=0,
    vmax=1
)

axes[0, 1].set_title(
    "Initial Mask"
)

fig.colorbar(
    im,
    ax=axes[0, 1]
)


# Final mask
im = axes[0, 2].imshow(
    mask_final,
    extent=extent,
    origin="lower",
    vmin=0,
    vmax=1
)

axes[0, 2].set_title(
    "Final Mask"
)

fig.colorbar(
    im,
    ax=axes[0, 2]
)


# Initial resist
im = axes[1, 0].imshow(
    resist_initial,
    extent=extent,
    origin="lower",
    vmin=0,
    vmax=1
)

axes[1, 0].set_title(
    "Initial Resist"
)

fig.colorbar(
    im,
    ax=axes[1, 0]
)


# Final resist
im = axes[1, 1].imshow(
    resist_final,
    extent=extent,
    origin="lower",
    vmin=0,
    vmax=1
)

axes[1, 1].set_title(
    "Final Resist"
)

fig.colorbar(
    im,
    ax=axes[1, 1]
)


# Loss
axes[1, 2].plot(
    loss_history
)

axes[1, 2].set_title(
    "Printing Loss"
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


for ax in axes.flat:

    if ax != axes[1, 2]:

        ax.set_xlabel(
            "x (um)"
        )

        ax.set_ylabel(
            "y (um)"
        )


plt.tight_layout()

plt.show()

# =========================================================
# 20. Gradient check for partial coherent ILT
# =========================================================

epsilon_fd = 1e-5

test_pixels = [
    (50, 50),
    (60, 60),
    (64, 64),
    (70, 70),
    (80, 80)
]


print()
print("=" * 70)
print("PARTIAL COHERENT GRADIENT CHECK")
print("=" * 70)


# First compute analytical gradient at current Z
loss_check, dL_dZ_check, _, _ = backward_pass(
    Z
)


for i, j in test_pixels:

    analytical_gradient = dL_dZ_check[
        i,
        j
    ]


    # -------------------------
    # Z + epsilon
    # -------------------------

    Z_plus = Z.copy()

    Z_plus[
        i,
        j
    ] += epsilon_fd


    loss_plus, _, _, _, _, _ = forward_pass(
        Z_plus
    )


    # -------------------------
    # Z - epsilon
    # -------------------------

    Z_minus = Z.copy()

    Z_minus[
        i,
        j
    ] -= epsilon_fd


    loss_minus, _, _, _, _, _ = forward_pass(
        Z_minus
    )


    # -------------------------
    # Finite difference
    # -------------------------

    fd_gradient = (
        loss_plus
        -
        loss_minus
    ) / (
        2.0
        *
        epsilon_fd
    )


    # -------------------------
    # Relative error
    # -------------------------

    denominator = max(
        abs(
            analytical_gradient
        ),
        abs(
            fd_gradient
        ),
        1e-12
    )


    relative_error = (
        abs(
            analytical_gradient
            -
            fd_gradient
        )
        /
        denominator
    )


    print(
        f"Pixel ({i}, {j})"
    )

    print(
        "  Analytical =",
        analytical_gradient
    )

    print(
        "  Finite diff =",
        fd_gradient
    )

    print(
        "  Relative error =",
        relative_error
    )

    print()