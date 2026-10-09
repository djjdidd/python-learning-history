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
# 2. Target
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

Z_initial = np.log(
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
# 7. Partial coherent source
# =========================================================

source_shift = 0.6

source_points = [
    (+source_shift, 0.0, 0.5),
    (-source_shift, 0.0, 0.5)
]


# =========================================================
# 8. Process conditions
# =========================================================

focus_list = np.array([
    -0.05,
     0.00,
    +0.05
])

dose_list = np.array([
    0.95,
    1.00,
    1.05
])


# =========================================================
# 9. Defocused shifted pupil
# =========================================================

def make_defocused_pupil(
    source_x,
    source_y,
    focus
):

    FX_shift = (
        FX - source_x
    )

    FY_shift = (
        FY - source_y
    )


    # Aperture
    aperture = (
        np.sqrt(
            FX_shift ** 2
            +
            FY_shift ** 2
        )
        <= cutoff
    ).astype(float)


    # Defocus phase
    phase = np.exp(
        -1j
        *
        np.pi
        *
        wavelength
        *
        focus
        *
        (
            FX_shift ** 2
            +
            FY_shift ** 2
        )
    )


    pupil = (
        aperture
        *
        phase
    )


    return pupil


# =========================================================
# 10. Partial coherent imaging
# =========================================================

def optical_forward_partial(
    mask_input,
    focus
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

        pupil_s = make_defocused_pupil(
            source_x,
            source_y,
            focus
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
# 11. Fixed reference normalization
# =========================================================
#
# Use target at nominal focus as reference.
# This value stays fixed during optimization.
#

reference_intensity, _, _ = optical_forward_partial(
    target,
    focus=0.0
)

reference_peak = np.max(
    reference_intensity
)


# =========================================================
# 12. Resist model
# =========================================================

threshold = 0.5
beta = 20.0


def soft_resist(
    intensity
):

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
# 13. Printing loss
# =========================================================

def printing_loss(
    resist
):

    return np.mean(
        (
            resist
            -
            target
        ) ** 2
    )


# =========================================================
# 14. One process condition: forward
# =========================================================

def condition_forward(
    mask,
    focus,
    dose
):

    (
        intensity,
        fields,
        pupils
    ) = optical_forward_partial(
        mask,
        focus
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
        resist
    )


    return (
        loss,
        resist,
        intensity,
        fields,
        pupils
    )


# =========================================================
# 15. One process condition: gradient
# =========================================================

def condition_gradient(
    mask,
    focus,
    dose
):

    (
        loss,
        resist,
        intensity,
        fields,
        pupils
    ) = condition_forward(
        mask,
        focus,
        dose
    )


    # -----------------------------------------------------
    # Loss -> resist
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
    # Resist -> effective intensity
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


    dL_dIeff = (
        dL_dR
        *
        dR_dIeff
    )


    # -----------------------------------------------------
    # Effective intensity -> optical intensity
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

    grad_mask = np.zeros_like(
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


        # dL / dE_s
        dL_dE_s = (
            2.0
            *
            weight
            *
            dL_dI
            *
            field_s
        )


        grad_source = np.fft.ifft2(
            np.fft.fft2(
                dL_dE_s
            )
            *
            np.conj(
                pupil_s
            )
        )


        grad_mask += np.real(
            grad_source
        )


    return (
        loss,
        grad_mask,
        resist
    )


# =========================================================
# 16. Nominal objective
# =========================================================

def nominal_backward(
    Z
):

    mask = sigmoid(
        Z
    )


    (
        loss,
        grad_mask,
        resist
    ) = condition_gradient(
        mask,
        focus=0.0,
        dose=1.0
    )


    # mask = sigmoid(Z)
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
        grad_mask
        *
        dM_dZ
    )


    return (
        loss,
        dL_dZ
    )


# =========================================================
# 17. Robust objective
# =========================================================

def robust_backward(
    Z
):

    mask = sigmoid(
        Z
    )


    total_loss = 0.0


    grad_robust = np.zeros_like(
        mask,
        dtype=float
    )


    condition_count = 0


    # -----------------------------------------------------
    # Outer loop: focus
    # -----------------------------------------------------

    for focus in focus_list:

        # -------------------------------------------------
        # Inner loop: dose
        # -------------------------------------------------

        for dose in dose_list:

            (
                loss_condition,
                grad_condition,
                resist_condition
            ) = condition_gradient(
                mask,
                focus,
                dose
            )


            total_loss += (
                loss_condition
            )


            grad_robust += (
                grad_condition
            )


            condition_count += 1


    # -----------------------------------------------------
    # Average across process conditions
    # -----------------------------------------------------

    robust_loss = (
        total_loss
        /
        condition_count
    )


    grad_robust = (
        grad_robust
        /
        condition_count
    )


    # -----------------------------------------------------
    # mask -> Z
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
        grad_robust
        *
        dM_dZ
    )


    return (
        robust_loss,
        dL_dZ
    )


# =========================================================
# 18. Generic optimizer
# =========================================================

def optimize(
    mode,
    num_iters=100,
    learning_rate=10.0
):

    Z = Z_initial.copy()

    loss_history = []


    for it in range(
        num_iters
    ):

        if mode == "nominal":

            loss, dL_dZ = nominal_backward(
                Z
            )


        elif mode == "robust":

            loss, dL_dZ = robust_backward(
                Z
            )


        else:

            raise ValueError(
                "mode must be 'nominal' or 'robust'"
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
            it % 20 == 0
            or
            it == num_iters - 1
        ):

            print(
                mode,
                "| iteration",
                it,
                "| loss =",
                loss
            )


    final_mask = sigmoid(
        Z
    )


    return (
        final_mask,
        np.array(
            loss_history
        )
    )


# =========================================================
# 19. Run nominal ILT
# =========================================================

print()
print("=" * 70)
print("NOMINAL ILT")
print("=" * 70)


mask_nominal, history_nominal = optimize(
    mode="nominal",
    num_iters=100,
    learning_rate=10.0
)


# =========================================================
# 20. Run robust ILT
# =========================================================

print()
print("=" * 70)
print("ROBUST ILT")
print("=" * 70)


mask_robust, history_robust = optimize(
    mode="robust",
    num_iters=100,
    learning_rate=10.0
)


# =========================================================
# 21. Evaluate process window loss matrix
# =========================================================

def evaluate_process_matrix(
    mask
):

    loss_matrix = np.zeros(
        (
            len(focus_list),
            len(dose_list)
        )
    )


    for i, focus in enumerate(
        focus_list
    ):

        for j, dose in enumerate(
            dose_list
        ):

            (
                loss,
                resist,
                intensity,
                fields,
                pupils
            ) = condition_forward(
                mask,
                focus,
                dose
            )


            loss_matrix[
                i,
                j
            ] = loss


    return loss_matrix


loss_matrix_nominal = evaluate_process_matrix(
    mask_nominal
)

loss_matrix_robust = evaluate_process_matrix(
    mask_robust
)


# =========================================================
# 22. Nominal-condition resist
# =========================================================

(
    nominal_loss_at_nominal,
    resist_nominal,
    _,
    _,
    _
) = condition_forward(
    mask_nominal,
    focus=0.0,
    dose=1.0
)


(
    robust_loss_at_nominal,
    resist_robust,
    _,
    _,
    _
) = condition_forward(
    mask_robust,
    focus=0.0,
    dose=1.0
)


# =========================================================
# 23. Statistics
# =========================================================

print()
print("=" * 70)
print("PROCESS CONDITION COMPARISON")
print("=" * 70)


print()
print("Nominal ILT mask")

print(
    "Nominal-condition loss =",
    nominal_loss_at_nominal
)

print(
    "Average process loss   =",
    np.mean(
        loss_matrix_nominal
    )
)

print(
    "Worst process loss     =",
    np.max(
        loss_matrix_nominal
    )
)


print()
print("Robust ILT mask")

print(
    "Nominal-condition loss =",
    robust_loss_at_nominal
)

print(
    "Average process loss   =",
    np.mean(
        loss_matrix_robust
    )
)

print(
    "Worst process loss     =",
    np.max(
        loss_matrix_robust
    )
)


# =========================================================
# 24. Plot comparison
# =========================================================

extent = [
    x[0],
    x[-1],
    y[0],
    y[-1]
]


fig, axes = plt.subplots(
    2,
    4,
    figsize=(18, 9)
)


# ---------------------------------------------------------
# Target
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Nominal mask
# ---------------------------------------------------------

im = axes[0, 1].imshow(
    mask_nominal,
    extent=extent,
    origin="lower",
    vmin=0,
    vmax=1
)

axes[0, 1].set_title(
    "Nominal ILT Mask"
)

fig.colorbar(
    im,
    ax=axes[0, 1]
)


# ---------------------------------------------------------
# Robust mask
# ---------------------------------------------------------

im = axes[0, 2].imshow(
    mask_robust,
    extent=extent,
    origin="lower",
    vmin=0,
    vmax=1
)

axes[0, 2].set_title(
    "Robust ILT Mask"
)

fig.colorbar(
    im,
    ax=axes[0, 2]
)


# ---------------------------------------------------------
# Mask difference
# ---------------------------------------------------------

mask_difference = (
    mask_robust
    -
    mask_nominal
)


im = axes[0, 3].imshow(
    mask_difference,
    extent=extent,
    origin="lower"
)

axes[0, 3].set_title(
    "Robust - Nominal Mask"
)

fig.colorbar(
    im,
    ax=axes[0, 3]
)


# ---------------------------------------------------------
# Nominal resist
# ---------------------------------------------------------

im = axes[1, 0].imshow(
    resist_nominal,
    extent=extent,
    origin="lower",
    vmin=0,
    vmax=1
)

axes[1, 0].set_title(
    "Nominal Mask\nat Focus=0, Dose=1"
)

fig.colorbar(
    im,
    ax=axes[1, 0]
)


# ---------------------------------------------------------
# Robust resist
# ---------------------------------------------------------

im = axes[1, 1].imshow(
    resist_robust,
    extent=extent,
    origin="lower",
    vmin=0,
    vmax=1
)

axes[1, 1].set_title(
    "Robust Mask\nat Focus=0, Dose=1"
)

fig.colorbar(
    im,
    ax=axes[1, 1]
)


# ---------------------------------------------------------
# Nominal process loss map
# ---------------------------------------------------------

im = axes[1, 2].imshow(
    loss_matrix_nominal,
    origin="lower",
    aspect="auto"
)

axes[1, 2].set_title(
    "Nominal ILT\nProcess Loss Map"
)

axes[1, 2].set_xticks(
    range(
        len(dose_list)
    )
)

axes[1, 2].set_xticklabels(
    dose_list
)

axes[1, 2].set_yticks(
    range(
        len(focus_list)
    )
)

axes[1, 2].set_yticklabels(
    focus_list
)

axes[1, 2].set_xlabel(
    "Dose"
)

axes[1, 2].set_ylabel(
    "Focus (um)"
)

fig.colorbar(
    im,
    ax=axes[1, 2]
)


# ---------------------------------------------------------
# Robust process loss map
# ---------------------------------------------------------

im = axes[1, 3].imshow(
    loss_matrix_robust,
    origin="lower",
    aspect="auto"
)

axes[1, 3].set_title(
    "Robust ILT\nProcess Loss Map"
)

axes[1, 3].set_xticks(
    range(
        len(dose_list)
    )
)

axes[1, 3].set_xticklabels(
    dose_list
)

axes[1, 3].set_yticks(
    range(
        len(focus_list)
    )
)

axes[1, 3].set_yticklabels(
    focus_list
)

axes[1, 3].set_xlabel(
    "Dose"
)

axes[1, 3].set_ylabel(
    "Focus (um)"
)

fig.colorbar(
    im,
    ax=axes[1, 3]
)


for ax in axes[0, :]:

    ax.set_xlabel(
        "x (um)"
    )

    ax.set_ylabel(
        "y (um)"
    )


axes[1, 0].set_xlabel(
    "x (um)"
)

axes[1, 0].set_ylabel(
    "y (um)"
)

axes[1, 1].set_xlabel(
    "x (um)"
)

axes[1, 1].set_ylabel(
    "y (um)"
)


plt.tight_layout()

plt.show()


# =========================================================
# 25. Optimization history
# =========================================================

plt.figure(
    figsize=(7, 4)
)

plt.plot(
    history_nominal,
    label="Nominal ILT"
)

plt.plot(
    history_robust,
    label="Robust ILT"
)

plt.xlabel(
    "Iteration"
)

plt.ylabel(
    "Optimization Loss"
)

plt.title(
    "Optimization History"
)

plt.legend()

plt.grid(
    True
)

plt.show()
# =========================================================
# 26. Robust gradient check
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
print("ROBUST ILT GRADIENT CHECK")
print("=" * 70)


# Analytical robust gradient
loss_robust, dL_dZ_robust = robust_backward(
    Z_initial
)


for i, j in test_pixels:

    # -----------------------------------------------------
    # Analytical gradient
    # -----------------------------------------------------

    analytical = dL_dZ_robust[
        i,
        j
    ]


    # -----------------------------------------------------
    # Z + epsilon
    # -----------------------------------------------------

    Z_plus = Z_initial.copy()

    Z_plus[
        i,
        j
    ] += epsilon_fd


    loss_plus, _ = robust_backward(
        Z_plus
    )


    # -----------------------------------------------------
    # Z - epsilon
    # -----------------------------------------------------

    Z_minus = Z_initial.copy()

    Z_minus[
        i,
        j
    ] -= epsilon_fd


    loss_minus, _ = robust_backward(
        Z_minus
    )


    # -----------------------------------------------------
    # Finite-difference gradient
    # -----------------------------------------------------

    finite_difference = (
        loss_plus
        -
        loss_minus
    ) / (
        2.0
        *
        epsilon_fd
    )


    # -----------------------------------------------------
    # Relative error
    # -----------------------------------------------------

    denominator = max(
        abs(analytical),
        abs(finite_difference),
        1e-12
    )


    relative_error = (
        abs(
            analytical
            -
            finite_difference
        )
        /
        denominator
    )


    print(
        f"Pixel ({i}, {j})"
    )

    print(
        "  Analytical  =",
        analytical
    )

    print(
        "  Finite diff =",
        finite_difference
    )

    print(
        "  Relative error =",
        relative_error
    )

    print()