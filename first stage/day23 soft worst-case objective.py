import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Grid
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

target = (
    (np.abs(X) <= 0.60 / 2)
    &
    (np.abs(Y) <= 0.40 / 2)
).astype(float)


# =========================================================
# 3. Initial optimization variable Z
# =========================================================

eps = 0.10

mask_init = np.clip(
    target,
    eps,
    1.0 - eps
)

Z_initial = np.log(
    mask_init
    /
    (1.0 - mask_init)
)


def sigmoid(z):

    return 1.0 / (
        1.0 + np.exp(-z)
    )


# =========================================================
# 4. Optical parameters
# =========================================================

wavelength = 0.193
NA = 0.85

cutoff = NA / wavelength


freq = np.fft.fftfreq(
    N,
    d=dx
)

FX, FY = np.meshgrid(
    freq,
    freq
)


# =========================================================
# 5. Partial coherent source
# =========================================================

source_shift = 0.6

source_points = [
    (+source_shift, 0.0, 0.5),
    (-source_shift, 0.0, 0.5)
]


# =========================================================
# 6. Process conditions
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
# 7. Defocused shifted pupil
# =========================================================

def make_pupil(
    source_x,
    source_y,
    focus
):

    fx = FX - source_x
    fy = FY - source_y


    aperture = (
        np.sqrt(
            fx**2
            +
            fy**2
        )
        <= cutoff
    ).astype(float)


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
            fx**2
            +
            fy**2
        )
    )


    return (
        aperture
        *
        phase
    )


# =========================================================
# 8. Partial coherent imaging
# =========================================================

def optical_forward(
    mask,
    focus
):

    spectrum = np.fft.fft2(
        mask
    )


    total_intensity = np.zeros_like(
        mask,
        dtype=float
    )


    fields = []
    pupils = []


    for (
        source_x,
        source_y,
        weight
    ) in source_points:

        pupil = make_pupil(
            source_x,
            source_y,
            focus
        )


        field = np.fft.ifft2(
            spectrum
            *
            pupil
        )


        total_intensity += (
            weight
            *
            np.abs(field)**2
        )


        fields.append(
            field
        )

        pupils.append(
            pupil
        )


    return (
        total_intensity,
        fields,
        pupils
    )


# =========================================================
# 9. Fixed normalization reference
# =========================================================

reference_intensity, _, _ = optical_forward(
    target,
    focus=0.0
)

reference_peak = np.max(
    reference_intensity
)


# =========================================================
# 10. Resist
# =========================================================

threshold = 0.5
beta = 20.0


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
# 11. One process condition:
#     loss + mask gradient
# =========================================================

def condition_loss_gradient(
    mask,
    focus,
    dose
):

    (
        intensity,
        fields,
        pupils
    ) = optical_forward(
        mask,
        focus
    )


    # Fixed normalization + dose
    intensity_eff = (
        dose
        *
        intensity
        /
        reference_peak
    )


    resist = soft_resist(
        intensity_eff
    )


    # -----------------------------------------------------
    # Printing loss
    # -----------------------------------------------------

    loss = np.mean(
        (
            resist
            -
            target
        )**2
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
    # dL / dI
    # -----------------------------------------------------

    dL_dI = (
        dL_dR
        *
        dR_dIeff
        *
        dose
        /
        reference_peak
    )


    # -----------------------------------------------------
    # Partial coherent adjoint
    # -----------------------------------------------------

    grad_mask = np.zeros_like(
        mask,
        dtype=float
    )


    for (
        source,
        field,
        pupil
    ) in zip(
        source_points,
        fields,
        pupils
    ):

        _, _, weight = source


        dL_dE = (
            2.0
            *
            weight
            *
            dL_dI
            *
            field
        )


        grad_source = np.fft.ifft2(
            np.fft.fft2(
                dL_dE
            )
            *
            np.conj(
                pupil
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
# 12. Soft worst-case objective
# =========================================================

# Smaller tau -> closer to hard worst-case
# Larger tau  -> closer to average
tau = 0.002


def soft_worst_backward(
    Z
):

    mask = sigmoid(
        Z
    )


    condition_losses = []
    condition_gradients = []


    # -----------------------------------------------------
    # Compute loss and gradient for every process condition
    # -----------------------------------------------------

    for focus in focus_list:

        for dose in dose_list:

            (
                loss,
                grad_mask,
                _
            ) = condition_loss_gradient(
                mask,
                focus,
                dose
            )


            condition_losses.append(
                loss
            )

            condition_gradients.append(
                grad_mask
            )


    condition_losses = np.array(
        condition_losses
    )


    # =====================================================
    # Stable softmax weights
    # =====================================================

    max_loss = np.max(
        condition_losses
    )


    exp_values = np.exp(
        (
            condition_losses
            -
            max_loss
        )
        /
        tau
    )


    weights = (
        exp_values
        /
        np.sum(
            exp_values
        )
    )


    # =====================================================
    # Soft worst-case loss
    #
    # tau * log(mean(exp(loss/tau)))
    # =====================================================

    soft_loss = (
        max_loss
        +
        tau
        *
        np.log(
            np.mean(
                exp_values
            )
        )
    )


    # =====================================================
    # Weighted process-condition gradient
    # =====================================================

    grad_mask_total = np.zeros_like(
        mask,
        dtype=float
    )


    for weight, grad in zip(
        weights,
        condition_gradients
    ):

        grad_mask_total += (
            weight
            *
            grad
        )


    # =====================================================
    # mask = sigmoid(Z)
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
        grad_mask_total
        *
        dM_dZ
    )


    return (
        soft_loss,
        dL_dZ,
        condition_losses,
        weights
    )


# =========================================================
# 13. Optimization
# =========================================================

learning_rate = 10.0
num_iters = 150


Z = Z_initial.copy()

loss_history = []


for it in range(
    num_iters
):

    (
        loss,
        dL_dZ,
        condition_losses,
        weights
    ) = soft_worst_backward(
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
        it % 20 == 0
        or
        it == num_iters - 1
    ):

        print(
            f"Iteration {it:3d} | "
            f"Soft worst loss = {loss:.10f} | "
            f"Worst condition = {np.max(condition_losses):.10f}"
        )


# =========================================================
# 14. Final mask
# =========================================================

mask_final = sigmoid(
    Z
)


# =========================================================
# 15. Final process loss + weight maps
# =========================================================

(
    final_soft_loss,
    _,
    final_condition_losses,
    final_weights
) = soft_worst_backward(
    Z
)


loss_matrix = final_condition_losses.reshape(
    len(focus_list),
    len(dose_list)
)


weight_matrix = final_weights.reshape(
    len(focus_list),
    len(dose_list)
)


# =========================================================
# 16. Nominal-condition resist
# =========================================================

(
    nominal_loss,
    _,
    resist_nominal
) = condition_loss_gradient(
    mask_final,
    focus=0.0,
    dose=1.0
)


# =========================================================
# 17. Print results
# =========================================================

print()
print("=" * 70)
print("SOFT WORST-CASE ROBUST ILT")
print("=" * 70)

print(
    "Nominal loss =",
    nominal_loss
)

print(
    "Average process loss =",
    np.mean(
        loss_matrix
    )
)

print(
    "Worst process loss =",
    np.max(
        loss_matrix
    )
)

print(
    "Soft worst loss =",
    final_soft_loss
)

print()
print("Process loss matrix:")
print(
    loss_matrix
)

print()
print("Process weight matrix:")
print(
    weight_matrix
)

print()
print(
    "Weight sum =",
    np.sum(
        weight_matrix
    )
)


# =========================================================
# 18. Plot
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
    figsize=(15, 9)
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
# Initial mask
# ---------------------------------------------------------

im = axes[0, 1].imshow(
    sigmoid(
        Z_initial
    ),
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


# ---------------------------------------------------------
# Final mask
# ---------------------------------------------------------

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
    "Printed Resist\nFocus=0, Dose=1"
)

fig.colorbar(
    im,
    ax=axes[1, 0]
)


# ---------------------------------------------------------
# Process loss map
# ---------------------------------------------------------

im = axes[1, 1].imshow(
    loss_matrix,
    origin="lower",
    aspect="auto"
)

axes[1, 1].set_title(
    "Process Loss Map"
)

axes[1, 1].set_xticks(
    range(
        len(dose_list)
    )
)

axes[1, 1].set_xticklabels(
    dose_list
)

axes[1, 1].set_yticks(
    range(
        len(focus_list)
    )
)

axes[1, 1].set_yticklabels(
    focus_list
)

axes[1, 1].set_xlabel(
    "Dose"
)

axes[1, 1].set_ylabel(
    "Focus (um)"
)

fig.colorbar(
    im,
    ax=axes[1, 1]
)


# ---------------------------------------------------------
# Process weight map
# ---------------------------------------------------------

im = axes[1, 2].imshow(
    weight_matrix,
    origin="lower",
    aspect="auto",
    vmin=0
)

axes[1, 2].set_title(
    "Soft Worst-Case Weight Map"
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


plt.tight_layout()

plt.show()


# =========================================================
# 19. Optimization history
# =========================================================

plt.figure(
    figsize=(7, 4)
)

plt.plot(
    loss_history
)

plt.xlabel(
    "Iteration"
)

plt.ylabel(
    "Soft Worst-Case Loss"
)

plt.title(
    "Optimization History"
)

plt.grid(
    True
)

plt.show()