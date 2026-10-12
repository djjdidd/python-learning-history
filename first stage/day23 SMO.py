import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Spatial grid
# =========================================================

N = 128
L = 2.0                    # um
dx = L / N

x = np.arange(N) * dx - L / 2
y = np.arange(N) * dx - L / 2

X, Y = np.meshgrid(x, y)


# =========================================================
# 2. Target pattern
# =========================================================

target = (
    (np.abs(X) <= 0.60 / 2)
    &
    (np.abs(Y) <= 0.40 / 2)
).astype(float)


# =========================================================
# 3. Initial mask -> Z
# =========================================================

eps = 0.10

mask_initial = np.clip(
    target,
    eps,
    1.0 - eps
)

Z_initial = np.log(
    mask_initial
    /
    (1.0 - mask_initial)
)


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
# 4. Optical parameters
# =========================================================

wavelength = 0.193          # um
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
# 5. Process conditions
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
# 6. Source model
# =========================================================
# Horizontal dipole:
#
# (+shift, 0)
# (-shift, 0)
#
# Both have weight = 0.5
# =========================================================

def get_source_points(
    shift
):

    return [
        (+shift, 0.0, 0.5),
        (-shift, 0.0, 0.5)
    ]


# =========================================================
# 7. Defocused shifted pupil
# =========================================================

def make_pupil(
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


    # -----------------------------------------------------
    # Aperture
    # -----------------------------------------------------

    aperture = (
        np.sqrt(
            FX_shift**2
            +
            FY_shift**2
        )
        <= cutoff
    ).astype(float)


    # -----------------------------------------------------
    # Defocus phase
    # -----------------------------------------------------

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
            FX_shift**2
            +
            FY_shift**2
        )
    )


    pupil = (
        aperture
        *
        phase
    )


    return pupil


# =========================================================
# 8. Partial coherent imaging
# =========================================================

def optical_forward(
    mask,
    focus,
    source_shift
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


    source_points = get_source_points(
        source_shift
    )


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


        intensity_source = (
            np.abs(field)**2
        )


        total_intensity += (
            weight
            *
            intensity_source
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
# The reference must remain fixed during optimization.
# We use the initial source shift 0.6 at best focus.
# =========================================================

initial_source_shift = 0.6


reference_intensity, _, _ = optical_forward(
    target,
    focus=0.0,
    source_shift=initial_source_shift
)


reference_peak = np.max(
    reference_intensity
)


# =========================================================
# 10. Resist model
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
# 11. Loss for one process condition
# =========================================================

def condition_loss(
    mask,
    focus,
    dose,
    source_shift
):

    intensity, _, _ = optical_forward(
        mask,
        focus,
        source_shift
    )


    effective_intensity = (
        dose
        *
        intensity
        /
        reference_peak
    )


    resist = soft_resist(
        effective_intensity
    )


    loss = np.mean(
        (
            resist
            -
            target
        )**2
    )


    return loss


# =========================================================
# 12. Loss + gradient for one process condition
# =========================================================

def condition_loss_gradient(
    mask,
    focus,
    dose,
    source_shift
):

    (
        intensity,
        fields,
        pupils
    ) = optical_forward(
        mask,
        focus,
        source_shift
    )


    effective_intensity = (
        dose
        *
        intensity
        /
        reference_peak
    )


    resist = soft_resist(
        effective_intensity
    )


    # =====================================================
    # Printing loss
    # =====================================================

    loss = np.mean(
        (
            resist
            -
            target
        )**2
    )


    # =====================================================
    # dL / dR
    # =====================================================

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


    # =====================================================
    # Resist derivative
    # =====================================================

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


    # =====================================================
    # dL / dI
    # =====================================================

    dL_dI = (
        dL_dR
        *
        dR_dIeff
        *
        dose
        /
        reference_peak
    )


    # =====================================================
    # Partial coherent adjoint
    # =====================================================

    grad_mask = np.zeros_like(
        mask,
        dtype=float
    )


    source_points = get_source_points(
        source_shift
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

        _, _, source_weight = source


        # -------------------------------------------------
        # Intensity -> field
        # -------------------------------------------------

        dL_dE = (
            2.0
            *
            source_weight
            *
            dL_dI
            *
            field
        )


        # -------------------------------------------------
        # Optical adjoint
        # -------------------------------------------------

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
        grad_mask
    )


# =========================================================
# 13. Soft worst-case parameters
# =========================================================

tau = 0.002


# =========================================================
# 14. Stable soft worst-case calculation
# =========================================================

def soft_worst_value_and_weights(
    losses
):

    losses = np.array(
        losses
    )


    max_loss = np.max(
        losses
    )


    exp_values = np.exp(
        (
            losses
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


    return (
        soft_loss,
        weights
    )


# =========================================================
# 15. Robust backward for fixed source
# =========================================================

def robust_backward(
    Z,
    source_shift
):

    mask = sigmoid(
        Z
    )


    condition_losses = []
    condition_gradients = []


    # =====================================================
    # All focus / dose conditions
    # =====================================================

    for focus in focus_list:

        for dose in dose_list:

            (
                loss,
                grad_mask
            ) = condition_loss_gradient(
                mask,
                focus,
                dose,
                source_shift
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
    # Soft worst-case weights
    # =====================================================

    (
        soft_loss,
        weights
    ) = soft_worst_value_and_weights(
        condition_losses
    )


    # =====================================================
    # Weighted process-condition gradient
    # =====================================================

    grad_mask_total = np.zeros_like(
        mask,
        dtype=float
    )


    for (
        process_weight,
        grad
    ) in zip(
        weights,
        condition_gradients
    ):

        grad_mask_total += (
            process_weight
            *
            grad
        )


    # =====================================================
    # M = sigmoid(Z)
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
# 16. Fixed source -> optimize mask
# =========================================================

def optimize_mask(
    Z,
    source_shift,
    num_iters=30,
    learning_rate=10.0
):

    loss_history = []


    for _ in range(
        num_iters
    ):

        (
            loss,
            dL_dZ,
            _,
            _
        ) = robust_backward(
            Z,
            source_shift
        )


        Z = (
            Z
            -
            learning_rate
            *
            dL_dZ
        )


        loss_history.append(
            loss
        )


    return (
        Z,
        np.array(
            loss_history
        )
    )


# =========================================================
# 17. Fixed mask -> evaluate one source shift
# =========================================================

def evaluate_source_shift(
    mask,
    source_shift
):

    losses = []


    for focus in focus_list:

        for dose in dose_list:

            loss = condition_loss(
                mask,
                focus,
                dose,
                source_shift
            )


            losses.append(
                loss
            )


    losses = np.array(
        losses
    )


    (
        soft_loss,
        weights
    ) = soft_worst_value_and_weights(
        losses
    )


    return (
        soft_loss,
        losses,
        weights
    )


# =========================================================
# 18. Fixed mask -> scan source
# =========================================================

def find_best_source(
    mask,
    shift_candidates
):

    best_shift = None
    best_loss = np.inf


    scan_losses = []


    for shift in shift_candidates:

        (
            soft_loss,
            _,
            _
        ) = evaluate_source_shift(
            mask,
            shift
        )


        scan_losses.append(
            soft_loss
        )


        if soft_loss < best_loss:

            best_loss = soft_loss
            best_shift = shift


    return (
        best_shift,
        best_loss,
        np.array(
            scan_losses
        )
    )


# =========================================================
# 19. Alternating SMO parameters
# =========================================================

shift_candidates = np.linspace(
    0.0,
    1.2,
    13
)


num_smo_cycles = 5

mask_iterations_per_cycle = 30
learning_rate = 10.0


# =========================================================
# 20. Alternating SMO
# =========================================================

Z = Z_initial.copy()

current_shift = initial_source_shift


shift_history = []
loss_history_smo = []


print()
print("=" * 70)
print("ALTERNATING SOURCE-MASK OPTIMIZATION")
print("=" * 70)


for cycle in range(
    num_smo_cycles
):

    print()
    print(
        f"SMO Cycle {cycle + 1}"
    )

    print(
        "-" * 50
    )


    # =====================================================
    # A. Fix source -> optimize mask
    # =====================================================

    Z, mask_loss_history = optimize_mask(
        Z,
        current_shift,
        num_iters=mask_iterations_per_cycle,
        learning_rate=learning_rate
    )


    mask_now = sigmoid(
        Z
    )


    # =====================================================
    # B. Fix mask -> optimize source
    # =====================================================

    (
        best_shift,
        best_loss,
        scan_losses
    ) = find_best_source(
        mask_now,
        shift_candidates
    )


    current_shift = best_shift


    shift_history.append(
        current_shift
    )


    loss_history_smo.append(
        best_loss
    )


    print(
        "Best source shift =",
        current_shift
    )


    print(
        "Soft robust loss  =",
        best_loss
    )


# =========================================================
# 21. Final mask refinement
# =========================================================
# The last source scan changes the source once more.
# Therefore we do one final mask optimization using that
# final source.
# =========================================================

Z, _ = optimize_mask(
    Z,
    current_shift,
    num_iters=mask_iterations_per_cycle,
    learning_rate=learning_rate
)


mask_final = sigmoid(
    Z
)


# =========================================================
# 22. Final process evaluation
# =========================================================

(
    final_soft_loss,
    final_losses,
    final_weights
) = evaluate_source_shift(
    mask_final,
    current_shift
)


loss_matrix = final_losses.reshape(
    len(focus_list),
    len(dose_list)
)


weight_matrix = final_weights.reshape(
    len(focus_list),
    len(dose_list)
)


# =========================================================
# 23. Final metrics
# =========================================================

nominal_focus_index = np.where(
    np.isclose(
        focus_list,
        0.0
    )
)[0][0]


nominal_dose_index = np.where(
    np.isclose(
        dose_list,
        1.0
    )
)[0][0]


nominal_loss = loss_matrix[
    nominal_focus_index,
    nominal_dose_index
]


average_loss = np.mean(
    loss_matrix
)


worst_loss = np.max(
    loss_matrix
)


print()
print("=" * 70)
print("FINAL SMO RESULT")
print("=" * 70)


print(
    "Final source shift =",
    current_shift
)

print(
    "Nominal loss =",
    nominal_loss
)

print(
    "Average process loss =",
    average_loss
)

print(
    "Worst process loss =",
    worst_loss
)

print(
    "Soft worst-case loss =",
    final_soft_loss
)


print()
print(
    "Process loss matrix:"
)

print(
    loss_matrix
)


print()
print(
    "Process weight matrix:"
)

print(
    weight_matrix
)


# =========================================================
# 24. Nominal printed resist
# =========================================================

intensity_nominal, _, _ = optical_forward(
    mask_final,
    focus=0.0,
    source_shift=current_shift
)


resist_nominal = soft_resist(
    intensity_nominal
    /
    reference_peak
)


# =========================================================
# 25. Final comparison figure
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
    f"Final SMO Mask\nSource Shift = {current_shift:.2f}"
)

fig.colorbar(
    im,
    ax=axes[0, 2]
)


# ---------------------------------------------------------
# Printed resist
# ---------------------------------------------------------

im = axes[1, 0].imshow(
    resist_nominal,
    extent=extent,
    origin="lower",
    vmin=0,
    vmax=1
)

axes[1, 0].set_title(
    "Nominal Printed Resist"
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
# Weight map
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
# 26. SMO history
# =========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4)
)


axes[0].plot(
    np.arange(
        1,
        len(shift_history) + 1
    ),
    shift_history,
    marker="o"
)

axes[0].set_xlabel(
    "SMO Cycle"
)

axes[0].set_ylabel(
    "Best Source Shift (1/um)"
)

axes[0].set_title(
    "Source Evolution"
)

axes[0].grid(
    True
)


axes[1].plot(
    np.arange(
        1,
        len(loss_history_smo) + 1
    ),
    loss_history_smo,
    marker="o"
)

axes[1].set_xlabel(
    "SMO Cycle"
)

axes[1].set_ylabel(
    "Soft Robust Loss"
)

axes[1].set_title(
    "SMO Loss"
)

axes[1].grid(
    True
)


plt.tight_layout()

plt.show()