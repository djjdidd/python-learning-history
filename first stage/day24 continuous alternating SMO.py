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
# 3. Mask parameterization
#
# Z -> sigmoid -> mask
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


eps = 0.10

mask_initial = np.clip(
    target,
    eps,
    1.0 - eps
)

Z_initial = np.log(
    mask_initial
    /
    (
        1.0 - mask_initial
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
# 6. Resist parameters
# =========================================================

threshold = 0.5
beta = 20.0


def soft_resist(
    intensity
):

    argument = (
        -beta
        *
        (
            intensity
            -
            threshold
        )
    )

    argument = np.clip(
        argument,
        -60.0,
        60.0
    )

    return (
        1.0
        /
        (
            1.0
            +
            np.exp(argument)
        )
    )


# =========================================================
# 7. Soft pupil
#
# We use soft pupil because source position
# needs a differentiable optical model.
# =========================================================

alpha = 20.0
small_number = 1e-12


def soft_pupil_and_gradient(
    source_x,
    source_y,
    focus
):

    # -----------------------------------------------------
    # Shifted frequency coordinates
    # -----------------------------------------------------

    fx = FX - source_x
    fy = FY - source_y


    # -----------------------------------------------------
    # Radius
    # -----------------------------------------------------

    r = np.sqrt(
        fx**2
        +
        fy**2
        +
        small_number
    )


    # =====================================================
    # Soft aperture
    # =====================================================

    argument = (
        alpha
        *
        (
            r - cutoff
        )
    )

    argument = np.clip(
        argument,
        -60.0,
        60.0
    )

    aperture = (
        1.0
        /
        (
            1.0
            +
            np.exp(argument)
        )
    )


    # =====================================================
    # dA / d(source_x)
    # =====================================================

    dA_dsx = (
        alpha
        *
        aperture
        *
        (
            1.0 - aperture
        )
        *
        fx
        /
        r
    )


    # =====================================================
    # Defocus phase
    # =====================================================

    q = (
        fx**2
        +
        fy**2
    )

    phase = np.exp(
        -1j
        *
        np.pi
        *
        wavelength
        *
        focus
        *
        q
    )


    # =====================================================
    # dPhase / d(source_x)
    # =====================================================

    dphase_dsx = (
        1j
        *
        2.0
        *
        np.pi
        *
        wavelength
        *
        focus
        *
        fx
        *
        phase
    )


    # =====================================================
    # Complex pupil
    # =====================================================

    pupil = (
        aperture
        *
        phase
    )


    # =====================================================
    # dP / d(source_x)
    # =====================================================

    dpupil_dsx = (
        dA_dsx
        *
        phase
        +
        aperture
        *
        dphase_dsx
    )


    return (
        pupil,
        dpupil_dsx
    )


# =========================================================
# 8. Symmetric horizontal dipole
# =========================================================

def get_source_points(
    source_shift
):

    return [

        (
            +source_shift,
            0.0,
            0.5,
            +1.0
        ),

        (
            -source_shift,
            0.0,
            0.5,
            -1.0
        )
    ]


# =========================================================
# 9. Optical forward for mask optimization
#
# Return:
#
# total intensity
# field/pupil information for backward
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

    source_records = []


    for (
        source_x,
        source_y,
        weight,
        source_sign
    ) in get_source_points(
        source_shift
    ):

        (
            pupil,
            _
        ) = soft_pupil_and_gradient(
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


        source_records.append(
            (
                field,
                pupil,
                weight
            )
        )


    return (
        total_intensity,
        source_records
    )


# =========================================================
# 10. Optical forward + source gradient
# =========================================================

def optical_forward_source_gradient(
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

    dI_ds = np.zeros_like(
        mask,
        dtype=float
    )


    for (
        source_x,
        source_y,
        weight,
        source_sign
    ) in get_source_points(
        source_shift
    ):

        (
            pupil,
            dpupil_dsx
        ) = soft_pupil_and_gradient(
            source_x,
            source_y,
            focus
        )


        # -------------------------------------------------
        # dP/ds
        #
        # right source: +1
        # left source : -1
        # -------------------------------------------------

        dpupil_ds = (
            dpupil_dsx
            *
            source_sign
        )


        # -------------------------------------------------
        # Field
        # -------------------------------------------------

        field = np.fft.ifft2(
            spectrum
            *
            pupil
        )


        # -------------------------------------------------
        # dE/ds
        # -------------------------------------------------

        dfield_ds = np.fft.ifft2(
            spectrum
            *
            dpupil_ds
        )


        # -------------------------------------------------
        # Intensity
        # -------------------------------------------------

        total_intensity += (
            weight
            *
            np.abs(field)**2
        )


        # -------------------------------------------------
        # dI/ds
        # -------------------------------------------------

        dI_ds += (
            weight
            *
            2.0
            *
            np.real(
                np.conj(field)
                *
                dfield_ds
            )
        )


    return (
        total_intensity,
        dI_ds
    )


# =========================================================
# 11. Fixed normalization reference
# =========================================================
#
# IMPORTANT:
#
# It remains fixed during ALL optimization.
# =========================================================

reference_source_shift = 0.60


reference_intensity, _ = optical_forward(
    target,
    focus=0.0,
    source_shift=reference_source_shift
)


reference_peak = np.max(
    reference_intensity
)


print(
    "Reference peak =",
    reference_peak
)


# =========================================================
# 12. One process condition:
#     source loss + dL/ds
# =========================================================

def condition_source_gradient(
    mask,
    focus,
    dose,
    source_shift
):

    (
        intensity,
        dI_ds
    ) = optical_forward_source_gradient(
        mask,
        focus,
        source_shift
    )


    # =====================================================
    # Dose
    # =====================================================

    effective_intensity = (
        dose
        *
        intensity
        /
        reference_peak
    )


    # =====================================================
    # Resist
    # =====================================================

    resist = soft_resist(
        effective_intensity
    )


    # =====================================================
    # Loss
    # =====================================================

    loss = np.mean(
        (
            resist
            -
            target
        )**2
    )


    # =====================================================
    # dL/dR
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
    # dR/dIeff
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
    # dL/dI
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
    # dL/ds
    # =====================================================

    dL_ds = np.sum(
        dL_dI
        *
        dI_ds
    )


    return (
        loss,
        dL_ds
    )


# =========================================================
# 13. One process condition:
#     mask loss + dL/dZ
# =========================================================

def condition_mask_gradient(
    Z,
    focus,
    dose,
    source_shift
):

    mask = sigmoid(
        Z
    )


    (
        intensity,
        source_records
    ) = optical_forward(
        mask,
        focus,
        source_shift
    )


    # =====================================================
    # Dose
    # =====================================================

    effective_intensity = (
        dose
        *
        intensity
        /
        reference_peak
    )


    # =====================================================
    # Resist
    # =====================================================

    resist = soft_resist(
        effective_intensity
    )


    # =====================================================
    # Loss
    # =====================================================

    loss = np.mean(
        (
            resist
            -
            target
        )**2
    )


    # =====================================================
    # dL/dR
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
    # dR/dIeff
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
    # dL/dI
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
    # Backward through partial-coherent imaging
    # =====================================================

    dL_dmask = np.zeros_like(
        mask,
        dtype=float
    )


    for (
        field,
        pupil,
        weight
    ) in source_records:


        # -------------------------------------------------
        # dL/dE
        # -------------------------------------------------

        dL_dfield = (
            2.0
            *
            weight
            *
            dL_dI
            *
            field
        )


        # -------------------------------------------------
        # Optical adjoint
        # -------------------------------------------------

        mask_gradient_source = (
            np.fft.ifft2(
                np.fft.fft2(
                    dL_dfield
                )
                *
                np.conj(
                    pupil
                )
            )
        )


        dL_dmask += np.real(
            mask_gradient_source
        )


    # =====================================================
    # M = sigmoid(Z)
    #
    # dM/dZ = M(1-M)
    # =====================================================

    dL_dZ = (
        dL_dmask
        *
        mask
        *
        (
            1.0
            -
            mask
        )
    )


    return (
        loss,
        dL_dZ
    )


# =========================================================
# 14. Soft worst-case parameters
# =========================================================

tau = 0.002


def soft_worst_weights(
    losses
):

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


    robust_loss = (
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
        robust_loss,
        weights
    )


# =========================================================
# 15. Robust source gradient
# =========================================================

def robust_source_gradient(
    mask,
    source_shift
):

    losses = []
    gradients = []


    for focus in focus_list:

        for dose in dose_list:

            (
                loss,
                dL_ds
            ) = condition_source_gradient(
                mask,
                focus,
                dose,
                source_shift
            )


            losses.append(
                loss
            )

            gradients.append(
                dL_ds
            )


    losses = np.array(
        losses
    )

    gradients = np.array(
        gradients
    )


    (
        robust_loss,
        weights
    ) = soft_worst_weights(
        losses
    )


    robust_gradient = np.sum(
        weights
        *
        gradients
    )


    return (
        robust_loss,
        robust_gradient,
        losses,
        weights
    )


# =========================================================
# 16. Robust mask gradient
# =========================================================

def robust_mask_gradient(
    Z,
    source_shift
):

    losses = []
    gradients = []


    for focus in focus_list:

        for dose in dose_list:

            (
                loss,
                dL_dZ
            ) = condition_mask_gradient(
                Z,
                focus,
                dose,
                source_shift
            )


            losses.append(
                loss
            )

            gradients.append(
                dL_dZ
            )


    losses = np.array(
        losses
    )

    gradients = np.array(
        gradients
    )


    (
        robust_loss,
        weights
    ) = soft_worst_weights(
        losses
    )


    # =====================================================
    # Weighted sum over process conditions
    # =====================================================

    robust_gradient = np.zeros_like(
        Z
    )


    for k in range(
        len(weights)
    ):

        robust_gradient += (
            weights[k]
            *
            gradients[k]
        )


    return (
        robust_loss,
        robust_gradient,
        losses,
        weights
    )


# =========================================================
# 17. Evaluation function
# =========================================================

def evaluate_process(
    mask,
    source_shift
):

    losses = []

    nominal_resist = None
    nominal_loss = None


    for focus in focus_list:

        for dose in dose_list:

            (
                intensity,
                _
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


            loss = np.mean(
                (
                    resist
                    -
                    target
                )**2
            )


            losses.append(
                loss
            )


            if (
                focus == 0.0
                and
                dose == 1.0
            ):

                nominal_resist = resist.copy()
                nominal_loss = loss


    losses = np.array(
        losses
    )


    (
        soft_loss,
        weights
    ) = soft_worst_weights(
        losses
    )


    return (
        nominal_loss,
        np.mean(losses),
        np.max(losses),
        soft_loss,
        losses,
        weights,
        nominal_resist
    )


# =========================================================
# 18. Optimization parameters
# =========================================================

num_cycles = 5


# ---------------------------------------------------------
# Source optimization
# ---------------------------------------------------------

num_source_steps = 50

source_learning_rate = 3.0

source_gradient_tolerance = 1e-6


# ---------------------------------------------------------
# Mask optimization
# ---------------------------------------------------------

num_mask_steps = 30

mask_learning_rate = 10.0


# =========================================================
# 19. Initialize SMO
# =========================================================

Z = Z_initial.copy()

source_shift = 0.60


initial_mask = sigmoid(
    Z
)


# =========================================================
# History
# =========================================================

cycle_history = []

source_cycle_history = [
    source_shift
]

soft_loss_cycle_history = []

average_loss_cycle_history = []

worst_loss_cycle_history = []


source_step_history = []

source_value_history = []


# =========================================================
# Initial evaluation
# =========================================================

(
    nominal_initial,
    average_initial,
    worst_initial,
    soft_initial,
    losses_initial,
    weights_initial,
    resist_initial
) = evaluate_process(
    initial_mask,
    source_shift
)


print()
print("=" * 80)
print("INITIAL RESULT")
print("=" * 80)

print(
    "Source shift =",
    source_shift
)

print(
    "Nominal loss =",
    nominal_initial
)

print(
    "Average loss =",
    average_initial
)

print(
    "Worst loss   =",
    worst_initial
)

print(
    "Soft loss    =",
    soft_initial
)


# =========================================================
# 20. Continuous Alternating SMO
# =========================================================

global_source_step = 0


for cycle in range(
    num_cycles
):


    print()
    print("=" * 80)
    print(
        f"SMO CYCLE {cycle + 1}"
    )
    print("=" * 80)


    # =====================================================
    # PART A
    #
    # FIX MASK
    # OPTIMIZE SOURCE
    # =====================================================

    mask_fixed = sigmoid(
        Z
    )


    for source_iter in range(
        num_source_steps
    ):

        (
            source_robust_loss,
            dL_ds,
            source_losses,
            source_weights
        ) = robust_source_gradient(
            mask_fixed,
            source_shift
        )


        # -------------------------------------------------
        # Save source history
        # -------------------------------------------------

        source_step_history.append(
            global_source_step
        )

        source_value_history.append(
            source_shift
        )

        global_source_step += 1


        # -------------------------------------------------
        # Optional early stopping
        # -------------------------------------------------

        if (
            abs(dL_ds)
            <
            source_gradient_tolerance
        ):

            break


        # -------------------------------------------------
        # Gradient descent
        # -------------------------------------------------

        source_shift = (
            source_shift
            -
            source_learning_rate
            *
            dL_ds
        )


        # -------------------------------------------------
        # Keep source inside allowed range
        # -------------------------------------------------

        source_shift = np.clip(
            source_shift,
            0.0,
            1.2
        )


    print()
    print(
        "After SOURCE optimization:"
    )

    print(
        "Source shift =",
        source_shift
    )

    print(
        "Last source gradient =",
        dL_ds
    )

    print(
        "Soft robust loss =",
        source_robust_loss
    )


    # =====================================================
    # PART B
    #
    # FIX SOURCE
    # OPTIMIZE MASK
    # =====================================================

    for mask_iter in range(
        num_mask_steps
    ):

        (
            mask_robust_loss,
            dL_dZ,
            mask_losses,
            mask_weights
        ) = robust_mask_gradient(
            Z,
            source_shift
        )


        Z = (
            Z
            -
            mask_learning_rate
            *
            dL_dZ
        )


    # =====================================================
    # Evaluate complete cycle
    # =====================================================

    current_mask = sigmoid(
        Z
    )


    (
        nominal_loss,
        average_loss,
        worst_loss,
        soft_loss,
        condition_losses,
        process_weights,
        nominal_resist
    ) = evaluate_process(
        current_mask,
        source_shift
    )


    source_cycle_history.append(
        source_shift
    )

    soft_loss_cycle_history.append(
        soft_loss
    )

    average_loss_cycle_history.append(
        average_loss
    )

    worst_loss_cycle_history.append(
        worst_loss
    )


    cycle_history.append(
        (
            cycle + 1,
            source_shift,
            nominal_loss,
            average_loss,
            worst_loss,
            soft_loss
        )
    )


    print()
    print(
        "After MASK optimization:"
    )

    print(
        "Source shift =",
        source_shift
    )

    print(
        "Nominal loss =",
        nominal_loss
    )

    print(
        "Average loss =",
        average_loss
    )

    print(
        "Worst loss   =",
        worst_loss
    )

    print(
        "Soft loss    =",
        soft_loss
    )


# =========================================================
# 21. Final result
# =========================================================

final_mask = sigmoid(
    Z
)


(
    final_nominal_loss,
    final_average_loss,
    final_worst_loss,
    final_soft_loss,
    final_condition_losses,
    final_weights,
    final_nominal_resist
) = evaluate_process(
    final_mask,
    source_shift
)


loss_matrix = (
    final_condition_losses.reshape(
        len(focus_list),
        len(dose_list)
    )
)


weight_matrix = (
    final_weights.reshape(
        len(focus_list),
        len(dose_list)
    )
)


print()
print("=" * 80)
print("FINAL CONTINUOUS ALTERNATING SMO RESULT")
print("=" * 80)

print(
    "Initial source shift =",
    0.60
)

print(
    "Final source shift   =",
    source_shift
)

print()

print(
    "Initial soft loss =",
    soft_initial
)

print(
    "Final soft loss   =",
    final_soft_loss
)

print()

print(
    "Final nominal loss =",
    final_nominal_loss
)

print(
    "Final average loss =",
    final_average_loss
)

print(
    "Final worst loss =",
    final_worst_loss
)

print()

print(
    "Final process loss matrix:"
)

print(
    loss_matrix
)

print()

print(
    "Final soft-worst weight matrix:"
)

print(
    weight_matrix
)


# =========================================================
# 22. Print cycle summary
# =========================================================

print()
print("=" * 80)
print("CYCLE SUMMARY")
print("=" * 80)


for result in cycle_history:

    (
        cycle,
        shift,
        nominal,
        average,
        worst,
        soft
    ) = result


    print(
        f"Cycle {cycle} | "
        f"shift={shift:.6f} | "
        f"nominal={nominal:.8f} | "
        f"average={average:.8f} | "
        f"worst={worst:.8f} | "
        f"soft={soft:.8f}"
    )


# =========================================================
# 23. Plot
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


# =========================================================
# Target
# =========================================================

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


# =========================================================
# Initial mask
# =========================================================

im = axes[0, 1].imshow(
    initial_mask,
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


# =========================================================
# Final mask
# =========================================================

im = axes[0, 2].imshow(
    final_mask,
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


# =========================================================
# Final nominal resist
# =========================================================

im = axes[0, 3].imshow(
    final_nominal_resist,
    extent=extent,
    origin="lower",
    vmin=0,
    vmax=1
)

axes[0, 3].set_title(
    "Final Nominal Resist"
)

fig.colorbar(
    im,
    ax=axes[0, 3]
)


# =========================================================
# Continuous source movement
# =========================================================

axes[1, 0].plot(
    source_step_history,
    source_value_history
)

axes[1, 0].set_title(
    "Continuous Source Optimization"
)

axes[1, 0].set_xlabel(
    "Source Update Step"
)

axes[1, 0].set_ylabel(
    "Source Shift (1/um)"
)

axes[1, 0].grid(
    True
)


# =========================================================
# Source after each cycle
# =========================================================

axes[1, 1].plot(
    np.arange(
        len(source_cycle_history)
    ),
    source_cycle_history,
    marker="o"
)

axes[1, 1].set_title(
    "Source Shift per SMO Cycle"
)

axes[1, 1].set_xlabel(
    "Cycle"
)

axes[1, 1].set_ylabel(
    "Source Shift (1/um)"
)

axes[1, 1].grid(
    True
)


# =========================================================
# Process loss history
# =========================================================

cycle_axis = np.arange(
    1,
    num_cycles + 1
)


axes[1, 2].plot(
    cycle_axis,
    soft_loss_cycle_history,
    marker="o",
    label="Soft Worst"
)

axes[1, 2].plot(
    cycle_axis,
    average_loss_cycle_history,
    marker="o",
    label="Average"
)

axes[1, 2].plot(
    cycle_axis,
    worst_loss_cycle_history,
    marker="o",
    label="Worst"
)

axes[1, 2].set_title(
    "Loss per SMO Cycle"
)

axes[1, 2].set_xlabel(
    "SMO Cycle"
)

axes[1, 2].set_ylabel(
    "Loss"
)

axes[1, 2].legend()

axes[1, 2].grid(
    True
)


# =========================================================
# Final process weight map
# =========================================================

im = axes[1, 3].imshow(
    weight_matrix,
    origin="lower",
    aspect="auto"
)

axes[1, 3].set_title(
    "Final Soft-Worst Weights"
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


plt.tight_layout()

plt.show()

# =========================================================
# 24. Compare source optimum before and after mask optimization
# =========================================================

source_scan = np.arange(
    0.0,
    0.3001,
    0.005
)


initial_mask_scan_loss = []
final_mask_scan_loss = []


for s in source_scan:

    # =====================================================
    # Initial mask
    # =====================================================

    (
        robust_loss_initial,
        _,
        _,
        _
    ) = robust_source_gradient(
        initial_mask,
        s
    )

    initial_mask_scan_loss.append(
        robust_loss_initial
    )


    # =====================================================
    # Final mask
    # =====================================================

    (
        robust_loss_final,
        _,
        _,
        _
    ) = robust_source_gradient(
        final_mask,
        s
    )

    final_mask_scan_loss.append(
        robust_loss_final
    )


initial_mask_scan_loss = np.array(
    initial_mask_scan_loss
)

final_mask_scan_loss = np.array(
    final_mask_scan_loss
)


# =========================================================
# Best source for initial mask
# =========================================================

initial_best_index = np.argmin(
    initial_mask_scan_loss
)

initial_best_source = source_scan[
    initial_best_index
]


# =========================================================
# Best source for final mask
# =========================================================

final_best_index = np.argmin(
    final_mask_scan_loss
)

final_best_source = source_scan[
    final_best_index
]


print()
print("=" * 80)
print("SOURCE-MASK COUPLING CHECK")
print("=" * 80)

print(
    "Best source for INITIAL mask =",
    initial_best_source
)

print(
    "Best source for FINAL mask   =",
    final_best_source
)


# =========================================================
# Plot
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    source_scan,
    initial_mask_scan_loss,
    label="Initial Mask"
)

plt.plot(
    source_scan,
    final_mask_scan_loss,
    label="Final Mask"
)

plt.axvline(
    initial_best_source,
    linestyle="--"
)

plt.axvline(
    final_best_source,
    linestyle="--"
)

plt.xlabel(
    "Source Shift (1/um)"
)

plt.ylabel(
    "Soft-Worst Loss"
)

plt.title(
    "Source Optimum Before and After Mask Optimization"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.show()