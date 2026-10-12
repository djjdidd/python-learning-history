import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Spatial grid
# ============================================================

N = 128
L = 2.0                     # um
dx = L / N

x = np.arange(N) * dx - L / 2
y = np.arange(N) * dx - L / 2

X, Y = np.meshgrid(x, y)


# ============================================================
# 2. Target pattern
# ============================================================

target_width_x = 0.60
target_width_y = 0.40

target = (
    (np.abs(X) <= target_width_x / 2)
    &
    (np.abs(Y) <= target_width_y / 2)
).astype(float)


# ============================================================
# 3. Mask parameterization
#
# Z -> sigmoid(Z) -> mask
# ============================================================

def sigmoid(z):

    z = np.clip(
        z,
        -60.0,
        60.0
    )

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


# ============================================================
# 4. Optical parameters
# ============================================================

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


# ============================================================
# 5. Process window
# ============================================================

focus_list = np.array([
    -0.05,
     0.00,
     0.05
])

dose_list = np.array([
    0.95,
    1.00,
    1.05
])


# ============================================================
# 6. Soft resist
# ============================================================

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


# ============================================================
# 7. Soft pupil
#
# Soft pupil is used so that source position is differentiable.
# ============================================================

alpha = 20.0
small_number = 1e-12


def soft_pupil_and_gradient(
    source_x,
    source_y,
    focus
):

    # --------------------------------------------------------
    # Shifted frequency coordinates
    # --------------------------------------------------------

    fx = FX - source_x
    fy = FY - source_y


    # --------------------------------------------------------
    # Radius
    # --------------------------------------------------------

    r = np.sqrt(
        fx**2
        +
        fy**2
        +
        small_number
    )


    # ========================================================
    # Soft aperture
    #
    # A =
    #
    # 1 / [1 + exp(alpha * (r - cutoff))]
    # ========================================================

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


    # ========================================================
    # dA / d(source_x)
    # ========================================================

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


    # ========================================================
    # Defocus phase
    # ========================================================

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


    # ========================================================
    # dPhase / d(source_x)
    # ========================================================

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


    # ========================================================
    # Complete pupil
    # ========================================================

    pupil = (
        aperture
        *
        phase
    )


    # ========================================================
    # dP / d(source_x)
    #
    # Product rule
    # ========================================================

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


# ============================================================
# 8. Symmetric horizontal dipole
#
# source 1 = (+s, 0)
# source 2 = (-s, 0)
# ============================================================

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


# ============================================================
# 9. Normal optical forward
#
# Used for mask gradient and evaluation.
# ============================================================

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
        source_weight,
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
            source_weight
            *
            np.abs(field)**2
        )


        source_records.append(
            (
                field,
                pupil,
                source_weight
            )
        )


    return (
        total_intensity,
        source_records
    )


# ============================================================
# 10. Optical forward with source gradient
#
# Computes:
#
# I
#
# and
#
# dI / ds
# ============================================================

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
        source_weight,
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


        # ====================================================
        # source_x = +s or -s
        #
        # dP/ds =
        #
        # dP/d(source_x) * d(source_x)/ds
        # ====================================================

        dpupil_ds = (
            dpupil_dsx
            *
            source_sign
        )


        # ====================================================
        # Field
        # ====================================================

        field = np.fft.ifft2(
            spectrum
            *
            pupil
        )


        # ====================================================
        # dE / ds
        # ====================================================

        dfield_ds = np.fft.ifft2(
            spectrum
            *
            dpupil_ds
        )


        # ====================================================
        # Intensity
        # ====================================================

        total_intensity += (
            source_weight
            *
            np.abs(field)**2
        )


        # ====================================================
        # dI/ds
        #
        # = 2 Re(E* dE/ds)
        # ====================================================

        dI_ds += (
            source_weight
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


# ============================================================
# 11. Fixed reference normalization
#
# IMPORTANT:
# This value remains fixed for ALL four experiments.
# ============================================================

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


# ============================================================
# 12. One condition:
#     source loss + source gradient
# ============================================================

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


    # ========================================================
    # Dose
    # ========================================================

    effective_intensity = (
        dose
        *
        intensity
        /
        reference_peak
    )


    # ========================================================
    # Resist
    # ========================================================

    resist = soft_resist(
        effective_intensity
    )


    # ========================================================
    # Loss
    # ========================================================

    loss = np.mean(
        (
            resist
            -
            target
        )**2
    )


    # ========================================================
    # dL / dR
    # ========================================================

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


    # ========================================================
    # dR / dIeff
    # ========================================================

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


    # ========================================================
    # dL / dI
    # ========================================================

    dL_dI = (
        dL_dR
        *
        dR_dIeff
        *
        dose
        /
        reference_peak
    )


    # ========================================================
    # dL / ds
    # ========================================================

    dL_ds = np.sum(
        dL_dI
        *
        dI_ds
    )


    return (
        loss,
        dL_ds
    )


# ============================================================
# 13. One condition:
#     mask loss + mask gradient
# ============================================================

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


    # ========================================================
    # Effective intensity
    # ========================================================

    effective_intensity = (
        dose
        *
        intensity
        /
        reference_peak
    )


    # ========================================================
    # Resist
    # ========================================================

    resist = soft_resist(
        effective_intensity
    )


    # ========================================================
    # Loss
    # ========================================================

    loss = np.mean(
        (
            resist
            -
            target
        )**2
    )


    # ========================================================
    # dL / dR
    # ========================================================

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


    # ========================================================
    # dR / dIeff
    # ========================================================

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


    # ========================================================
    # dL / dI
    # ========================================================

    dL_dI = (
        dL_dR
        *
        dR_dIeff
        *
        dose
        /
        reference_peak
    )


    # ========================================================
    # Backward through optical system
    # ========================================================

    dL_dmask = np.zeros_like(
        mask,
        dtype=float
    )


    for (
        field,
        pupil,
        source_weight
    ) in source_records:


        # ====================================================
        # dL / dE
        # ====================================================

        dL_dfield = (
            2.0
            *
            source_weight
            *
            dL_dI
            *
            field
        )


        # ====================================================
        # Optical adjoint
        # ====================================================

        gradient_source = np.fft.ifft2(
            np.fft.fft2(
                dL_dfield
            )
            *
            np.conj(
                pupil
            )
        )


        dL_dmask += np.real(
            gradient_source
        )


    # ========================================================
    # M = sigmoid(Z)
    #
    # dM/dZ = M(1-M)
    # ========================================================

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


# ============================================================
# 14. Soft worst-case objective
# ============================================================

tau = 0.002


def soft_worst(
    losses
):

    losses = np.asarray(
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


# ============================================================
# 15. Robust source gradient
# ============================================================

def robust_source_gradient(
    mask,
    source_shift
):

    condition_losses = []
    condition_gradients = []


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


            condition_losses.append(
                loss
            )

            condition_gradients.append(
                dL_ds
            )


    condition_losses = np.array(
        condition_losses
    )

    condition_gradients = np.array(
        condition_gradients
    )


    (
        robust_loss,
        weights
    ) = soft_worst(
        condition_losses
    )


    robust_gradient = np.sum(
        weights
        *
        condition_gradients
    )


    return (
        robust_loss,
        robust_gradient
    )


# ============================================================
# 16. Robust mask gradient
# ============================================================

def robust_mask_gradient(
    Z,
    source_shift
):

    condition_losses = []
    condition_gradients = []


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


            condition_losses.append(
                loss
            )

            condition_gradients.append(
                dL_dZ
            )


    condition_losses = np.array(
        condition_losses
    )

    condition_gradients = np.array(
        condition_gradients
    )


    (
        robust_loss,
        weights
    ) = soft_worst(
        condition_losses
    )


    robust_gradient = np.zeros_like(
        Z
    )


    for k in range(
        len(weights)
    ):

        robust_gradient += (
            weights[k]
            *
            condition_gradients[k]
        )


    return (
        robust_loss,
        robust_gradient
    )


# ============================================================
# 17. Process-window evaluation
# ============================================================

def evaluate_process(
    mask,
    source_shift
):

    losses = []

    nominal_loss = None
    nominal_resist = None


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
                np.isclose(
                    focus,
                    0.0
                )
                and
                np.isclose(
                    dose,
                    1.0
                )
            ):

                nominal_loss = loss

                nominal_resist = resist.copy()


    losses = np.array(
        losses
    )


    (
        robust_loss,
        weights
    ) = soft_worst(
        losses
    )


    return {
        "source": source_shift,

        "nominal": nominal_loss,

        "average": np.mean(
            losses
        ),

        "worst": np.max(
            losses
        ),

        "soft": robust_loss,

        "losses": losses,

        "weights": weights,

        "resist": nominal_resist
    }


# ============================================================
# 18. Optimization settings
# ============================================================

initial_source = 0.60


# ------------------------------------------------------------
# Source optimizer
# ------------------------------------------------------------

source_learning_rate = 3.0

source_gradient_tolerance = 1e-6


# ------------------------------------------------------------
# Mask optimizer
# ------------------------------------------------------------

mask_learning_rate = 10.0


# ------------------------------------------------------------
# SMO structure
# ------------------------------------------------------------

num_cycles = 5

source_steps_per_cycle = 50

mask_steps_per_cycle = 30


# ============================================================
# 19. INITIAL
# ============================================================

Z0 = Z_initial.copy()

mask0 = sigmoid(
    Z0
)


initial_result = evaluate_process(
    mask0,
    initial_source
)


# ============================================================
# 20. SO ONLY
#
# Fixed mask
# Only optimize source
#
# Total source steps:
#
# 5 cycles * 50 = 250
# ============================================================

mask_so = mask0.copy()

source_so = initial_source


num_so_steps = (
    num_cycles
    *
    source_steps_per_cycle
)


so_source_history = [
    source_so
]


for iteration in range(
    num_so_steps
):

    (
        robust_loss,
        dL_ds
    ) = robust_source_gradient(
        mask_so,
        source_so
    )


    if (
        abs(dL_ds)
        <
        source_gradient_tolerance
    ):

        break


    source_so = (
        source_so
        -
        source_learning_rate
        *
        dL_ds
    )


    source_so = np.clip(
        source_so,
        0.0,
        1.2
    )


    so_source_history.append(
        source_so
    )


so_result = evaluate_process(
    mask_so,
    source_so
)


# ============================================================
# 21. MO ONLY
#
# Fixed source = 0.60
# Only optimize mask
#
# Total mask steps:
#
# 5 cycles * 30 = 150
# ============================================================

Z_mo = Z0.copy()

source_mo = initial_source


num_mo_steps = (
    num_cycles
    *
    mask_steps_per_cycle
)


mo_loss_history = []


for iteration in range(
    num_mo_steps
):

    (
        robust_loss,
        dL_dZ
    ) = robust_mask_gradient(
        Z_mo,
        source_mo
    )


    mo_loss_history.append(
        robust_loss
    )


    Z_mo = (
        Z_mo
        -
        mask_learning_rate
        *
        dL_dZ
    )


mask_mo = sigmoid(
    Z_mo
)


mo_result = evaluate_process(
    mask_mo,
    source_mo
)


# ============================================================
# 22. ALTERNATING SMO
#
# Fix mask -> optimize source
#
# Fix source -> optimize mask
#
# Repeat
# ============================================================

Z_smo = Z0.copy()

source_smo = initial_source


smo_source_cycle_history = [
    source_smo
]

smo_soft_cycle_history = []


for cycle in range(
    num_cycles
):

    # ========================================================
    # A. FIX MASK -> OPTIMIZE SOURCE
    # ========================================================

    mask_fixed = sigmoid(
        Z_smo
    )


    for iteration in range(
        source_steps_per_cycle
    ):

        (
            robust_loss,
            dL_ds
        ) = robust_source_gradient(
            mask_fixed,
            source_smo
        )


        if (
            abs(dL_ds)
            <
            source_gradient_tolerance
        ):

            break


        source_smo = (
            source_smo
            -
            source_learning_rate
            *
            dL_ds
        )


        source_smo = np.clip(
            source_smo,
            0.0,
            1.2
        )


    # ========================================================
    # B. FIX SOURCE -> OPTIMIZE MASK
    # ========================================================

    for iteration in range(
        mask_steps_per_cycle
    ):

        (
            robust_loss,
            dL_dZ
        ) = robust_mask_gradient(
            Z_smo,
            source_smo
        )


        Z_smo = (
            Z_smo
            -
            mask_learning_rate
            *
            dL_dZ
        )


    # ========================================================
    # Evaluate this SMO cycle
    # ========================================================

    current_mask = sigmoid(
        Z_smo
    )


    current_result = evaluate_process(
        current_mask,
        source_smo
    )


    smo_source_cycle_history.append(
        source_smo
    )


    smo_soft_cycle_history.append(
        current_result[
            "soft"
        ]
    )


    print(
        f"SMO Cycle {cycle + 1} | "
        f"source = {source_smo:.6f} | "
        f"soft = {current_result['soft']:.8f}"
    )


mask_smo = sigmoid(
    Z_smo
)


smo_result = evaluate_process(
    mask_smo,
    source_smo
)


# ============================================================
# 23. Print comparison table
# ============================================================

print()
print("=" * 100)
print("INITIAL / SO / MO / SMO COMPARISON")
print("=" * 100)


print(
    f"{'Method':<12}"
    f"{'Source':>12}"
    f"{'Nominal':>16}"
    f"{'Average':>16}"
    f"{'Worst':>16}"
    f"{'Soft-Worst':>16}"
)


print("-" * 100)


results = {

    "Initial": initial_result,

    "SO only": so_result,

    "MO only": mo_result,

    "SMO": smo_result
}


for name, result in results.items():

    print(

        f"{name:<12}"

        f"{result['source']:>12.6f}"

        f"{result['nominal']:>16.8f}"

        f"{result['average']:>16.8f}"

        f"{result['worst']:>16.8f}"

        f"{result['soft']:>16.8f}"
    )


# ============================================================
# 24. Improvement relative to Initial
# ============================================================

print()
print("=" * 100)
print("SOFT-WORST IMPROVEMENT RELATIVE TO INITIAL")
print("=" * 100)


initial_soft = initial_result[
    "soft"
]


for name in [
    "SO only",
    "MO only",
    "SMO"
]:

    improvement = (
        initial_soft
        -
        results[name][
            "soft"
        ]
    )


    percentage = (
        improvement
        /
        initial_soft
        *
        100.0
    )


    print(

        f"{name:<12}"

        f" improvement = {improvement:.10f}"

        f"   ({percentage:.3f} %)"
    )


# ============================================================
# 25. Final process-loss matrices
# ============================================================

print()
print("=" * 100)
print("PROCESS LOSS MATRICES")
print("=" * 100)


for name, result in results.items():

    matrix = (
        result[
            "losses"
        ]
        .reshape(
            len(focus_list),
            len(dose_list)
        )
    )


    print()
    print(
        name
    )

    print(
        matrix
    )


# ============================================================
# 26. Final soft-worst weight matrices
# ============================================================

print()
print("=" * 100)
print("FINAL SOFT-WORST WEIGHT MATRICES")
print("=" * 100)


for name, result in results.items():

    matrix = (
        result[
            "weights"
        ]
        .reshape(
            len(focus_list),
            len(dose_list)
        )
    )


    print()
    print(
        name
    )

    print(
        matrix
    )


# ============================================================
# 27. Visualization
# ============================================================

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


# ============================================================
# Top row: Masks
# ============================================================

mask_list = [

    mask0,

    mask_so,

    mask_mo,

    mask_smo
]


title_list = [

    "Initial",

    "SO Only",

    "MO Only",

    "SMO"
]


for i in range(
    4
):

    im = axes[0, i].imshow(

        mask_list[i],

        extent=extent,

        origin="lower",

        vmin=0,

        vmax=1
    )


    axes[0, i].set_title(
        title_list[i]
        +
        " Mask"
    )


    axes[0, i].set_xlabel(
        "x (um)"
    )

    axes[0, i].set_ylabel(
        "y (um)"
    )


    fig.colorbar(
        im,
        ax=axes[0, i]
    )


# ============================================================
# Bottom row: Nominal resist
# ============================================================

resist_list = [

    initial_result[
        "resist"
    ],

    so_result[
        "resist"
    ],

    mo_result[
        "resist"
    ],

    smo_result[
        "resist"
    ]
]


for i in range(
    4
):

    im = axes[1, i].imshow(

        resist_list[i],

        extent=extent,

        origin="lower",

        vmin=0,

        vmax=1
    )


    # Target contour
    axes[1, i].contour(

        X,

        Y,

        target,

        levels=[0.5],

        linewidths=1.2
    )


    axes[1, i].set_title(
        title_list[i]
        +
        " Nominal Resist"
    )


    axes[1, i].set_xlabel(
        "x (um)"
    )

    axes[1, i].set_ylabel(
        "y (um)"
    )


    fig.colorbar(
        im,
        ax=axes[1, i]
    )


plt.tight_layout()

plt.show()


# ============================================================
# 28. Loss comparison
# ============================================================

method_names = [
    "Initial",
    "SO",
    "MO",
    "SMO"
]


soft_values = [

    initial_result[
        "soft"
    ],

    so_result[
        "soft"
    ],

    mo_result[
        "soft"
    ],

    smo_result[
        "soft"
    ]
]


worst_values = [

    initial_result[
        "worst"
    ],

    so_result[
        "worst"
    ],

    mo_result[
        "worst"
    ],

    smo_result[
        "worst"
    ]
]


average_values = [

    initial_result[
        "average"
    ],

    so_result[
        "average"
    ],

    mo_result[
        "average"
    ],

    smo_result[
        "average"
    ]
]


positions = np.arange(
    len(
        method_names
    )
)


width = 0.25


plt.figure(
    figsize=(9, 5)
)


plt.bar(

    positions - width,

    average_values,

    width,

    label="Average"
)


plt.bar(

    positions,

    soft_values,

    width,

    label="Soft-Worst"
)


plt.bar(

    positions + width,

    worst_values,

    width,

    label="Worst"
)


plt.xticks(
    positions,
    method_names
)


plt.ylabel(
    "Loss"
)


plt.title(
    "Initial vs SO vs MO vs SMO"
)


plt.legend()


plt.grid(
    axis="y"
)


plt.tight_layout()

plt.show()


# ============================================================
# 29. Source comparison
# ============================================================

source_values = [

    initial_source,

    source_so,

    source_mo,

    source_smo
]


plt.figure(
    figsize=(8, 5)
)


plt.bar(
    method_names,
    source_values
)


plt.ylabel(
    "Source Shift (1/um)"
)


plt.title(
    "Final Source Shift"
)


plt.grid(
    axis="y"
)


plt.tight_layout()

plt.show()


# ============================================================
# 30. SO continuous source history
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    so_source_history
)


plt.xlabel(
    "Source Optimization Step"
)


plt.ylabel(
    "Source Shift (1/um)"
)


plt.title(
    "SO Only: Continuous Source Optimization"
)


plt.grid(
    True
)


plt.tight_layout()

plt.show()


# ============================================================
# 31. SMO source evolution
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    np.arange(
        len(
            smo_source_cycle_history
        )
    ),

    smo_source_cycle_history,

    marker="o"
)


plt.xlabel(
    "SMO Cycle"
)


plt.ylabel(
    "Source Shift (1/um)"
)


plt.title(
    "Source Shift During Alternating SMO"
)


plt.grid(
    True
)


plt.tight_layout()

plt.show()