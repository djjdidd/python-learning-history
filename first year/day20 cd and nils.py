import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Basic parameters
# =========================================================

N = 256
L = 4.0
dx = L / N

x = np.arange(N) * dx - L / 2
y = np.arange(N) * dx - L / 2

X, Y = np.meshgrid(x, y)

wavelength = 0.193
NA = 0.85

cutoff = NA / wavelength


# =========================================================
# 2. 2D periodic line-space mask
# =========================================================

period = 0.20
line_width = 0.10

position_in_period = (
    np.mod(
        X + period / 2,
        period
    )
    -
    period / 2
)

mask = (
    np.abs(position_in_period)
    <= line_width / 2
).astype(float)

target_cd = line_width


# =========================================================
# 3. 2D frequency grid
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
# 4. Shifted circular pupil
# =========================================================

def make_shifted_pupil_2d(
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
# 5. Partial-coherent imaging
# =========================================================

def partial_coherent_image_2d(
    mask_input,
    source_positions,
    source_weights
):

    spectrum = np.fft.fft2(
        mask_input
    )

    intensity = np.zeros(
        (N, N)
    )

    for (source_x, source_y), weight in zip(
        source_positions,
        source_weights
    ):

        pupil = make_shifted_pupil_2d(
            source_x,
            source_y
        )

        field = np.fft.ifft2(
            spectrum
            *
            pupil
        )

        intensity += (
            weight
            *
            np.abs(field) ** 2
        )

    return intensity


# =========================================================
# 6. Soft resist
# =========================================================

def soft_resist(
    intensity,
    threshold=0.5,
    beta=20.0
):

    resist = (
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

    return resist


threshold = 0.5
beta = 20.0


# =========================================================
# 7. Fixed reference intensity
# =========================================================

normal_positions = [
    (0.0, 0.0)
]

normal_weights = [
    1.0
]

intensity_normal = partial_coherent_image_2d(
    mask,
    normal_positions,
    normal_weights
)

reference_peak = np.max(
    intensity_normal
)


# =========================================================
# 8. Subpixel interpolation
# =========================================================

def interpolate_edge(
    x1,
    x2,
    y1,
    y2,
    threshold
):

    if y2 == y1:
        return 0.5 * (x1 + x2)

    edge_position = (
        x1
        +
        (threshold - y1)
        /
        (y2 - y1)
        *
        (x2 - x1)
    )

    return edge_position


# =========================================================
# 9. Measure CD
# =========================================================

def measure_cd(
    coordinate,
    resist_profile,
    threshold=0.5
):

    binary = (
        resist_profile
        >= threshold
    ).astype(float)

    transition = np.diff(
        binary
    )

    left_edges = np.where(
        transition == 1
    )[0]

    right_edges = np.where(
        transition == -1
    )[0]

    center_index = np.argmin(
        np.abs(coordinate)
    )

    left_candidates = left_edges[
        left_edges < center_index
    ]

    right_candidates = right_edges[
        right_edges >= center_index
    ]

    if (
        len(left_candidates) == 0
        or
        len(right_candidates) == 0
    ):
        return np.nan, np.nan, np.nan


    left_index = left_candidates[-1]
    right_index = right_candidates[0]


    left_position = interpolate_edge(
        coordinate[left_index],
        coordinate[left_index + 1],
        resist_profile[left_index],
        resist_profile[left_index + 1],
        threshold
    )


    right_position = interpolate_edge(
        coordinate[right_index],
        coordinate[right_index + 1],
        resist_profile[right_index],
        resist_profile[right_index + 1],
        threshold
    )


    cd = (
        right_position
        -
        left_position
    )

    return (
        cd,
        left_position,
        right_position
    )


# =========================================================
# 10. Center row
# =========================================================

center_y_index = np.argmin(
    np.abs(y)
)


# =========================================================
# 11. Source-shift and dose ranges
# =========================================================

source_shift_list = np.linspace(
    0.0,
    2.0,
    41
)

dose_list = np.linspace(
    0.3,
    2.5,
    441
)


best_dose_list = []
best_cd_error_list = []
best_cd_list = []


# =========================================================
# 12. Scan source shift
# =========================================================

for source_shift in source_shift_list:

    # Horizontal dipole
    source_positions = [
        (+source_shift, 0.0),
        (-source_shift, 0.0)
    ]

    source_weights = [
        0.5,
        0.5
    ]


    # -----------------------------------------------------
    # Aerial image
    # -----------------------------------------------------

    intensity = partial_coherent_image_2d(
        mask,
        source_positions,
        source_weights
    )


    # Fixed normalization
    intensity_norm = (
        intensity
        /
        reference_peak
    )


    # -----------------------------------------------------
    # Search dose
    # -----------------------------------------------------

    best_error = np.inf
    best_dose = np.nan
    best_cd = np.nan


    for dose in dose_list:

        # Dose scales aerial intensity
        effective_intensity = (
            dose
            *
            intensity_norm
        )


        # Resist response
        resist = soft_resist(
            effective_intensity,
            threshold,
            beta
        )


        # Center horizontal profile
        resist_profile = resist[
            center_y_index,
            :
        ]


        # Measure CD
        current_cd, _, _ = measure_cd(
            x,
            resist_profile,
            threshold=0.5
        )


        if np.isnan(current_cd):
            continue


        cd_error_nm = (
            current_cd
            -
            target_cd
        ) * 1000


        # Keep best dose for this source shift
        if abs(cd_error_nm) < best_error:

            best_error = abs(
                cd_error_nm
            )

            best_dose = dose

            best_cd = (
                current_cd
                *
                1000
            )


    best_dose_list.append(
        best_dose
    )

    best_cd_error_list.append(
        best_error
    )

    best_cd_list.append(
        best_cd
    )


    print(
        f"Shift = {source_shift:.2f}  "
        f"Best dose = {best_dose:.3f}  "
        f"CD = {best_cd:.3f} nm  "
        f"|CD error| = {best_error:.3f} nm"
    )


# =========================================================
# 13. Find overall best combination
# =========================================================

best_cd_error_array = np.array(
    best_cd_error_list
)

best_index = np.nanargmin(
    best_cd_error_array
)


best_source_shift = source_shift_list[
    best_index
]

best_dose = best_dose_list[
    best_index
]

best_cd = best_cd_list[
    best_index
]

best_cd_error = best_cd_error_list[
    best_index
]


print()
print("=" * 60)
print("Best Source + Dose Combination")
print("=" * 60)

print(
    "Target CD =",
    target_cd * 1000,
    "nm"
)

print(
    "Best source shift =",
    best_source_shift,
    "1/um"
)

print(
    "Best dose =",
    best_dose
)

print(
    "Printed CD =",
    best_cd,
    "nm"
)

print(
    "Minimum |CD error| =",
    best_cd_error,
    "nm"
)


# =========================================================
# 14. Plot minimum CD error vs source shift
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    source_shift_list,
    best_cd_error_list,
    marker="o"
)

plt.xlabel(
    "Source Shift (1/um)"
)

plt.ylabel(
    "Minimum |CD Error| (nm)"
)

plt.title(
    "CD Error after Dose Optimization"
)

plt.grid()

plt.show()
# =========================================================
# 15. Contrast
# =========================================================

def measure_contrast(
    intensity_profile
):

    Imax = np.max(
        intensity_profile
    )

    Imin = np.min(
        intensity_profile
    )

    return (
        (Imax - Imin)
        /
        (Imax + Imin)
    )


# =========================================================
# 16. NILS
# =========================================================

def measure_nils(
    coordinate,
    intensity_profile,
    threshold=0.5
):

    # Use the same threshold crossing
    cd, left_edge, right_edge = measure_cd(
        coordinate,
        intensity_profile,
        threshold
    )

    if np.isnan(cd):
        return np.nan


    # Intensity slope
    gradient = np.gradient(
        intensity_profile,
        coordinate
    )


    # Interpolate gradient at true edge positions
    left_gradient = np.interp(
        left_edge,
        coordinate,
        gradient
    )

    right_gradient = np.interp(
        right_edge,
        coordinate,
        gradient
    )


    left_nils = (
        cd
        /
        threshold
        *
        np.abs(left_gradient)
    )

    right_nils = (
        cd
        /
        threshold
        *
        np.abs(right_gradient)
    )


    return 0.5 * (
        left_nils
        +
        right_nils
    )
# =========================================================
# 17. Compare source quality after CD calibration
# =========================================================

contrast_after_cd = []
nils_after_cd = []


for source_shift, best_dose in zip(
    source_shift_list,
    best_dose_list
):

    source_positions = [
        (+source_shift, 0.0),
        (-source_shift, 0.0)
    ]

    source_weights = [
        0.5,
        0.5
    ]


    # Aerial image
    intensity = partial_coherent_image_2d(
        mask,
        source_positions,
        source_weights
    )


    intensity_norm = (
        intensity
        /
        reference_peak
    )


    # Use the best dose found before
    effective_intensity = (
        best_dose
        *
        intensity_norm
    )


    profile = effective_intensity[
        center_y_index,
        :
    ]


    current_contrast = measure_contrast(
        profile
    )

    current_nils = measure_nils(
        x,
        profile,
        threshold=0.5
    )


    contrast_after_cd.append(
        current_contrast
    )

    nils_after_cd.append(
        current_nils
    )
# =========================================================
# 18. Compare only CD-feasible solutions
# =========================================================

cd_tolerance_nm = 2.0

feasible_indices = np.where(
    np.array(best_cd_error_list)
    <= cd_tolerance_nm
)[0]


print()
print("=" * 60)
print("CD-Calibrated Illumination Comparison")
print("=" * 60)

print(
    "Number of feasible source shifts =",
    len(feasible_indices)
)


if len(feasible_indices) > 0:

    best_contrast_index = feasible_indices[
        np.argmax(
            np.array(contrast_after_cd)[
                feasible_indices
            ]
        )
    ]

    best_nils_index = feasible_indices[
        np.argmax(
            np.array(nils_after_cd)[
                feasible_indices
            ]
        )
    ]


    print()
    print("Best contrast among CD-feasible sources")

    print(
        "Source shift =",
        source_shift_list[
            best_contrast_index
        ],
        "1/um"
    )

    print(
        "Dose =",
        best_dose_list[
            best_contrast_index
        ]
    )

    print(
        "|CD error| =",
        best_cd_error_list[
            best_contrast_index
        ],
        "nm"
    )

    print(
        "Contrast =",
        contrast_after_cd[
            best_contrast_index
        ]
    )


    print()
    print("Best NILS among CD-feasible sources")

    print(
        "Source shift =",
        source_shift_list[
            best_nils_index
        ],
        "1/um"
    )

    print(
        "Dose =",
        best_dose_list[
            best_nils_index
        ]
    )

    print(
        "|CD error| =",
        best_cd_error_list[
            best_nils_index
        ],
        "nm"
    )

    print(
        "NILS =",
        nils_after_cd[
            best_nils_index
        ]
    )

else:

    print(
        "No source satisfies the CD tolerance."
    )
# =========================================================
# Dose sensitivity test
# =========================================================

candidate_sources = [
    {
        "shift": 0.60,
        "nominal_dose": 0.390
    },
    {
        "shift": 0.70,
        "nominal_dose": 0.385
    }
]


dose_factors = np.array([
    0.95,
    0.975,
    1.00,
    1.025,
    1.05
])


plt.figure(figsize=(8, 5))


for candidate in candidate_sources:

    source_shift = candidate["shift"]
    nominal_dose = candidate["nominal_dose"]


    # -----------------------------------------
    # Build dipole source
    # -----------------------------------------

    source_positions = [
        (+source_shift, 0.0),
        (-source_shift, 0.0)
    ]

    source_weights = [
        0.5,
        0.5
    ]


    # -----------------------------------------
    # Aerial image
    # -----------------------------------------

    intensity = partial_coherent_image_2d(
        mask,
        source_positions,
        source_weights
    )

    intensity_norm = (
        intensity
        /
        reference_peak
    )


    cd_list = []


    # -----------------------------------------
    # Change dose around nominal dose
    # -----------------------------------------

    for factor in dose_factors:

        dose = (
            nominal_dose
            *
            factor
        )


        effective_intensity = (
            dose
            *
            intensity_norm
        )


        resist = soft_resist(
            effective_intensity,
            threshold,
            beta
        )


        resist_profile = resist[
            center_y_index,
            :
        ]


        cd, _, _ = measure_cd(
            x,
            resist_profile,
            threshold=0.5
        )


        cd_nm = (
            cd * 1000
        )

        cd_list.append(
            cd_nm
        )


        print(
            f"Shift = {source_shift:.2f}, "
            f"Dose factor = {factor:.3f}, "
            f"Dose = {dose:.4f}, "
            f"CD = {cd_nm:.3f} nm"
        )


    # -----------------------------------------
    # Plot
    # -----------------------------------------

    plt.plot(
        (dose_factors - 1) * 100,
        cd_list,
        marker="o",
        label=f"Shift = {source_shift:.2f}"
    )


# Target CD
plt.axhline(
    target_cd * 1000,
    linestyle="--",
    label="Target CD"
)

plt.xlabel(
    "Dose Variation (%)"
)

plt.ylabel(
    "Printed CD (nm)"
)

plt.title(
    "Dose Sensitivity of Different Illumination"
)

plt.grid()
plt.legend()

plt.show()
# =========================================================
# Exposure Latitude test
# =========================================================

cd_min_nm = 95.0
cd_max_nm = 105.0


candidate_sources = [
    {
        "shift": 0.60,
        "nominal_dose": 0.390
    },
    {
        "shift": 0.70,
        "nominal_dose": 0.385
    }
]


for candidate in candidate_sources:

    source_shift = candidate["shift"]
    nominal_dose = candidate["nominal_dose"]


    # -----------------------------------------------------
    # Dipole illumination
    # -----------------------------------------------------

    source_positions = [
        (+source_shift, 0.0),
        (-source_shift, 0.0)
    ]

    source_weights = [
        0.5,
        0.5
    ]


    # -----------------------------------------------------
    # Aerial image
    # -----------------------------------------------------

    intensity = partial_coherent_image_2d(
        mask,
        source_positions,
        source_weights
    )

    intensity_norm = (
        intensity
        /
        reference_peak
    )


    # -----------------------------------------------------
    # Fine dose scan
    # -----------------------------------------------------

    dose_factors_fine = np.linspace(
        0.85,
        1.15,
        301
    )

    valid_doses = []


    for factor in dose_factors_fine:

        dose = (
            nominal_dose
            *
            factor
        )

        effective_intensity = (
            dose
            *
            intensity_norm
        )

        resist = soft_resist(
            effective_intensity,
            threshold,
            beta
        )

        profile = resist[
            center_y_index,
            :
        ]

        cd, _, _ = measure_cd(
            x,
            profile,
            threshold=0.5
        )

        if np.isnan(cd):
            continue

        cd_nm = cd * 1000


        # CD specification
        if (
            cd_min_nm
            <= cd_nm
            <= cd_max_nm
        ):

            valid_doses.append(
                dose
            )


    # -----------------------------------------------------
    # Exposure latitude
    # -----------------------------------------------------

    if len(valid_doses) > 0:

        dose_min = min(
            valid_doses
        )

        dose_max = max(
            valid_doses
        )


        exposure_latitude = (
            (dose_max - dose_min)
            /
            nominal_dose
            *
            100
        )


        print()
        print(
            f"Source shift = {source_shift:.2f}"
        )

        print(
            f"Dose range = "
            f"{dose_min:.4f} ~ {dose_max:.4f}"
        )

        print(
            f"Exposure Latitude = "
            f"{exposure_latitude:.2f}%"
        )

    else:

        print(
            f"Source shift = {source_shift:.2f}: "
            f"No valid dose range"
        )