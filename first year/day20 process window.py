import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Basic parameters
# =========================================================

N = 256
L = 4.0                  # um
dx = L / N

x = np.arange(N) * dx - L / 2
y = np.arange(N) * dx - L / 2

X, Y = np.meshgrid(x, y)

wavelength = 0.193       # um
NA = 0.85

cutoff = NA / wavelength


print("dx =", dx, "um")
print("cutoff =", cutoff, "1/um")


# =========================================================
# 2. Periodic line-space mask
# =========================================================

period = 0.20            # um
line_width = 0.10        # um

target_cd = line_width


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


# =========================================================
# 3. Frequency grid
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
# 4. Defocused shifted pupil
# =========================================================

def make_defocused_pupil_2d(
    source_x,
    source_y,
    defocus
):

    # Frequency coordinates relative to source point
    FX_shift = FX - source_x
    FY_shift = FY - source_y


    r2 = (
        FX_shift ** 2
        +
        FY_shift ** 2
    )


    # Circular pupil aperture
    aperture = (
        np.sqrt(r2)
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
        defocus
        *
        r2
    )


    pupil = (
        aperture
        *
        phase
    )


    return pupil


# =========================================================
# 5. Partial-coherent imaging
# =========================================================

def partial_coherent_image_2d_defocus(
    mask_input,
    source_positions,
    source_weights,
    defocus
):

    # Mask spectrum
    spectrum = np.fft.fft2(
        mask_input
    )


    intensity = np.zeros(
        (N, N),
        dtype=float
    )


    # Abbe source-point summation
    for (source_x, source_y), weight in zip(
        source_positions,
        source_weights
    ):

        pupil = make_defocused_pupil_2d(
            source_x,
            source_y,
            defocus
        )


        field = np.fft.ifft2(
            spectrum
            *
            pupil
        )


        # Different source points add intensity
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
# IMPORTANT:
# Use one fixed reference.
# Do NOT normalize each focus/dose condition independently.

normal_positions = [
    (0.0, 0.0)
]

normal_weights = [
    1.0
]


reference_intensity = partial_coherent_image_2d_defocus(
    mask,
    normal_positions,
    normal_weights,
    defocus=0.0
)


reference_peak = np.max(
    reference_intensity
)


print(
    "Reference peak =",
    reference_peak
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

        return (
            x1 + x2
        ) / 2


    edge_position = (
        x1
        +
        (
            threshold - y1
        )
        /
        (
            y2 - y1
        )
        *
        (
            x2 - x1
        )
    )


    return edge_position


# =========================================================
# 9. Measure CD
# =========================================================

def measure_cd(
    coordinate,
    profile,
    threshold=0.5
):

    binary = (
        profile
        >= threshold
    ).astype(float)


    transition = np.diff(
        binary
    )


    # 0 -> 1
    left_edges = np.where(
        transition == 1
    )[0]


    # 1 -> 0
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

        return (
            np.nan,
            np.nan,
            np.nan
        )


    left_index = left_candidates[-1]

    right_index = right_candidates[0]


    left_position = interpolate_edge(
        coordinate[left_index],
        coordinate[left_index + 1],
        profile[left_index],
        profile[left_index + 1],
        threshold
    )


    right_position = interpolate_edge(
        coordinate[right_index],
        coordinate[right_index + 1],
        profile[right_index],
        profile[right_index + 1],
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
# 11. Illumination
# =========================================================

source_shift = 0.60


source_positions = [
    (+source_shift, 0.0),
    (-source_shift, 0.0)
]


source_weights = [
    0.5,
    0.5
]


# Nominal dose found previously
nominal_dose = 0.390


# =========================================================
# 12. Nominal condition check
# =========================================================

intensity_nominal = partial_coherent_image_2d_defocus(
    mask,
    source_positions,
    source_weights,
    defocus=0.0
)


intensity_nominal_norm = (
    intensity_nominal
    /
    reference_peak
)


effective_intensity_nominal = (
    nominal_dose
    *
    intensity_nominal_norm
)


resist_nominal = soft_resist(
    effective_intensity_nominal,
    threshold,
    beta
)


profile_nominal = resist_nominal[
    center_y_index,
    :
]


cd_nominal, _, _ = measure_cd(
    x,
    profile_nominal,
    threshold=0.5
)


print()
print("=" * 60)
print("Nominal Condition")
print("=" * 60)

print(
    "Source shift =",
    source_shift,
    "1/um"
)

print(
    "Nominal dose =",
    nominal_dose
)

print(
    "Nominal CD =",
    cd_nominal * 1000,
    "nm"
)


# =========================================================
# 13. CD specification
# =========================================================

cd_min_nm = 98.0
cd_max_nm = 102.0


print()
print(
    "CD specification =",
    cd_min_nm,
    "~",
    cd_max_nm,
    "nm"
)


# =========================================================
# 14. Process Window scan range
# =========================================================

# Focus:
# -0.10 um ~ +0.10 um
# = -100 nm ~ +100 nm

focus_pw = np.linspace(
    -0.20,
    0.20,
    181
)


# Dose:
# nominal dose +/- 10 %

dose_factor_pw = np.linspace(
    0.90,
    1.10,
    81
)


dose_pw = (
    nominal_dose
    *
    dose_factor_pw
)


# =========================================================
# 15. CD matrix
# =========================================================

cd_matrix = np.full(
    (
        len(dose_pw),
        len(focus_pw)
    ),
    np.nan
)


# =========================================================
# 16. Dose-focus scan
# =========================================================

for j, defocus in enumerate(
    focus_pw
):

    # -----------------------------------------------------
    # Optical image depends on focus
    # -----------------------------------------------------

    intensity = partial_coherent_image_2d_defocus(
        mask,
        source_positions,
        source_weights,
        defocus
    )


    intensity_norm = (
        intensity
        /
        reference_peak
    )


    # -----------------------------------------------------
    # Dose scan
    # -----------------------------------------------------

    for i, dose in enumerate(
        dose_pw
    ):

        # Dose scales intensity
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
        profile = resist[
            center_y_index,
            :
        ]


        # CD
        cd, _, _ = measure_cd(
            x,
            profile,
            threshold=0.5
        )


        if np.isfinite(cd):

            cd_matrix[i, j] = (
                cd
                *
                1000
            )


# =========================================================
# 17. Valid Process Window
# =========================================================

valid_window = (
    np.isfinite(cd_matrix)
    &
    (
        cd_matrix
        >= cd_min_nm
    )
    &
    (
        cd_matrix
        <= cd_max_nm
    )
)


# =========================================================
# 18. Plot Process Window
# =========================================================

plt.figure(
    figsize=(8, 6)
)


plt.contourf(
    focus_pw * 1000,
    (
        dose_factor_pw - 1
    ) * 100,
    valid_window.astype(float),
    levels=[
        0.5,
        1.5
    ]
)


# Best focus
plt.axvline(
    0,
    linestyle="--",
    label="Best focus"
)


# Nominal dose
plt.axhline(
    0,
    linestyle="--",
    label="Nominal dose"
)


plt.xlabel(
    "Defocus (nm)"
)


plt.ylabel(
    "Dose Variation (%)"
)


plt.title(
    "Dose-Focus Process Window"
)


plt.grid()

plt.legend()

plt.show()


# =========================================================
# 19. CD contour map
# =========================================================

plt.figure(
    figsize=(8, 6)
)


contour = plt.contour(
    focus_pw * 1000,
    (
        dose_factor_pw - 1
    ) * 100,
    cd_matrix,
    levels=[
        98,
        100,
        102
    ]
)


plt.clabel(
    contour,
    inline=True,
    fontsize=9
)


plt.axvline(
    0,
    linestyle="--"
)


plt.axhline(
    0,
    linestyle="--"
)


plt.xlabel(
    "Defocus (nm)"
)


plt.ylabel(
    "Dose Variation (%)"
)


plt.title(
    "CD Contours in Dose-Focus Space"
)


plt.grid()

plt.show()


# =========================================================
# 20. DOF at nominal dose
# =========================================================

nominal_dose_index = np.argmin(
    np.abs(
        dose_factor_pw - 1.0
    )
)


cd_at_nominal_dose = cd_matrix[
    nominal_dose_index,
    :
]


valid_focus = (
    np.isfinite(
        cd_at_nominal_dose
    )
    &
    (
        cd_at_nominal_dose
        >= cd_min_nm
    )
    &
    (
        cd_at_nominal_dose
        <= cd_max_nm
    )
)


focus_center_index = np.argmin(
    np.abs(focus_pw)
)


left_index = focus_center_index

while (
    left_index > 0
    and
    valid_focus[
        left_index - 1
    ]
):

    left_index -= 1


right_index = focus_center_index

while (
    right_index
    <
    len(focus_pw) - 1
    and
    valid_focus[
        right_index + 1
    ]
):

    right_index += 1


focus_min = focus_pw[
    left_index
]


focus_max = focus_pw[
    right_index
]


DOF = (
    focus_max
    -
    focus_min
)


print()
print("=" * 60)
print("DOF at Nominal Dose")
print("=" * 60)

print(
    "Focus range =",
    focus_min * 1000,
    "to",
    focus_max * 1000,
    "nm"
)

print(
    "DOF =",
    DOF * 1000,
    "nm"
)


# =========================================================
# 21. Exposure Latitude at best focus
# =========================================================

best_focus_index = np.argmin(
    np.abs(focus_pw)
)


cd_at_best_focus = cd_matrix[
    :,
    best_focus_index
]


valid_dose = (
    np.isfinite(
        cd_at_best_focus
    )
    &
    (
        cd_at_best_focus
        >= cd_min_nm
    )
    &
    (
        cd_at_best_focus
        <= cd_max_nm
    )
)


valid_dose_indices = np.where(
    valid_dose
)[0]


if len(valid_dose_indices) > 0:

    dose_min = dose_pw[
        valid_dose_indices[0]
    ]

    dose_max = dose_pw[
        valid_dose_indices[-1]
    ]


    exposure_latitude = (
        (
            dose_max
            -
            dose_min
        )
        /
        nominal_dose
        *
        100
    )


    print()
    print("=" * 60)
    print("Exposure Latitude at Best Focus")
    print("=" * 60)

    print(
        "Dose range =",
        dose_min,
        "to",
        dose_max
    )

    print(
        "Exposure Latitude =",
        exposure_latitude,
        "%"
    )

# =========================================================
# Better operating dose at best focus
# =========================================================

dose_center = (
    dose_min
    +
    dose_max
) / 2


dose_shift_percent = (
    (dose_center - nominal_dose)
    /
    nominal_dose
    *
    100
)


print()
print("=" * 60)
print("Centered Operating Dose")
print("=" * 60)

print(
    "Old nominal dose =",
    nominal_dose
)

print(
    "Centered dose =",
    dose_center
)

print(
    "Dose change from old nominal =",
    dose_shift_percent,
    "%"
)
# =========================================================
# CD at centered operating point
# =========================================================

effective_intensity_center = (
    dose_center
    *
    intensity_nominal_norm
)


resist_center = soft_resist(
    effective_intensity_center,
    threshold,
    beta
)


profile_center = resist_center[
    center_y_index,
    :
]


cd_center, _, _ = measure_cd(
    x,
    profile_center,
    threshold=0.5
)


print(
    "CD at centered dose =",
    cd_center * 1000,
    "nm"
)


# =========================================================
# DOF at centered dose
# =========================================================

centered_dose = dose_center


cd_focus_centered = []


for defocus in focus_pw:

    intensity = partial_coherent_image_2d_defocus(
        mask,
        source_positions,
        source_weights,
        defocus
    )

    intensity_norm = (
        intensity
        /
        reference_peak
    )

    effective_intensity = (
        centered_dose
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

    cd_focus_centered.append(
        cd * 1000
    )


cd_focus_centered = np.array(
    cd_focus_centered
)


# CD specification
valid_centered = (
    np.isfinite(cd_focus_centered)
    &
    (cd_focus_centered >= cd_min_nm)
    &
    (cd_focus_centered <= cd_max_nm)
)


center_index_focus = np.argmin(
    np.abs(focus_pw)
)


left_index = center_index_focus

while (
    left_index > 0
    and
    valid_centered[left_index - 1]
):
    left_index -= 1


right_index = center_index_focus

while (
    right_index < len(focus_pw) - 1
    and
    valid_centered[right_index + 1]
):
    right_index += 1


focus_min_centered = focus_pw[
    left_index
]

focus_max_centered = focus_pw[
    right_index
]


DOF_centered = (
    focus_max_centered
    -
    focus_min_centered
)


print()
print("=" * 60)
print("DOF at Centered Dose")
print("=" * 60)

print(
    "Centered dose =",
    centered_dose
)

print(
    "Focus range =",
    focus_min_centered * 1000,
    "to",
    focus_max_centered * 1000,
    "nm"
)

print(
    "DOF =",
    DOF_centered * 1000,
    "nm"
)