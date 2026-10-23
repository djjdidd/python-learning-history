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

    return (
        np.sqrt(
            (FX - source_x) ** 2
            +
            (FY - source_y) ** 2
        )
        <= cutoff
    ).astype(float)


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
            spectrum * pupil
        )

        intensity += (
            weight
            *
            np.abs(field) ** 2
        )

    return intensity


# =========================================================
# 6. Normal illumination
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


# =========================================================
# 7. Horizontal dipole illumination
# =========================================================

source_shift = 0.6

dipole_positions = [
    (+source_shift, 0.0),
    (-source_shift, 0.0)
]

dipole_weights = [
    0.5,
    0.5
]

intensity_dipole = partial_coherent_image_2d(
    mask,
    dipole_positions,
    dipole_weights
)


# =========================================================
# 8. Same normalization reference
# =========================================================

reference_peak = np.max(
    intensity_normal
)

normal_norm = (
    intensity_normal
    /
    reference_peak
)

dipole_norm = (
    intensity_dipole
    /
    reference_peak
)


# =========================================================
# 9. Take center horizontal profile
# =========================================================

center_y_index = np.argmin(
    np.abs(y)
)

profile_normal = normal_norm[
    center_y_index,
    :
]

profile_dipole = dipole_norm[
    center_y_index,
    :
]


# =========================================================
# 10. Contrast
# =========================================================

def contrast(profile):

    Imax = np.max(profile)
    Imin = np.min(profile)

    return (
        (Imax - Imin)
        /
        (Imax + Imin)
    )


contrast_normal = contrast(
    profile_normal
)

contrast_dipole = contrast(
    profile_dipole
)


print()
print("=" * 50)
print("2D Periodic Pattern")
print("=" * 50)

print(
    "Cutoff frequency =",
    cutoff,
    "1/um"
)

print(
    "First diffraction order =",
    1 / period,
    "1/um"
)

print(
    "Normal contrast =",
    contrast_normal
)

print(
    "Dipole contrast =",
    contrast_dipole
)


# =========================================================
# 11. Plot center profiles
# =========================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    x,
    profile_normal,
    label="Normal"
)

plt.plot(
    x,
    profile_dipole,
    label="Horizontal Dipole"
)

plt.xlim(
    -0.6,
    0.6
)

plt.xlabel(
    "x (um)"
)

plt.ylabel(
    "Normalized Intensity"
)

plt.title(
    "2D Periodic Pattern - Center Profile"
)

plt.grid()
plt.legend()

plt.show()
# =========================================================
# 12. Scan dipole source shift
# =========================================================

source_shift_list = np.linspace(
    0.0,
    2.0,
    41
)

contrast_list = []


for source_shift in source_shift_list:

    dipole_positions = [
        (+source_shift, 0.0),
        (-source_shift, 0.0)
    ]

    dipole_weights = [
        0.5,
        0.5
    ]


    intensity = partial_coherent_image_2d(
        mask,
        dipole_positions,
        dipole_weights
    )


    # Same reference
    intensity_norm = (
        intensity
        /
        reference_peak
    )


    profile = intensity_norm[
        center_y_index,
        :
    ]


    current_contrast = contrast(
        profile
    )


    contrast_list.append(
        current_contrast
    )


# =========================================================
# 13. Find best source shift
# =========================================================

best_index = np.argmax(
    contrast_list
)

best_source_shift = source_shift_list[
    best_index
]

best_contrast = contrast_list[
    best_index
]


print()
print("=" * 50)
print("Dipole Source Optimization")
print("=" * 50)

print(
    "Best source shift =",
    best_source_shift,
    "1/um"
)

print(
    "Best contrast =",
    best_contrast
)


# =========================================================
# 14. Plot
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    source_shift_list,
    contrast_list,
    marker="o"
)

plt.axvline(
    best_source_shift,
    linestyle="--"
)

plt.xlabel(
    "Source Shift (1/um)"
)

plt.ylabel(
    "Contrast"
)

plt.title(
    "Contrast vs Dipole Source Shift"
)

plt.grid()

plt.show()
# =========================================================
# 15. Soft resist
# =========================================================

def soft_resist(
    intensity,
    threshold=0.5,
    beta=20.0
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


threshold = 0.5
beta = 20.0


# =========================================================
# 16. Subpixel edge interpolation
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

    return (
        x1
        +
        (threshold - y1)
        /
        (y2 - y1)
        *
        (x2 - x1)
    )


# =========================================================
# 17. Measure center-line CD
# =========================================================

def measure_cd(
    coordinate,
    resist_profile,
    threshold=0.5
):

    binary = (
        resist_profile >= threshold
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
# 18. NILS from aerial image
# =========================================================

def measure_nils(
    coordinate,
    intensity_profile,
    intensity_threshold=0.5
):

    binary = (
        intensity_profile
        >= intensity_threshold
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
        return np.nan

    left_index = left_candidates[-1]
    right_index = right_candidates[0]


    left_position = interpolate_edge(
        coordinate[left_index],
        coordinate[left_index + 1],
        intensity_profile[left_index],
        intensity_profile[left_index + 1],
        intensity_threshold
    )

    right_position = interpolate_edge(
        coordinate[right_index],
        coordinate[right_index + 1],
        intensity_profile[right_index],
        intensity_profile[right_index + 1],
        intensity_threshold
    )


    cd = (
        right_position
        -
        left_position
    )


    gradient = np.gradient(
        intensity_profile,
        coordinate
    )


    left_gradient = np.interp(
        left_position,
        coordinate,
        gradient
    )

    right_gradient = np.interp(
        right_position,
        coordinate,
        gradient
    )


    left_nils = (
        cd
        /
        intensity_threshold
        *
        np.abs(left_gradient)
    )

    right_nils = (
        cd
        /
        intensity_threshold
        *
        np.abs(right_gradient)
    )


    return 0.5 * (
        left_nils
        +
        right_nils
    )
# =========================================================
# 19. Scan source shift using multiple metrics
# =========================================================

source_shift_list = np.linspace(
    0.0,
    2.0,
    41
)

contrast_list = []
cd_error_list = []
nils_list = []


target_cd = line_width


for source_shift in source_shift_list:

    dipole_positions = [
        (+source_shift, 0.0),
        (-source_shift, 0.0)
    ]

    dipole_weights = [
        0.5,
        0.5
    ]


    # -----------------------------------------
    # Aerial image
    # -----------------------------------------

    intensity = partial_coherent_image_2d(
        mask,
        dipole_positions,
        dipole_weights
    )


    intensity_norm = (
        intensity
        /
        reference_peak
    )


    # -----------------------------------------
    # Center horizontal profile
    # -----------------------------------------

    profile = intensity_norm[
        center_y_index,
        :
    ]


    # -----------------------------------------
    # Contrast
    # -----------------------------------------

    current_contrast = contrast(
        profile
    )


    # -----------------------------------------
    # Resist
    # -----------------------------------------

    resist = soft_resist(
        intensity_norm,
        threshold,
        beta
    )

    resist_profile = resist[
        center_y_index,
        :
    ]


    # -----------------------------------------
    # CD
    # -----------------------------------------

    current_cd, _, _ = measure_cd(
        x,
        resist_profile,
        threshold=0.5
    )


    cd_error_nm = (
        current_cd
        -
        target_cd
    ) * 1000


    # -----------------------------------------
    # NILS
    # -----------------------------------------

    current_nils = measure_nils(
        x,
        profile,
        intensity_threshold=0.5
    )


    contrast_list.append(
        current_contrast
    )

    cd_error_list.append(
        cd_error_nm
    )

    nils_list.append(
        current_nils
    )
# =========================================================
# 20. Find best source shift for each metric
# =========================================================

best_contrast_index = np.nanargmax(
    contrast_list
)

best_cd_index = np.nanargmin(
    np.abs(cd_error_list)
)

best_nils_index = np.nanargmax(
    nils_list
)


print()
print("=" * 60)
print("2D Dipole Source Optimization")
print("=" * 60)

print(
    "Best shift for contrast =",
    source_shift_list[
        best_contrast_index
    ],
    "1/um"
)

print(
    "Best contrast =",
    contrast_list[
        best_contrast_index
    ]
)

print()

print(
    "Best shift for CD accuracy =",
    source_shift_list[
        best_cd_index
    ],
    "1/um"
)

print(
    "Minimum |CD error| =",
    abs(
        cd_error_list[
            best_cd_index
        ]
    ),
    "nm"
)

print()

print(
    "Best shift for NILS =",
    source_shift_list[
        best_nils_index
    ],
    "1/um"
)

print(
    "Best NILS =",
    nils_list[
        best_nils_index
    ]
)
# =========================================================
# 21. Normalize metrics
# =========================================================

contrast_array = np.array(
    contrast_list
)

cd_error_array = np.abs(
    np.array(cd_error_list)
)

nils_array = np.array(
    nils_list
)


contrast_score = (
    contrast_array
    /
    np.nanmax(contrast_array)
)

nils_score = (
    nils_array
    /
    np.nanmax(nils_array)
)


# CD error 越小越好
# 所以转换成“分数”，越大越好
cd_score = (
    1
    -
    cd_error_array
    /
    np.nanmax(cd_error_array)
)
# =========================================================
# 22. Combined score
# =========================================================

weight_contrast = 1 / 3
weight_cd = 1 / 3
weight_nils = 1 / 3


combined_score = (
    weight_contrast * contrast_score
    +
    weight_cd * cd_score
    +
    weight_nils * nils_score
)
# =========================================================
# 23. Find best combined source shift
# =========================================================

best_combined_index = np.nanargmax(
    combined_score
)

best_combined_shift = source_shift_list[
    best_combined_index
]


print()
print("=" * 60)
print("Combined Source Optimization")
print("=" * 60)

print(
    "Best combined source shift =",
    best_combined_shift,
    "1/um"
)

print(
    "Contrast =",
    contrast_array[
        best_combined_index
    ]
)

print(
    "CD error =",
    cd_error_array[
        best_combined_index
    ],
    "nm"
)

print(
    "NILS =",
    nils_array[
        best_combined_index
    ]
)

print(
    "Combined score =",
    combined_score[
        best_combined_index
    ]
)
plt.figure(figsize=(8, 5))

plt.plot(
    source_shift_list,
    contrast_score,
    label="Contrast Score"
)

plt.plot(
    source_shift_list,
    cd_score,
    label="CD Score"
)

plt.plot(
    source_shift_list,
    nils_score,
    label="NILS Score"
)

plt.plot(
    source_shift_list,
    combined_score,
    linewidth=2,
    label="Combined Score"
)

plt.xlabel(
    "Source Shift (1/um)"
)

plt.ylabel(
    "Normalized Score"
)

plt.title(
    "Multi-Metric Source Optimization"
)

plt.grid()
plt.legend()

plt.show()
# =========================================================
# CD-constrained source optimization
# =========================================================

cd_tolerance_nm = 10.0

best_index = None
best_score = -np.inf


for i in range(len(source_shift_list)):

    cd_error = abs(
        cd_error_list[i]
    )

    # First satisfy CD specification
    if cd_error <= cd_tolerance_nm:

        # Among feasible solutions:
        # maximize contrast + NILS
        score = (
            0.5 * contrast_score[i]
            +
            0.5 * nils_score[i]
        )

        if score > best_score:

            best_score = score
            best_index = i


print()
print("=" * 60)
print("CD-Constrained Source Optimization")
print("=" * 60)

if best_index is not None:

    print(
        "Best feasible source shift =",
        source_shift_list[best_index],
        "1/um"
    )

    print(
        "CD error =",
        abs(cd_error_list[best_index]),
        "nm"
    )

    print(
        "Contrast =",
        contrast_list[best_index]
    )

    print(
        "NILS =",
        nils_list[best_index]
    )

else:

    print(
        "No source shift satisfies the CD tolerance."
    )