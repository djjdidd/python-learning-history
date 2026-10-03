import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Basic parameters
# =========================================================

N = 128
L = 4.0
dx = L / N

x = np.arange(N) * dx - L / 2
y = np.arange(N) * dx - L / 2

X, Y = np.meshgrid(x, y)

wavelength = 0.193
NA = 0.85
cutoff = NA / wavelength


# =========================================================
# 2. 2D rectangular mask
# =========================================================

width_x = 0.6
width_y = 0.4

mask = (
    (np.abs(X) <= width_x / 2)
    &
    (np.abs(Y) <= width_y / 2)
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
# 5. 2D partial-coherent imaging
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
# 6. Soft resist
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
# 7. Reference intensity
#    Use normal illumination as fixed reference
# =========================================================

normal_source_positions = [
    (0.0, 0.0)
]

normal_source_weights = [
    1.0
]

intensity_normal = partial_coherent_image_2d(
    mask,
    normal_source_positions,
    normal_source_weights
)

reference_peak = np.max(
    intensity_normal
)


# =========================================================
# 8. Subpixel edge interpolation
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
# 9. Find printed edge nearest target edge
# =========================================================

def find_edge_near_target(
    coordinate,
    resist_profile,
    target_position,
    threshold=0.5
):

    binary = (
        resist_profile >= threshold
    ).astype(float)

    transition = np.diff(
        binary
    )

    edge_indices = np.where(
        transition != 0
    )[0]

    if len(edge_indices) == 0:
        return np.nan

    edge_positions = []

    for index in edge_indices:

        edge_position = interpolate_edge(
            coordinate[index],
            coordinate[index + 1],
            resist_profile[index],
            resist_profile[index + 1],
            threshold
        )

        edge_positions.append(
            edge_position
        )

    edge_positions = np.array(
        edge_positions
    )

    nearest_index = np.argmin(
        np.abs(
            edge_positions
            -
            target_position
        )
    )

    return edge_positions[
        nearest_index
    ]


# =========================================================
# 10. EPE sampling positions
# =========================================================

num_samples = 9

sample_y = np.linspace(
    -0.15,
    0.15,
    num_samples
)

sample_x = np.linspace(
    -0.25,
    0.25,
    num_samples
)

target_left = -width_x / 2
target_right = +width_x / 2

target_bottom = -width_y / 2
target_top = +width_y / 2


# =========================================================
# 11. Measure vertical-edge EPE
# =========================================================

def measure_vertical_epe(
    resist
):

    epe_left = []
    epe_right = []

    for y_value in sample_y:

        row_index = np.argmin(
            np.abs(
                y - y_value
            )
        )

        profile = resist[
            row_index,
            :
        ]

        printed_left = find_edge_near_target(
            x,
            profile,
            target_left
        )

        printed_right = find_edge_near_target(
            x,
            profile,
            target_right
        )

        epe_left.append(
            printed_left
            -
            target_left
        )

        epe_right.append(
            printed_right
            -
            target_right
        )

    return (
        np.array(epe_left),
        np.array(epe_right)
    )


# =========================================================
# 12. Measure horizontal-edge EPE
# =========================================================

def measure_horizontal_epe(
    resist
):

    epe_bottom = []
    epe_top = []

    for x_value in sample_x:

        column_index = np.argmin(
            np.abs(
                x - x_value
            )
        )

        profile = resist[
            :,
            column_index
        ]

        printed_bottom = find_edge_near_target(
            y,
            profile,
            target_bottom
        )

        printed_top = find_edge_near_target(
            y,
            profile,
            target_top
        )

        epe_bottom.append(
            printed_bottom
            -
            target_bottom
        )

        epe_top.append(
            printed_top
            -
            target_top
        )

    return (
        np.array(epe_bottom),
        np.array(epe_top)
    )


# =========================================================
# 13. Scan source shift
# =========================================================

source_shift_list = np.linspace(
    0.0,
    2.0,
    21
)

mean_epe_list = []
max_epe_list = []


for source_shift in source_shift_list:

    source_positions = [
        (+source_shift, 0.0),
        (-source_shift, 0.0),
        (0.0, +source_shift),
        (0.0, -source_shift)
    ]

    source_weights = np.ones(
        len(source_positions)
    )

    source_weights = (
        source_weights
        /
        np.sum(source_weights)
    )


    # Aerial image
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


    # Resist
    resist = soft_resist(
        intensity_norm,
        threshold,
        beta
    )


    # EPE
    epe_left, epe_right = (
        measure_vertical_epe(
            resist
        )
    )

    epe_bottom, epe_top = (
        measure_horizontal_epe(
            resist
        )
    )


    all_epe = np.concatenate([
        epe_left,
        epe_right,
        epe_bottom,
        epe_top
    ])


    abs_epe_nm = (
        np.abs(all_epe)
        *
        1000
    )


    mean_epe = np.nanmean(
        abs_epe_nm
    )

    max_epe = np.nanmax(
        abs_epe_nm
    )


    mean_epe_list.append(
        mean_epe
    )

    max_epe_list.append(
        max_epe
    )


    print(
        f"Shift = {source_shift:.2f}  "
        f"Mean |EPE| = {mean_epe:.3f} nm  "
        f"Max |EPE| = {max_epe:.3f} nm"
    )


# =========================================================
# 14. Find best shifts
# =========================================================

best_mean_index = np.argmin(
    mean_epe_list
)

best_max_index = np.argmin(
    max_epe_list
)


best_shift_mean = source_shift_list[
    best_mean_index
]

best_shift_max = source_shift_list[
    best_max_index
]


print()
print("=" * 60)
print("Best Source Shift")
print("=" * 60)

print(
    "Best shift for mean |EPE| =",
    best_shift_mean
)

print(
    "Minimum mean |EPE| =",
    mean_epe_list[
        best_mean_index
    ],
    "nm"
)

print()

print(
    "Best shift for max |EPE| =",
    best_shift_max
)

print(
    "Minimum max |EPE| =",
    max_epe_list[
        best_max_index
    ],
    "nm"
)


# =========================================================
# 15. Plot EPE vs source shift
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    source_shift_list,
    mean_epe_list,
    marker="o",
    label="Mean |EPE|"
)

plt.plot(
    source_shift_list,
    max_epe_list,
    marker="o",
    label="Max |EPE|"
)

plt.xlabel(
    "Source Shift (1/um)"
)

plt.ylabel(
    "EPE (nm)"
)

plt.title(
    "2D EPE vs Source Shift"
)

plt.grid()
plt.legend()

plt.show()