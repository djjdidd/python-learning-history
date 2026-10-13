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

X, Y = np.meshgrid(
    x,
    y
)

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
# 4. Shifted 2D pupil
# =========================================================

def make_shifted_pupil_2d(
    FX,
    FY,
    cutoff,
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
    mask,
    source_positions,
    source_weights
):

    spectrum = np.fft.fft2(
        mask
    )

    intensity = np.zeros_like(
        mask,
        dtype=float
    )

    for (source_x, source_y), weight in zip(
        source_positions,
        source_weights
    ):

        pupil = make_shifted_pupil_2d(
            FX,
            FY,
            cutoff,
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


# =========================================================
# 7. 4-point off-axis illumination
# =========================================================

source_shift = 0.3

offaxis_source_positions = [
    (+source_shift, 0.0),
    (-source_shift, 0.0),
    (0.0, +source_shift),
    (0.0, -source_shift)
]

offaxis_source_weights = np.ones(
    len(offaxis_source_positions)
)

offaxis_source_weights = (
    offaxis_source_weights
    /
    np.sum(offaxis_source_weights)
)

intensity_offaxis = partial_coherent_image_2d(
    mask,
    offaxis_source_positions,
    offaxis_source_weights
)


# =========================================================
# 8. Use SAME normalization reference
# =========================================================

reference_peak = np.max(
    intensity_normal
)

normal_norm = (
    intensity_normal
    /
    reference_peak
)

offaxis_norm = (
    intensity_offaxis
    /
    reference_peak
)


# =========================================================
# 9. Soft resist
# =========================================================

def soft_resist_2d(
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

resist_normal = soft_resist_2d(
    normal_norm,
    threshold,
    beta
)

resist_offaxis = soft_resist_2d(
    offaxis_norm,
    threshold,
    beta
)


# =========================================================
# 10. Printed binary pattern
# =========================================================

printed_normal = (
    resist_normal >= 0.5
).astype(float)

printed_offaxis = (
    resist_offaxis >= 0.5
).astype(float)


# =========================================================
# 11. Print some simple metrics
# =========================================================

print()
print("=" * 50)
print("2D Partial-Coherent Lithography")
print("=" * 50)

print(
    "Normal peak intensity =",
    np.max(intensity_normal)
)

print(
    "Off-axis peak / normal peak =",
    np.max(offaxis_norm)
)

print(
    "Normal printed area =",
    np.sum(printed_normal) * dx * dx,
    "um^2"
)

print(
    "Off-axis printed area =",
    np.sum(printed_offaxis) * dx * dx,
    "um^2"
)


# =========================================================
# 12. Plot aerial images
# =========================================================

zoom = 0.6


plt.figure(figsize=(6, 5))

plt.imshow(
    normal_norm,
    extent=[
        x.min(),
        x.max(),
        y.min(),
        y.max()
    ],
    origin="lower",
    vmin=0,
    vmax=1
)

plt.xlim(-zoom, zoom)
plt.ylim(-zoom, zoom)

plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.title("Normal Illumination - Aerial Image")

plt.colorbar()

plt.show()


plt.figure(figsize=(6, 5))

plt.imshow(
    offaxis_norm,
    extent=[
        x.min(),
        x.max(),
        y.min(),
        y.max()
    ],
    origin="lower",
    vmin=0,
    vmax=1
)

plt.xlim(-zoom, zoom)
plt.ylim(-zoom, zoom)

plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.title("4-Point Off-Axis - Aerial Image")

plt.colorbar()

plt.show()


# =========================================================
# 13. Plot printed patterns
# =========================================================

plt.figure(figsize=(6, 5))

plt.imshow(
    printed_normal,
    extent=[
        x.min(),
        x.max(),
        y.min(),
        y.max()
    ],
    origin="lower",
    vmin=0,
    vmax=1
)

plt.xlim(-zoom, zoom)
plt.ylim(-zoom, zoom)

plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.title("Normal Illumination - Printed Pattern")

plt.show()


plt.figure(figsize=(6, 5))

plt.imshow(
    printed_offaxis,
    extent=[
        x.min(),
        x.max(),
        y.min(),
        y.max()
    ],
    origin="lower",
    vmin=0,
    vmax=1
)

plt.xlim(-zoom, zoom)
plt.ylim(-zoom, zoom)

plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.title("4-Point Off-Axis - Printed Pattern")

plt.show()
# =========================================================
# 14. 1D subpixel edge interpolation
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
# 15. Measure CD from one 1D resist profile
# =========================================================

def measure_cd_1d(
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

    return cd, left_position, right_position


# =========================================================
# 16. Extract center row and center column
# =========================================================

center_x_index = np.argmin(
    np.abs(x)
)

center_y_index = np.argmin(
    np.abs(y)
)


# y = 0 horizontal profile
normal_profile_x = resist_normal[
    center_y_index,
    :
]

offaxis_profile_x = resist_offaxis[
    center_y_index,
    :
]


# x = 0 vertical profile
normal_profile_y = resist_normal[
    :,
    center_x_index
]

offaxis_profile_y = resist_offaxis[
    :,
    center_x_index
]


# =========================================================
# 17. Measure CDx
# =========================================================

cd_x_normal, _, _ = measure_cd_1d(
    x,
    normal_profile_x
)

cd_x_offaxis, _, _ = measure_cd_1d(
    x,
    offaxis_profile_x
)


# =========================================================
# 18. Measure CDy
# =========================================================

cd_y_normal, _, _ = measure_cd_1d(
    y,
    normal_profile_y
)

cd_y_offaxis, _, _ = measure_cd_1d(
    y,
    offaxis_profile_y
)


# =========================================================
# 19. Print results
# =========================================================

print()
print("=" * 55)
print("2D Center-Line CD Comparison")
print("=" * 55)

print(
    "Target CDx =",
    width_x * 1000,
    "nm"
)

print(
    "Normal CDx =",
    cd_x_normal * 1000,
    "nm"
)

print(
    "Off-axis CDx =",
    cd_x_offaxis * 1000,
    "nm"
)

print()

print(
    "Target CDy =",
    width_y * 1000,
    "nm"
)

print(
    "Normal CDy =",
    cd_y_normal * 1000,
    "nm"
)

print(
    "Off-axis CDy =",
    cd_y_offaxis * 1000,
    "nm"
)
# =========================================================
# 20. Find one threshold crossing near a target edge
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

    transition = np.diff(binary)

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

    # Choose printed edge closest to target edge
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
# 21. Sample positions along target edges
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
# 22. Measure EPE on vertical edges
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


        # Coordinate-based EPE
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
# 23. Measure EPE on horizontal edges
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
# 24. Normal illumination EPE
# =========================================================

normal_left, normal_right = (
    measure_vertical_epe(
        resist_normal
    )
)

normal_bottom, normal_top = (
    measure_horizontal_epe(
        resist_normal
    )
)


# =========================================================
# 25. Off-axis illumination EPE
# =========================================================

offaxis_left, offaxis_right = (
    measure_vertical_epe(
        resist_offaxis
    )
)

offaxis_bottom, offaxis_top = (
    measure_horizontal_epe(
        resist_offaxis
    )
)


# =========================================================
# 26. Combine absolute EPE
# =========================================================

normal_all_epe = np.concatenate([
    normal_left,
    normal_right,
    normal_bottom,
    normal_top
])

offaxis_all_epe = np.concatenate([
    offaxis_left,
    offaxis_right,
    offaxis_bottom,
    offaxis_top
])


normal_abs_epe_nm = (
    np.abs(normal_all_epe)
    *
    1000
)

offaxis_abs_epe_nm = (
    np.abs(offaxis_all_epe)
    *
    1000
)


# =========================================================
# 27. Print EPE metrics
# =========================================================

print()
print("=" * 55)
print("2D EPE Comparison")
print("=" * 55)

print(
    "Normal mean |EPE| =",
    np.nanmean(normal_abs_epe_nm),
    "nm"
)

print(
    "Normal max |EPE| =",
    np.nanmax(normal_abs_epe_nm),
    "nm"
)

print()

print(
    "Off-axis mean |EPE| =",
    np.nanmean(offaxis_abs_epe_nm),
    "nm"
)

print(
    "Off-axis max |EPE| =",
    np.nanmax(offaxis_abs_epe_nm),
    "nm"
)
# =========================================================
# 28. Plot EPE along vertical edges
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    sample_y * 1000,
    np.abs(normal_left) * 1000,
    marker="o",
    label="Normal - Left"
)

plt.plot(
    sample_y * 1000,
    np.abs(normal_right) * 1000,
    marker="o",
    label="Normal - Right"
)

plt.plot(
    sample_y * 1000,
    np.abs(offaxis_left) * 1000,
    marker="o",
    linestyle="--",
    label="Off-axis - Left"
)

plt.plot(
    sample_y * 1000,
    np.abs(offaxis_right) * 1000,
    marker="o",
    linestyle="--",
    label="Off-axis - Right"
)

plt.xlabel("Position along vertical edge y (nm)")
plt.ylabel("|EPE| (nm)")

plt.title("EPE Along Vertical Edges")

plt.grid()
plt.legend()

plt.show()


# =========================================================
# 29. Plot EPE along horizontal edges
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    sample_x * 1000,
    np.abs(normal_bottom) * 1000,
    marker="o",
    label="Normal - Bottom"
)

plt.plot(
    sample_x * 1000,
    np.abs(normal_top) * 1000,
    marker="o",
    label="Normal - Top"
)

plt.plot(
    sample_x * 1000,
    np.abs(offaxis_bottom) * 1000,
    marker="o",
    linestyle="--",
    label="Off-axis - Bottom"
)

plt.plot(
    sample_x * 1000,
    np.abs(offaxis_top) * 1000,
    marker="o",
    linestyle="--",
    label="Off-axis - Top"
)

plt.xlabel("Position along horizontal edge x (nm)")
plt.ylabel("|EPE| (nm)")

plt.title("EPE Along Horizontal Edges")

plt.grid()
plt.legend()

plt.show()