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
# 2. 2D periodic line-space mask
# =========================================================

period = 0.20            # um
line_width = 0.10        # um

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

    # Relative frequency coordinates
    # for this source point
    FX_shift = FX - source_x
    FY_shift = FY - source_y

    r2 = (
        FX_shift ** 2
        +
        FY_shift ** 2
    )

    # Circular aperture
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
# 5. Partial-coherent imaging with defocus
# =========================================================

def partial_coherent_image_2d_defocus(
    mask_input,
    source_positions,
    source_weights,
    defocus
):

    # Mask -> frequency domain
    spectrum = np.fft.fft2(
        mask_input
    )

    intensity = np.zeros(
        (N, N),
        dtype=float
    )

    # Abbe:
    # calculate each source point separately,
    # then add intensities
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
# We use ONE fixed reference peak.
# Do not normalize every defocus condition separately.

normal_positions = [
    (0.0, 0.0)
]

normal_weights = [
    1.0
]

intensity_reference = partial_coherent_image_2d_defocus(
    mask,
    normal_positions,
    normal_weights,
    defocus=0.0
)

reference_peak = np.max(
    intensity_reference
)

print(
    "Reference peak =",
    reference_peak
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


    # Subpixel left edge
    left_position = interpolate_edge(
        coordinate[left_index],
        coordinate[left_index + 1],
        profile[left_index],
        profile[left_index + 1],
        threshold
    )


    # Subpixel right edge
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
# 11. Illumination and nominal dose
# =========================================================

source_shift = 0.60

nominal_dose = 0.390


# Horizontal dipole illumination
source_positions = [
    (+source_shift, 0.0),
    (-source_shift, 0.0)
]

source_weights = [
    0.5,
    0.5
]


# =========================================================
# 12. Defocus values
# =========================================================

defocus_list = [
    -0.05,
    -0.025,
     0.00,
    +0.025,
    +0.05
]


# Save profiles for plotting
profile_list = []

cd_list = []


print()
print("=" * 60)
print("Simple Defocus Test")
print("=" * 60)


# =========================================================
# 13. Defocus scan
# =========================================================

for defocus in defocus_list:

    # -----------------------------------------------------
    # Optical imaging
    # -----------------------------------------------------

    intensity = partial_coherent_image_2d_defocus(
        mask,
        source_positions,
        source_weights,
        defocus
    )


    # Fixed normalization
    intensity_norm = (
        intensity
        /
        reference_peak
    )


    # -----------------------------------------------------
    # Dose
    # -----------------------------------------------------

    effective_intensity = (
        nominal_dose
        *
        intensity_norm
    )


    # -----------------------------------------------------
    # Resist
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # CD
    # -----------------------------------------------------

    cd, left_edge, right_edge = measure_cd(
        x,
        profile,
        threshold=0.5
    )


    cd_nm = (
        cd
        *
        1000
    )


    cd_list.append(
        cd_nm
    )

    profile_list.append(
        profile
    )


    print(
        f"Defocus = {defocus:+.2f} um, "
        f"CD = {cd_nm:.3f} nm"
    )


# =========================================================
# 14. Plot resist profiles
# =========================================================

plt.figure(
    figsize=(8, 5)
)


for defocus, profile in zip(
    defocus_list,
    profile_list
):

    plt.plot(
        x,
        profile,
        label=f"Defocus = {defocus:+.2f} um"
    )


plt.axhline(
    0.5,
    linestyle="--",
    label="Resist threshold"
)

plt.xlim(
    -0.15,
    0.15
)

plt.xlabel(
    "x (um)"
)

plt.ylabel(
    "Resist response"
)

plt.title(
    "Resist Profiles under Defocus"
)

plt.grid()
plt.legend()

plt.show()


# =========================================================
# 15. Plot CD vs defocus
# =========================================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    defocus_list,
    cd_list,
    marker="o"
)

plt.axhline(
    target_cd * 1000,
    linestyle="--",
    label="Target CD"
)

plt.xlabel(
    "Defocus (um)"
)

plt.ylabel(
    "Printed CD (nm)"
)

plt.title(
    "CD vs Defocus"
)

plt.grid()
plt.legend()

plt.show()
# =========================================================
# Fine focus scan for DOF
# =========================================================

focus_list = np.linspace(
    -0.15,
    0.15,
    301
)

cd_focus_list = []


for defocus in focus_list:

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
        nominal_dose
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

    cd_focus_list.append(
        cd * 1000
    )


cd_focus_array = np.array(
    cd_focus_list
)
plt.figure(figsize=(8, 5))

plt.plot(
    focus_list * 1000,
    cd_focus_array
)

plt.axhline(
    100,
    linestyle="--",
    label="Target CD"
)

plt.axhline(
    98,
    linestyle="--",
    label="CD lower limit"
)

plt.axhline(
    102,
    linestyle="--",
    label="CD upper limit"
)

plt.xlabel(
    "Defocus (nm)"
)

plt.ylabel(
    "Printed CD (nm)"
)

plt.title(
    "CD vs Focus"
)
plt.ylim(90,105)
plt.grid()
plt.legend()

plt.show()
# =========================================================
# Calculate DOF from the continuous valid region
# around best focus
# =========================================================

cd_min_nm = 98.0
cd_max_nm = 102.0


valid = (
    np.isfinite(cd_focus_array)
    &
    (cd_focus_array >= cd_min_nm)
    &
    (cd_focus_array <= cd_max_nm)
)


# Index closest to best focus
center_index_focus = np.argmin(
    np.abs(focus_list)
)


# Search left from best focus
left_index = center_index_focus

while (
    left_index > 0
    and
    valid[left_index - 1]
):
    left_index -= 1


# Search right from best focus
right_index = center_index_focus

while (
    right_index < len(focus_list) - 1
    and
    valid[right_index + 1]
):
    right_index += 1


focus_min = focus_list[
    left_index
]

focus_max = focus_list[
    right_index
]


DOF = (
    focus_max
    -
    focus_min
)


print()
print("=" * 60)
print("Depth of Focus")
print("=" * 60)

print(
    "Valid focus range =",
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