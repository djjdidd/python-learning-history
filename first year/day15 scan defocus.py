import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Spatial grid
# =========================================================

N = 4096

x = np.linspace(
    -8.0,
    8.0,
    N
)

dx = x[1] - x[0]


# =========================================================
# 2. Periodic line-space mask
# =========================================================

period = 0.4       # μm
line_width = 0.2   # μm


# Make the center line symmetric around x = 0
position_in_period = (
    np.mod(
        x + period / 2,
        period
    )
    - period / 2
)

mask = (
    np.abs(position_in_period)
    <=
    line_width / 2
).astype(float)


# =========================================================
# 3. Frequency grid
# =========================================================

freq = np.fft.fftfreq(
    N,
    d=dx
)


# =========================================================
# 4. Optical parameters
# =========================================================

wavelength = 0.193   # μm
NA = 0.85

cutoff = (
    NA / wavelength
)

print(
    "Cutoff frequency =",
    cutoff,
    "1/μm"
)


# =========================================================
# 5. Mask spectrum
# =========================================================

spectrum = np.fft.fft(
    mask
)


# =========================================================
# 6. Defocus pupil function
# =========================================================

def make_defocus_pupil(
    freq,
    cutoff,
    defocus_strength
):

    # -------------------------------------
    # Aperture
    # -------------------------------------

    aperture = (
        np.abs(freq)
        <=
        cutoff
    ).astype(float)


    # -------------------------------------
    # Defocus phase
    #
    # This is a simplified teaching model.
    # defocus_strength is dimensionless here.
    # -------------------------------------

    normalized_frequency = (
        freq / cutoff
    )

    phase = (
        defocus_strength
        *
        normalized_frequency**2
    )


    # -------------------------------------
    # Complex pupil
    # -------------------------------------

    pupil = (
        aperture
        *
        np.exp(
            1j * phase
        )
    )

    return pupil


# =========================================================
# 7. Fourier imaging function
# =========================================================

def imaging_with_defocus(
    spectrum,
    freq,
    cutoff,
    defocus_strength
):

    # Build defocused pupil
    pupil = make_defocus_pupil(
        freq,
        cutoff,
        defocus_strength
    )


    # Optical filtering
    filtered_spectrum = (
        spectrum
        *
        pupil
    )


    # Back to spatial domain
    field = np.fft.ifft(
        filtered_spectrum
    )


    # Complex field -> intensity
    intensity = (
        np.abs(field) ** 2
    )


    # Normalize only for shape comparison
    intensity = (
        intensity
        /
        np.max(intensity)
    )


    return intensity


# =========================================================
# 8. Defocus values
# =========================================================
defocus_scan = np.linspace(
    0.0,
    4.0,
    41
)
# =========================================================
# 9. Calculate aerial images
# =========================================================

intensity_list = []


for defocus_strength in defocus_scan:

    intensity = imaging_with_defocus(
        spectrum,
        freq,
        cutoff,
        defocus_strength
    )

    intensity_list.append(
        intensity
    )


# =========================================================
# 10. Plot mask
# =========================================================

plt.figure(
    figsize=(10, 4)
)

plt.plot(
    x,
    mask
)

plt.xlim(
    -0.5,
    0.5
)

plt.xlabel(
    "x (μm)"
)

plt.ylabel(
    "Mask Transmission"
)

plt.title(
    "Periodic Line-Space Mask"
)

plt.grid()

plt.show()


# =========================================================
# 11. Plot aerial images
# =========================================================

plt.figure(
    figsize=(10, 5)
)


for defocus_strength, intensity in zip(
    defocus_scan,
    intensity_list
):

    plt.plot(
        x,
        intensity,
        label=(
            f"Defocus = "
            f"{defocus_strength}"
        )
    )


# Threshold reference line
threshold = 0.5

plt.axhline(
    threshold,
    linestyle="--",
    label="Threshold = 0.5"
)


# Zoom into the central feature
plt.xlim(
    -0.4,
    0.4
)

plt.ylim(
    0,
    1.05
)

plt.xlabel(
    "x (μm)"
)

plt.ylabel(
    "Normalized Intensity"
)

plt.title(
    "Effect of Defocus on Aerial Image"
)

plt.legend()

plt.grid()

plt.show()


# =========================================================
# 12. Plot pupil phase
# =========================================================

plt.figure(
    figsize=(10, 5)
)


for defocus_strength in defocus_scan:

    pupil = make_defocus_pupil(
        freq,
        cutoff,
        defocus_strength
    )

    pupil_shifted = np.fft.fftshift(
        pupil
    )

    freq_shifted = np.fft.fftshift(
        freq
    )

    phase_shifted = np.angle(
        pupil_shifted
    )

    plt.plot(
        freq_shifted,
        phase_shifted,
        label=(
            f"Defocus = "
            f"{defocus_strength}"
        )
    )


plt.xlim(
    -cutoff * 1.2,
    cutoff * 1.2
)

plt.xlabel(
    "Spatial Frequency (1/μm)"
)

plt.ylabel(
    "Pupil Phase (rad)"
)

plt.title(
    "Defocus Phase Across the Pupil"
)

plt.legend()

plt.grid()

plt.show()

# =========================================================
# 13. Function:
# sub-pixel threshold crossing
# =========================================================

def interpolate_edge(
    x1,
    x2,
    I1,
    I2,
    threshold
):

    edge_x = (
        x1
        +
        (threshold - I1)
        /
        (I2 - I1)
        *
        (x2 - x1)
    )

    return edge_x


# =========================================================
# 14. Function:
# measure center-line CD and EPE
# =========================================================

def measure_cd_epe(
    x,
    intensity,
    threshold,
    target_left_edge,
    target_right_edge
):

    # -------------------------------------
    # 1. Convert intensity to binary resist
    # -------------------------------------

    resist_binary = (
        intensity >= threshold
    ).astype(float)


    # -------------------------------------
    # 2. Find 0->1 / 1->0 transitions
    # -------------------------------------

    transition = np.diff(
        resist_binary
    )

    left_edges = np.where(
        transition == 1
    )[0]

    right_edges = np.where(
        transition == -1
    )[0]


    # -------------------------------------
    # 3. Find the center feature
    # -------------------------------------

    center_index = np.argmin(
        np.abs(x)
    )

    left_candidates = left_edges[
        left_edges < center_index
    ]

    right_candidates = right_edges[
        right_edges > center_index
    ]


    if (
        len(left_candidates) == 0
        or
        len(right_candidates) == 0
    ):

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )


    left_index = (
        left_candidates[-1]
    )

    right_index = (
        right_candidates[0]
    )


    # -------------------------------------
    # 4. Sub-pixel left edge
    # -------------------------------------

    x_left = interpolate_edge(
        x[left_index],
        x[left_index + 1],
        intensity[left_index],
        intensity[left_index + 1],
        threshold
    )


    # -------------------------------------
    # 5. Sub-pixel right edge
    # -------------------------------------

    x_right = interpolate_edge(
        x[right_index],
        x[right_index + 1],
        intensity[right_index],
        intensity[right_index + 1],
        threshold
    )


    # -------------------------------------
    # 6. Printed CD
    # -------------------------------------

    printed_cd = (
        x_right
        -
        x_left
    )


    # -------------------------------------
    # 7. EPE
    # -------------------------------------

    epe_left = (
        x_left
        -
        target_left_edge
    )

    epe_right = (
        x_right
        -
        target_right_edge
    )


    return (
        printed_cd,
        epe_left,
        epe_right,
        x_left,
        x_right
    )


# =========================================================
# 15. Target edges
# =========================================================

target_left_edge = (
    -line_width / 2
)

target_right_edge = (
    line_width / 2
)

cd_scan = []

for defocus_strength in defocus_scan:

    intensity = imaging_with_defocus(
        spectrum,
        freq,
        cutoff,
        defocus_strength
    )

    (
        printed_cd,
        epe_left,
        epe_right,
        x_left,
        x_right
    ) = measure_cd_epe(
        x,
        intensity,
        threshold,
        target_left_edge,
        target_right_edge
    )

    cd_scan.append(
        printed_cd
    )
cd_scan = np.array(
    cd_scan
)
plt.figure(figsize=(8, 5))

plt.plot(
    defocus_scan,
    cd_scan * 1000,
    marker="o"
)

plt.axhline(
    line_width * 1000,
    linestyle="--",
    label="Target CD"
)
plt.ylim(
    130,140)
plt.xlabel(
    "Defocus Strength"
)

plt.ylabel(
    "Printed CD (nm)"
)

plt.title(
    "CD vs Defocus"
)

plt.legend()

plt.grid()

plt.show()
# =========================================================
# Dose-aware imaging
# =========================================================

def imaging_with_defocus_and_dose(
    spectrum,
    freq,
    cutoff,
    defocus_strength,
    dose
):

    pupil = make_defocus_pupil(
        freq,
        cutoff,
        defocus_strength
    )

    filtered_spectrum = (
        spectrum
        *
        pupil
    )

    field = np.fft.ifft(
        filtered_spectrum
    )

    intensity = (
        np.abs(field) ** 2
    )

    # Visualization/reference normalization
    intensity = (
        intensity
        /
        np.max(intensity)
    )

    # Dose scaling
    effective_intensity = (
        dose
        *
        intensity
    )

    return effective_intensity


# =========================================================
# Dose scan at best focus
# =========================================================

dose_scan = np.linspace(
    0.5,
    2.5,
    61
)

cd_vs_dose = []


for dose in dose_scan:

    intensity = imaging_with_defocus_and_dose(
        spectrum,
        freq,
        cutoff,
        defocus_strength=0.0,
        dose=dose
    )

    (
        printed_cd,
        epe_left,
        epe_right,
        x_left,
        x_right
    ) = measure_cd_epe(
        x,
        intensity,
        threshold,
        target_left_edge,
        target_right_edge
    )

    cd_vs_dose.append(
        printed_cd
    )


cd_vs_dose = np.array(
    cd_vs_dose
)


# =========================================================
# Find nominal dose
# =========================================================

target_cd = line_width

cd_error = np.abs(
    cd_vs_dose
    -
    target_cd
)

best_index = np.nanargmin(
    cd_error
)

nominal_dose = (
    dose_scan[
        best_index
    ]
)

nominal_cd = (
    cd_vs_dose[
        best_index
    ]
)


print()
print("=" * 60)

print(
    "Nominal dose =",
    nominal_dose
)

print(
    "Printed CD at nominal dose =",
    nominal_cd * 1000,
    "nm"
)

print("=" * 60)


# =========================================================
# Plot CD vs Dose
# =========================================================

plt.figure(
    figsize=(8, 5)
)
plt.ylim(100,200)
plt.plot(
    dose_scan,
    cd_vs_dose * 1000,
    marker="o"
)

plt.axhline(
    target_cd * 1000,
    linestyle="--",
    label="Target CD"
)

plt.axvline(
    nominal_dose,
    linestyle="--",
    label="Nominal Dose"
)

plt.xlabel(
    "Dose"
)

plt.ylabel(
    "Printed CD (nm)"
)

plt.title(
    "CD vs Dose at Best Focus"
)

plt.legend()

plt.grid()

plt.show()