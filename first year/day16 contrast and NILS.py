import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Basic parameters
# =========================================================

N = 4096

x = np.linspace(
    -8.0,
    8.0,
    N
)

dx = x[1] - x[0]

period = 0.20          # um
line_width = 0.10      # um

wavelength = 0.193     # um
NA = 0.85

threshold = 0.5

cutoff = (
    NA
    /
    wavelength
)


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


# =========================================================
# 2. Build centered periodic mask
# =========================================================

position_in_period = (
    np.mod(
        x + period / 2,
        period
    )
    -
    period / 2
)

mask = (
    np.abs(
        position_in_period
    )
    <= line_width / 2
).astype(float)


# =========================================================
# 3. Frequency grid
# =========================================================

freq = np.fft.fftfreq(
    N,
    d=dx
)


# =========================================================
# 4. Fourier spectrum of mask
# =========================================================

spectrum = np.fft.fft(
    mask
)


# =========================================================
# 5. Normal illumination
# =========================================================

pupil_normal = (
    np.abs(freq)
    <= cutoff
).astype(float)

field_normal = np.fft.ifft(
    spectrum
    *
    pupil_normal
)

intensity_normal_raw = (
    np.abs(field_normal) ** 2
)

intensity_normal = (
    intensity_normal_raw
    /
    np.max(intensity_normal_raw)
)


# =========================================================
# 6. Contrast function
# =========================================================

def image_contrast(
    intensity
):

    Imax = np.max(
        intensity
    )

    Imin = np.min(
        intensity
    )

    contrast = (
        (Imax - Imin)
        /
        (Imax + Imin)
    )

    return contrast


# =========================================================
# 7. Normal contrast
# =========================================================

contrast_normal = image_contrast(
    intensity_normal
)


# =========================================================
# 8. Manual dipole example
# =========================================================

source_shift_manual = 1.5


pupil_right_manual = (
    np.abs(
        freq - source_shift_manual
    )
    <= cutoff
).astype(float)


pupil_left_manual = (
    np.abs(
        freq + source_shift_manual
    )
    <= cutoff
).astype(float)


field_right_manual = np.fft.ifft(
    spectrum
    *
    pupil_right_manual
)


field_left_manual = np.fft.ifft(
    spectrum
    *
    pupil_left_manual
)


intensity_right_manual = (
    np.abs(field_right_manual) ** 2
)


intensity_left_manual = (
    np.abs(field_left_manual) ** 2
)


intensity_dipole_manual_raw = (
    0.5 * intensity_right_manual
    +
    0.5 * intensity_left_manual
)


intensity_dipole_manual = (
    intensity_dipole_manual_raw
    /
    np.max(intensity_dipole_manual_raw)
)


contrast_dipole_manual = (
    image_contrast(
        intensity_dipole_manual
    )
)


print()
print("=" * 60)

print(
    "Normal vs Manual Dipole"
)

print("=" * 60)

print(
    "Normal contrast =",
    contrast_normal
)

print(
    "Manual dipole contrast =",
    contrast_dipole_manual
)


# =========================================================
# 9. Plot normal vs manual dipole
# =========================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    x,
    intensity_normal,
    label="Normal illumination"
)

plt.plot(
    x,
    intensity_dipole_manual,
    label="Dipole illumination"
)

plt.xlim(
    -0.5,
    0.5
)

plt.xlabel(
    "x (um)"
)

plt.ylabel(
    "Normalized intensity"
)

plt.title(
    "Normal vs Dipole Illumination"
)

plt.grid()

plt.legend()

plt.show()


# =========================================================
# 10. Source shift scan
# =========================================================

source_shift_scan = np.linspace(
    0.0,
    3.0,
    121
)

contrast_scan = []


for source_shift in source_shift_scan:

    # Right source point
    pupil_right = (
        np.abs(
            freq - source_shift
        )
        <= cutoff
    ).astype(float)

    field_right = np.fft.ifft(
        spectrum
        *
        pupil_right
    )

    intensity_right = (
        np.abs(field_right) ** 2
    )


    # Left source point
    pupil_left = (
        np.abs(
            freq + source_shift
        )
        <= cutoff
    ).astype(float)

    field_left = np.fft.ifft(
        spectrum
        *
        pupil_left
    )

    intensity_left = (
        np.abs(field_left) ** 2
    )


    # Incoherent intensity sum
    intensity_dipole = (
        0.5 * intensity_right
        +
        0.5 * intensity_left
    )


    # Normalize
    intensity_dipole = (
        intensity_dipole
        /
        np.max(intensity_dipole)
    )


    # Contrast
    contrast = image_contrast(
        intensity_dipole
    )

    contrast_scan.append(
        contrast
    )


contrast_scan = np.array(
    contrast_scan
)


# =========================================================
# 11. Best source shift for contrast
# =========================================================

best_contrast_index = np.argmax(
    contrast_scan
)


best_source_shift = (
    source_shift_scan[
        best_contrast_index
    ]
)


best_contrast = (
    contrast_scan[
        best_contrast_index
    ]
)


print()
print("=" * 60)

print(
    "Contrast Optimization"
)

print("=" * 60)

print(
    "Best source shift for contrast =",
    best_source_shift,
    "1/um"
)

print(
    "Best contrast =",
    best_contrast
)


# =========================================================
# 12. NILS function
# =========================================================

def calculate_nils(
    x,
    intensity,
    threshold,
    target_cd
):

    # Convert aerial image to binary print result
    binary = (
        intensity
        >= threshold
    ).astype(float)


    # Find transitions
    transition = np.diff(
        binary
    )


    left_edges = np.where(
        transition == 1
    )[0]


    right_edges = np.where(
        transition == -1
    )[0]


    # Find center
    center_index = np.argmin(
        np.abs(x)
    )


    # Select left edge nearest center
    left_candidates = (
        left_edges[
            left_edges < center_index
        ]
    )


    # Select right edge nearest center
    right_candidates = (
        right_edges[
            right_edges > center_index
        ]
    )


    if (
        len(left_candidates) == 0
        or
        len(right_candidates) == 0
    ):

        return np.nan


    left_index = (
        left_candidates[-1]
    )


    right_index = (
        right_candidates[0]
    )


    # Image slope dI/dx
    dI_dx = np.gradient(
        intensity,
        x
    )


    slope_left = np.abs(
        dI_dx[
            left_index
        ]
    )


    slope_right = np.abs(
        dI_dx[
            right_index
        ]
    )


    # Average left/right edge slope
    slope_edge = (
        slope_left
        +
        slope_right
    ) / 2


    # At threshold crossing:
    # I_edge ≈ threshold
    I_edge = threshold


    nils = (
        target_cd
        /
        I_edge
        *
        slope_edge
    )


    return nils


# =========================================================
# 13. Normal NILS
# =========================================================

nils_normal = calculate_nils(
    x,
    intensity_normal,
    threshold,
    line_width
)


# =========================================================
# 14. Recalculate image using best contrast source
# =========================================================

pupil_right_best = (
    np.abs(
        freq - best_source_shift
    )
    <= cutoff
).astype(float)


pupil_left_best = (
    np.abs(
        freq + best_source_shift
    )
    <= cutoff
).astype(float)


field_right_best = np.fft.ifft(
    spectrum
    *
    pupil_right_best
)


field_left_best = np.fft.ifft(
    spectrum
    *
    pupil_left_best
)


intensity_best_contrast = (
    0.5
    *
    np.abs(field_right_best) ** 2
    +
    0.5
    *
    np.abs(field_left_best) ** 2
)


intensity_best_contrast = (
    intensity_best_contrast
    /
    np.max(intensity_best_contrast)
)


nils_at_best_contrast = calculate_nils(
    x,
    intensity_best_contrast,
    threshold,
    line_width
)


print()
print("=" * 60)

print(
    "NILS at Normal and Best-Contrast Source"
)

print("=" * 60)

print(
    "Normal NILS =",
    nils_normal
)

print(
    "NILS at best contrast source =",
    nils_at_best_contrast
)


# =========================================================
# 15. Scan source shift for NILS
# =========================================================

nils_scan = []


for source_shift in source_shift_scan:

    # Right source
    pupil_right = (
        np.abs(
            freq - source_shift
        )
        <= cutoff
    ).astype(float)


    field_right = np.fft.ifft(
        spectrum
        *
        pupil_right
    )


    intensity_right = (
        np.abs(field_right) ** 2
    )


    # Left source
    pupil_left = (
        np.abs(
            freq + source_shift
        )
        <= cutoff
    ).astype(float)


    field_left = np.fft.ifft(
        spectrum
        *
        pupil_left
    )


    intensity_left = (
        np.abs(field_left) ** 2
    )


    # Incoherent intensity sum
    intensity_dipole = (
        0.5 * intensity_right
        +
        0.5 * intensity_left
    )


    # Normalize
    intensity_dipole = (
        intensity_dipole
        /
        np.max(intensity_dipole)
    )


    nils = calculate_nils(
        x,
        intensity_dipole,
        threshold,
        line_width
    )


    nils_scan.append(
        nils
    )


nils_scan = np.array(
    nils_scan
)


# =========================================================
# 16. Find best source shift for NILS
# =========================================================

best_nils_index = np.nanargmax(
    nils_scan
)


best_nils_source_shift = (
    source_shift_scan[
        best_nils_index
    ]
)


best_nils = (
    nils_scan[
        best_nils_index
    ]
)


print()
print("=" * 60)

print(
    "NILS Optimization"
)

print("=" * 60)

print(
    "Best source shift for contrast =",
    best_source_shift,
    "1/um"
)

print(
    "Best contrast =",
    best_contrast
)

print()

print(
    "Best source shift for NILS =",
    best_nils_source_shift,
    "1/um"
)

print(
    "Best NILS =",
    best_nils
)


# =========================================================
# 17. Plot Contrast vs Source Shift
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    source_shift_scan,
    contrast_scan
)

plt.axvline(
    best_source_shift,
    linestyle="--",
    label=f"Best contrast shift = {best_source_shift:.3f}"
)

plt.xlabel(
    "Source Shift (1/um)"
)

plt.ylabel(
    "Image Contrast"
)

plt.title(
    "Contrast vs Source Shift"
)

plt.grid()

plt.legend()

plt.show()


# =========================================================
# 18. Plot NILS vs Source Shift
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    source_shift_scan,
    nils_scan
)

plt.axvline(
    best_nils_source_shift,
    linestyle="--",
    label=f"Best NILS shift = {best_nils_source_shift:.3f}"
)

plt.xlabel(
    "Source Shift (1/um)"
)

plt.ylabel(
    "NILS"
)

plt.title(
    "NILS vs Source Shift"
)

plt.grid()

plt.legend()

plt.show()


# =========================================================
# 19. Compare aerial images at two optimized sources
# =========================================================

# Best contrast source
pupil_right_contrast = (
    np.abs(
        freq - best_source_shift
    )
    <= cutoff
).astype(float)

pupil_left_contrast = (
    np.abs(
        freq + best_source_shift
    )
    <= cutoff
).astype(float)

field_right_contrast = np.fft.ifft(
    spectrum
    *
    pupil_right_contrast
)

field_left_contrast = np.fft.ifft(
    spectrum
    *
    pupil_left_contrast
)

intensity_contrast_opt = (
    0.5 * np.abs(field_right_contrast) ** 2
    +
    0.5 * np.abs(field_left_contrast) ** 2
)

intensity_contrast_opt = (
    intensity_contrast_opt
    /
    np.max(intensity_contrast_opt)
)


# Best NILS source
pupil_right_nils = (
    np.abs(
        freq - best_nils_source_shift
    )
    <= cutoff
).astype(float)

pupil_left_nils = (
    np.abs(
        freq + best_nils_source_shift
    )
    <= cutoff
).astype(float)

field_right_nils = np.fft.ifft(
    spectrum
    *
    pupil_right_nils
)

field_left_nils = np.fft.ifft(
    spectrum
    *
    pupil_left_nils
)

intensity_nils_opt = (
    0.5 * np.abs(field_right_nils) ** 2
    +
    0.5 * np.abs(field_left_nils) ** 2
)

intensity_nils_opt = (
    intensity_nils_opt
    /
    np.max(intensity_nils_opt)
)


plt.figure(
    figsize=(9, 5)
)

plt.plot(
    x,
    intensity_contrast_opt,
    label="Best Contrast Source"
)

plt.plot(
    x,
    intensity_nils_opt,
    label="Best NILS Source"
)

plt.axhline(
    threshold,
    linestyle="--",
    label="Threshold"
)

plt.xlim(
    -0.3,
    0.3
)

plt.xlabel(
    "x (um)"
)

plt.ylabel(
    "Normalized Intensity"
)

plt.title(
    "Best Contrast vs Best NILS Source"
)

plt.grid()

plt.legend()

plt.show()