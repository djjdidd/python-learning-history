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
# 2. 1D line-space mask
# =========================================================

period = 0.30       # μm
line_width = 0.15   # μm

position_in_period = np.mod(
    x,
    period
)

mask = (
    position_in_period
    <
    line_width
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
NA = 0.5

cutoff = (
    NA / wavelength
)

# =========================================================
# 5. Mask spectrum
# =========================================================

spectrum = np.fft.fft(
    mask
)


# =========================================================
# 6. Normal illumination
# =========================================================

pupil_normal = (
    np.abs(freq)
    <=
    cutoff
).astype(float)

filtered_normal = (
    spectrum
    *
    pupil_normal
)

field_normal = np.fft.ifft(
    filtered_normal
)

intensity_normal = (
    np.abs(field_normal) ** 2
)

intensity_normal_plot = (
    intensity_normal
    /
    np.max(intensity_normal)
)


# =========================================================
# 7. Normal contrast
# =========================================================

Imax_normal = np.max(
    intensity_normal_plot
)

Imin_normal = np.min(
    intensity_normal_plot
)

contrast_normal = (
    (Imax_normal - Imin_normal)
    /
    (Imax_normal + Imin_normal)
)

# =========================================================
# 8. Source shift scan
# =========================================================

source_shift_list = np.linspace(
    0.5,
    0.9,
    81
)

contrast_list = []

intensity_list = []


# =========================================================
# 9. Scan all dipole source positions
# =========================================================

for source_shift in source_shift_list:

    # Two symmetric source points
    source_points = [
        source_shift,
        -source_shift
    ]

    weight = 0.5

    intensity_dipole = np.zeros_like(
        x,
        dtype=float
    )


    # -----------------------------------------------------
    # Each source point forms one coherent image
    # -----------------------------------------------------

    for source_f in source_points:

        # Shifted pupil
        pupil_shifted = (
            np.abs(
                freq - source_f
            )
            <=
            cutoff
        ).astype(float)


        # Filter mask spectrum
        filtered = (
            spectrum
            *
            pupil_shifted
        )


        # Frequency domain -> spatial domain
        field = np.fft.ifft(
            filtered
        )


        # Complex field -> intensity
        intensity = (
            np.abs(field) ** 2
        )


        # Incoherent weighted sum
        intensity_dipole += (
            weight
            *
            intensity
        )


    # -----------------------------------------------------
    # Normalize only for image-shape comparison
    # -----------------------------------------------------

    intensity_dipole_plot = (
        intensity_dipole
        /
        np.max(intensity_dipole)
    )


    # -----------------------------------------------------
    # Contrast
    # -----------------------------------------------------

    Imax = np.max(
        intensity_dipole_plot
    )

    Imin = np.min(
        intensity_dipole_plot
    )

    contrast = (
        (Imax - Imin)
        /
        (Imax + Imin)
    )


    contrast_list.append(
        contrast
    )

    intensity_list.append(
        intensity_dipole_plot.copy()
    )


# =========================================================
# 10. Convert to numpy array
# =========================================================

contrast_list = np.array(
    contrast_list
)


# =========================================================
# 11. Find best source shift
# =========================================================

best_index = np.argmax(
    contrast_list
)

best_source_shift = (
    source_shift_list[
        best_index
    ]
)

best_contrast = (
    contrast_list[
        best_index
    ]
)

best_intensity = (
    intensity_list[
        best_index
    ]
)


# =========================================================
# 13. Plot source shift vs contrast
# =========================================================

plt.figure(
    figsize=(12, 5)
)
plt.subplot(1,3,1)
plt.plot(
    source_shift_list,
    contrast_list,
    marker="o"
)

plt.axhline(
    contrast_normal,
    linestyle="--",
    label="Normal illumination"
)

plt.axvline(
    best_source_shift,
    linestyle="--",
    label="Best source shift"
)

plt.xlabel(
    "Source Shift (1/μm)"
)

plt.ylabel(
    "Image Contrast"
)

plt.title(
    "Source Position Optimization"
)



# =========================================================
# 15. Plot mask spectrum
# =========================================================

freq_display = np.fft.fftshift(
    freq
)

spectrum_display = np.abs(
    np.fft.fftshift(
        spectrum
    )
)

spectrum_display = (
    spectrum_display
    /
    np.max(spectrum_display)
)


plt.subplot(1, 3, 2)

plt.plot(
    freq_display,
    spectrum_display,
    label="Mask Spectrum"
)


# Normal pupil
plt.axvspan(
    -cutoff,
    cutoff,
    alpha=0.2,
    label="Normal Pupil"
)


# Best +source pupil
plt.axvspan(
    best_source_shift - cutoff,
    best_source_shift + cutoff,
    alpha=0.2,
    label="+ Best Dipole Pupil"
)


# Best -source pupil
plt.axvspan(
    -best_source_shift - cutoff,
    -best_source_shift + cutoff,
    alpha=0.2,
    label="- Best Dipole Pupil"
)


plt.xlim(
    -8,
    8
)

plt.xlabel(
    "Spatial Frequency (1/μm)"
)

plt.ylabel(
    "Normalized Spectrum"
)

plt.title(
    "Spectrum and Optimized Pupil Coverage"
)

# =========================================================
# Fine scan around the coarse optimum
# =========================================================

source_shift_fine = np.linspace(
    0.5,
    0.9,
    41
)

contrast_fine = []


for source_shift in source_shift_fine:

    source_points = [
        source_shift,
        -source_shift
    ]

    intensity_dipole = np.zeros_like(
        x,
        dtype=float
    )

    weight = 0.5


    for source_f in source_points:

        pupil_shifted = (
            np.abs(
                freq - source_f
            )
            <=
            cutoff
        ).astype(float)


        filtered = (
            spectrum
            *
            pupil_shifted
        )


        field = np.fft.ifft(
            filtered
        )


        intensity = (
            np.abs(field) ** 2
        )


        intensity_dipole += (
            weight
            *
            intensity
        )


    # Normalize for contrast comparison
    intensity_plot = (
        intensity_dipole
        /
        np.max(intensity_dipole)
    )


    Imax = np.max(
        intensity_plot
    )

    Imin = np.min(
        intensity_plot
    )


    contrast = (
        (Imax - Imin)
        /
        (Imax + Imin)
    )


    contrast_fine.append(
        contrast
    )


# Convert to NumPy array
contrast_fine = np.array(
    contrast_fine
)


# Find fine optimum
best_index_fine = np.argmax(
    contrast_fine
)

best_source_fine = (
    source_shift_fine[
        best_index_fine
    ]
)

best_contrast_fine = (
    contrast_fine[
        best_index_fine
    ]
)


print()
print("=" * 60)

print(
    "Fine best source shift =",
    best_source_fine,
    "1/μm"
)

print(
    "Fine best contrast =",
    best_contrast_fine
)

print("=" * 60)


# Plot fine scan
plt.subplot(1, 3, 3)

plt.plot(
    source_shift_fine,
    contrast_fine,
    marker="o"
)

plt.axvline(
    best_source_fine,
    linestyle="--",
    label=(
        f"Best = "
        f"{best_source_fine:.2f}"
    )
)

plt.xlabel(
    "Source Shift (1/μm)"
)

plt.ylabel(
    "Image Contrast"
)

plt.title(
    "Fine Source Position Scan"
)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# =========================================================
# Compare:
# Normal vs manual dipole vs optimized dipole
# =========================================================

manual_shift = 1.5
optimized_shift = 0.78


def dipole_image(source_shift):

    source_points = [
        source_shift,
        -source_shift
    ]

    intensity_total = np.zeros_like(
        x,
        dtype=float
    )

    for source_f in source_points:

        pupil_shifted = (
            np.abs(freq - source_f)
            <= cutoff
        ).astype(float)

        filtered = (
            spectrum
            * pupil_shifted
        )

        field = np.fft.ifft(
            filtered
        )

        intensity = (
            np.abs(field) ** 2
        )

        intensity_total += (
            0.5 * intensity
        )

    intensity_plot = (
        intensity_total
        /
        np.max(intensity_total)
    )

    return intensity_plot


# ---------------------------------------------------------
# Two dipole images
# ---------------------------------------------------------

intensity_manual = dipole_image(
    manual_shift
)

intensity_optimized = dipole_image(
    optimized_shift
)


# =========================================================
# Contrast function
# =========================================================

def calculate_contrast(
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


contrast_manual = calculate_contrast(
    intensity_manual
)

contrast_optimized = calculate_contrast(
    intensity_optimized
)


# =========================================================
# NILS function
# =========================================================

def calculate_nils(
    intensity,
    edge_position,
    CD
):

    gradient = np.gradient(
        intensity,
        x
    )

    edge_index = np.argmin(
        np.abs(
            x - edge_position
        )
    )

    I_edge = intensity[
        edge_index
    ]

    slope = np.abs(
        gradient[
            edge_index
        ]
    )

    NILS = (
        CD
        /
        I_edge
        *
        slope
    )

    return (
        NILS,
        I_edge,
        slope
    )


edge_position = line_width
CD = line_width


# Normal
(
    nils_normal,
    Iedge_normal,
    slope_normal
) = calculate_nils(
    intensity_normal_plot,
    edge_position,
    CD
)


# Manual dipole
(
    nils_manual,
    Iedge_manual,
    slope_manual
) = calculate_nils(
    intensity_manual,
    edge_position,
    CD
)


# Optimized dipole
(
    nils_optimized,
    Iedge_optimized,
    slope_optimized
) = calculate_nils(
    intensity_optimized,
    edge_position,
    CD
)


# =========================================================
# Print comparison
# =========================================================

print()
print("=" * 70)

print("Normal illumination")
print(
    "Contrast =",
    contrast_normal
)
print(
    "NILS =",
    nils_normal
)

print()

print("Manual dipole, shift = 1.5")
print(
    "Contrast =",
    contrast_manual
)
print(
    "NILS =",
    nils_manual
)

print()

print("Optimized dipole, shift = 0.78")
print(
    "Contrast =",
    contrast_optimized
)
print(
    "NILS =",
    nils_optimized
)

print("=" * 70)


# =========================================================
# Plot aerial images
# =========================================================

plt.figure(
    figsize=(11, 5)
)

plt.plot(
    x,
    intensity_normal_plot,
    label="Normal"
)

plt.plot(
    x,
    intensity_manual,
    label="Manual Dipole, shift=1.5"
)

plt.plot(
    x,
    intensity_optimized,
    label="Optimized Dipole, shift=0.78"
)

plt.xlabel(
    "x (μm)"
)

plt.ylabel(
    "Normalized Intensity"
)

plt.title(
    "Illumination Comparison"
)

plt.legend()

plt.grid()

plt.show()
# =========================================================
# Scan source shift:
# Contrast vs NILS
# =========================================================

source_shift_scan = np.linspace(
    0.0,
    2.0,
    81
)

contrast_scan = []
nils_scan = []


for source_shift in source_shift_scan:

    # -------------------------------------
    # Build dipole image
    # -------------------------------------

    intensity_dipole = np.zeros_like(
        x,
        dtype=float
    )

    for source_f in [
        source_shift,
        -source_shift
    ]:

        pupil_shifted = (
            np.abs(
                freq - source_f
            )
            <= cutoff
        ).astype(float)

        filtered = (
            spectrum
            * pupil_shifted
        )

        field = np.fft.ifft(
            filtered
        )

        intensity = (
            np.abs(field) ** 2
        )

        intensity_dipole += (
            0.5 * intensity
        )


    # -------------------------------------
    # Normalize for shape comparison
    # -------------------------------------

    intensity_plot = (
        intensity_dipole
        /
        np.max(intensity_dipole)
    )


    # -------------------------------------
    # Contrast
    # -------------------------------------

    Imax = np.max(
        intensity_plot
    )

    Imin = np.min(
        intensity_plot
    )

    contrast = (
        (Imax - Imin)
        /
        (Imax + Imin)
    )

    contrast_scan.append(
        contrast
    )


    # -------------------------------------
    # NILS
    # -------------------------------------

    gradient = np.gradient(
        intensity_plot,
        x
    )

    edge_position = line_width

    edge_index = np.argmin(
        np.abs(
            x - edge_position
        )
    )

    I_edge = intensity_plot[
        edge_index
    ]

    slope = np.abs(
        gradient[
            edge_index
        ]
    )

    NILS = (
        line_width
        /
        I_edge
        *
        slope
    )

    nils_scan.append(
        NILS
    )


# =========================================================
# Convert to arrays
# =========================================================

contrast_scan = np.array(
    contrast_scan
)

nils_scan = np.array(
    nils_scan
)


# =========================================================
# Find best source positions
# =========================================================

best_contrast_index = np.argmax(
    contrast_scan
)

best_nils_index = np.argmax(
    nils_scan
)


best_shift_contrast = (
    source_shift_scan[
        best_contrast_index
    ]
)

best_shift_nils = (
    source_shift_scan[
        best_nils_index
    ]
)


print()
print("=" * 60)

print(
    "Best source for contrast =",
    best_shift_contrast,
    "1/μm"
)

print(
    "Maximum contrast =",
    contrast_scan[
        best_contrast_index
    ]
)

print()

print(
    "Best source for NILS =",
    best_shift_nils,
    "1/μm"
)

print(
    "Maximum NILS =",
    nils_scan[
        best_nils_index
    ]
)

print("=" * 60)


# =========================================================
# Plot Contrast and NILS
# =========================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    source_shift_scan,
    contrast_scan,
    label="Contrast"
)

plt.plot(
    source_shift_scan,
    nils_scan,
    label="NILS"
)

plt.axvline(
    best_shift_contrast,
    linestyle="--",
    label=(
        f"Best Contrast = "
        f"{best_shift_contrast:.2f}"
    )
)

plt.axvline(
    best_shift_nils,
    linestyle="--",
    label=(
        f"Best NILS = "
        f"{best_shift_nils:.2f}"
    )
)

plt.xlabel(
    "Source Shift (1/μm)"
)

plt.ylabel(
    "Metric Value"
)

plt.title(
    "Contrast and NILS vs Source Shift"
)

plt.legend()

plt.grid()

plt.show()
# =========================================================
# Multi-objective source optimization
# =========================================================

# Normalize both metrics
contrast_norm = (
    contrast_scan
    /
    np.max(contrast_scan)
)

nils_norm = (
    nils_scan
    /
    np.max(nils_scan)
)


# ---------------------------------------------------------
# Weights
# ---------------------------------------------------------

w_contrast = 0.5
w_nils = 0.5


# ---------------------------------------------------------
# Combined score
# ---------------------------------------------------------

score = (
    w_contrast
    * contrast_norm

    +

    w_nils
    * nils_norm
)


# ---------------------------------------------------------
# Find best compromise source
# ---------------------------------------------------------

best_score_index = np.argmax(
    score
)

best_shift_score = (
    source_shift_scan[
        best_score_index
    ]
)

best_score = (
    score[
        best_score_index
    ]
)


best_score_contrast = (
    contrast_scan[
        best_score_index
    ]
)

best_score_nils = (
    nils_scan[
        best_score_index
    ]
)


print()
print("=" * 60)

print(
    "Best compromise source shift =",
    best_shift_score,
    "1/μm"
)

print(
    "Combined score =",
    best_score
)

print(
    "Contrast at this source =",
    best_score_contrast
)

print(
    "NILS at this source =",
    best_score_nils
)

print("=" * 60)


# =========================================================
# Plot normalized metrics and score
# =========================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    source_shift_scan,
    contrast_norm,
    label="Normalized Contrast"
)

plt.plot(
    source_shift_scan,
    nils_norm,
    label="Normalized NILS"
)

plt.plot(
    source_shift_scan,
    score,
    label="Combined Score"
)

plt.axvline(
    best_shift_score,
    linestyle="--",
    label=(
        f"Best Compromise = "
        f"{best_shift_score:.3f}"
    )
)

plt.xlabel(
    "Source Shift (1/μm)"
)

plt.ylabel(
    "Normalized Metric"
)

plt.title(
    "Multi-objective Source Optimization"
)

plt.legend()

plt.grid()

plt.show()