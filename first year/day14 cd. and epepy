import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Spatial grid
# =========================================================

N = 2000

x = np.linspace(
    -2.0,
    2.0,
    N
)

dx = x[1] - x[0]


# =========================================================
# 2. Periodic line-space mask
# =========================================================

period = 0.4
line_width = 0.2

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
# 3. Simple optical blur using Gaussian PSF
# =========================================================

sigma = 0.05

psf = np.exp(
    -x**2
    /
    (2 * sigma**2)
)

psf = (
    psf
    /
    np.sum(psf)
)


# =========================================================
# 4. Convolution
# =========================================================

# Use FFT convolution manually

mask_fft = np.fft.fft(
    mask
)

psf_shifted = np.fft.ifftshift(
    psf
)

psf_fft = np.fft.fft(
    psf_shifted
)

aerial_image = np.real(
    np.fft.ifft(
        mask_fft
        *
        psf_fft
    )
)


# Normalize for visualization
aerial_image = (
    aerial_image
    /
    np.max(aerial_image)
)


# =========================================================
# 5. Threshold -> binary resist
# =========================================================

threshold = 0.5

resist_binary = (
    aerial_image
    >=
    threshold
).astype(float)


# =========================================================
# 6. Find transitions
# =========================================================

transition = np.diff(
    resist_binary
)


# 0 -> 1
left_edges = np.where(
    transition == 1
)[0]


# 1 -> 0
right_edges = np.where(
    transition == -1
)[0]


print(
    "Left edge indices =",
    left_edges
)

print(
    "Right edge indices =",
    right_edges
)


# =========================================================
# 7. Find center index
# =========================================================

center_index = np.argmin(
    np.abs(x)
)

print(
    "Center index =",
    center_index
)

print(
    "Center x =",
    x[center_index]
)


# =========================================================
# 8. Find nearest left/right edges around center
# =========================================================

left_candidates = left_edges[
    left_edges
    <
    center_index
]

right_candidates = right_edges[
    right_edges
    >
    center_index
]


# Safety check
if (
    len(left_candidates) == 0
    or
    len(right_candidates) == 0
):

    raise ValueError(
        "Could not find valid left/right edges around center."
    )


left_index = (
    left_candidates[-1]
)

right_index = (
    right_candidates[0]
)


# =========================================================
# 9. Convert index -> physical position
# =========================================================

x_left = x[
    left_index
]

x_right = x[
    right_index
]


# =========================================================
# 10. Calculate printed CD
# =========================================================

printed_cd = (
    x_right
    -
    x_left
)


print()
print("=" * 60)

print(
    "Left edge x =",
    x_left,
    "μm"
)

print(
    "Right edge x =",
    x_right,
    "μm"
)

print(
    "Printed CD =",
    printed_cd,
    "μm"
)

print("=" * 60)


# =========================================================
# 11. Plot everything
# =========================================================

plt.figure(
    figsize=(12, 8)
)


# ---------------------------------------------------------
# Mask
# ---------------------------------------------------------

plt.subplot(
    3,
    1,
    1
)

plt.plot(
    x,
    mask
)

plt.title(
    "Mask"
)

plt.ylabel(
    "Transmission"
)

plt.grid()


# ---------------------------------------------------------
# Aerial image
# ---------------------------------------------------------

plt.subplot(
    3,
    1,
    2
)

plt.plot(
    x,
    aerial_image,
    label="Aerial Image"
)

plt.axhline(
    threshold,
    linestyle="--",
    label="Threshold"
)

plt.axvline(
    x_left,
    linestyle="--",
    label="Left Edge"
)

plt.axvline(
    x_right,
    linestyle="--",
    label="Right Edge"
)

plt.title(
    "Aerial Image and Detected Edges"
)

plt.ylabel(
    "Normalized Intensity"
)

plt.legend()

plt.grid()


# ---------------------------------------------------------
# Binary resist
# ---------------------------------------------------------

plt.subplot(
    3,
    1,
    3
)

plt.plot(
    x,
    resist_binary
)

plt.axvline(
    x_left,
    linestyle="--"
)

plt.axvline(
    x_right,
    linestyle="--"
)

plt.title(
    f"Binary Resist, Printed CD = {printed_cd:.4f} μm"
)

plt.xlabel(
    "x (μm)"
)

plt.ylabel(
    "Resist"
)

plt.grid()


plt.tight_layout()

plt.show()
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
x_left_interp = interpolate_edge(
    x[left_index],
    x[left_index + 1],
    aerial_image[left_index],
    aerial_image[left_index + 1],
    threshold
)
x_right_interp = interpolate_edge(
    x[right_index],
    x[right_index + 1],
    aerial_image[right_index],
    aerial_image[right_index + 1],
    threshold
)
printed_cd_interp = (
    x_right_interp
    -
    x_left_interp
)
print()
print("Pixel-based CD =", printed_cd, "μm")

print(
    "Interpolated left edge =",
    x_left_interp,
    "μm"
)

print(
    "Interpolated right edge =",
    x_right_interp,
    "μm"
)

print(
    "Sub-pixel printed CD =",
    printed_cd_interp,
    "μm"
)
# =========================================================
# EPE calculation
# =========================================================

target_cd = line_width

target_left_edge = 0

target_right_edge = line_width


# Printed edges from interpolation
printed_left_edge = (
    x_left_interp
)

printed_right_edge = (
    x_right_interp
)


# ---------------------------------------------------------
# EPE
# ---------------------------------------------------------

EPE_left = (
    printed_left_edge
    -
    target_left_edge
)

EPE_right = (
    printed_right_edge
    -
    target_right_edge
)


print()
print("=" * 60)

print(
    "Target left edge =",
    target_left_edge,
    "μm"
)

print(
    "Printed left edge =",
    printed_left_edge,
    "μm"
)

print(
    "Left EPE =",
    EPE_left,
    "μm"
)


print()

print(
    "Target right edge =",
    target_right_edge,
    "μm"
)

print(
    "Printed right edge =",
    printed_right_edge,
    "μm"
)

print(
    "Right EPE =",
    EPE_right,
    "μm"
)

print("=" * 60)
# =========================================================
# Plot target edges and printed edges
# =========================================================

plt.figure(figsize=(12, 5))
plt.xlim(-0.25, 0.25)
# Aerial image
plt.plot(
    x,
    aerial_image,
    label="Aerial Image"
)

# Threshold
plt.axhline(
    threshold,
    linestyle="--",
    label="Threshold"
)


# ---------------------------------------------------------
# Target edges
# ---------------------------------------------------------

plt.axvline(
    target_left_edge,
    linestyle="--",
    label="Target Left Edge"
)

plt.axvline(
    target_right_edge,
    linestyle="--",
    label="Target Right Edge"
)


# ---------------------------------------------------------
# Printed edges
# ---------------------------------------------------------

plt.axvline(
    printed_left_edge,
    linestyle=":",
    label="Printed Left Edge"
)

plt.axvline(
    printed_right_edge,
    linestyle=":",
    label="Printed Right Edge"
)


# ---------------------------------------------------------
# Mark threshold crossings
# ---------------------------------------------------------

plt.scatter(
    [
        printed_left_edge,
        printed_right_edge
    ],
    [
        threshold,
        threshold
    ],
    s=80
)


# ---------------------------------------------------------
# Text annotation
# ---------------------------------------------------------

plt.text(
    printed_left_edge,
    threshold + 0.08,
    f"EPE L = {EPE_left*1000:.2f} nm",
    ha="center"
)

plt.text(
    printed_right_edge,
    threshold + 0.08,
    f"EPE R = {EPE_right*1000:.2f} nm",
    ha="center"
)


plt.xlabel(
    "x (μm)"
)

plt.ylabel(
    "Normalized Intensity"
)

plt.title(
    "Target Edge vs Printed Edge"
)

plt.legend()

plt.grid()

plt.show()