import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. Basic parameters
# =========================================================

period = 0.8        # μm
wavelength = 0.193 # μm
NA = 0.5

cutoff = NA / wavelength

print("Fundamental frequency =", 1 / period, "1/μm")
print("Cutoff frequency =", cutoff, "1/μm")


# =========================================================
# 2. Diffraction orders
# =========================================================

m_orders = np.arange(-4, 5)

diffraction_freq = (
    m_orders / period
)


# =========================================================
# 3. Two illumination conditions
# =========================================================

source_normal = 0.0

source_shifted = 1.5


# =========================================================
# 4. Pupil coverage range
# =========================================================

normal_left = (
    source_normal - cutoff
)

normal_right = (
    source_normal + cutoff
)


shifted_left = (
    source_shifted - cutoff
)

shifted_right = (
    source_shifted + cutoff
)


print()
print("Normal pupil coverage:")
print(
    normal_left,
    "to",
    normal_right
)

print()

print("Shifted pupil coverage:")
print(
    shifted_left,
    "to",
    shifted_right
)


# =========================================================
# 5. Check which orders pass
# =========================================================

pass_normal = (
    (diffraction_freq >= normal_left)
    &
    (diffraction_freq <= normal_right)
)

pass_shifted = (
    (diffraction_freq >= shifted_left)
    &
    (diffraction_freq <= shifted_right)
)


print()
print("Diffraction orders:")

for m, f, pn, ps in zip(
    m_orders,
    diffraction_freq,
    pass_normal,
    pass_shifted
):

    print(
        f"m={m:2d}, "
        f"f={f:6.2f}, "
        f"normal={pn}, "
        f"shifted={ps}"
    )


# =========================================================
# 6. Plot diffraction orders
# =========================================================

plt.figure(
    figsize=(12, 5)
)


# diffraction order stems
for m, f in zip(
    m_orders,
    diffraction_freq
):

    plt.vlines(
        f,
        0,
        1
    )

    plt.text(
        f,
        1.03,
        f"m={m}",
        ha="center"
    )


# =========================================================
# 7. Normal pupil range
# =========================================================

plt.axvspan(
    normal_left,
    normal_right,
    alpha=0.2,
    label="Normal illumination pupil"
)


# =========================================================
# 8. Shifted pupil range
# =========================================================

plt.axvspan(
    shifted_left,
    shifted_right,
    alpha=0.2,
    label="Shifted source pupil"
)


# =========================================================
# 9. Formatting
# =========================================================

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "Spatial Frequency (1/μm)"
)

plt.ylabel(
    "Relative Order Strength"
)

plt.title(
    "Diffraction Orders and Pupil Coverage"
)

plt.ylim(
    0,
    1.2
)

plt.grid()

plt.legend()

plt.tight_layout()

plt.show()