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

period = 0.4          # um
line_width = 0.2      # um

wavelength = 0.193    # um
NA = 0.85

threshold = 0.5


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


# =========================================================
# 4. FFT of mask
# =========================================================

spectrum = np.fft.fft(
    mask
)


# =========================================================
# 5. Physical defocus pupil
#
# P(f,z) = A(f) * exp[-i*pi*lambda*z*f^2]
#
# wavelength: um
# defocus_um: um
# freq: 1/um
# =========================================================

def make_physical_defocus_pupil(
    freq,
    cutoff,
    wavelength,
    defocus_um
):

    aperture = (
        np.abs(freq)
        <=
        cutoff
    ).astype(float)

    phase = (
        -np.pi
        *
        wavelength
        *
        defocus_um
        *
        freq ** 2
    )

    pupil = (
        aperture
        *
        np.exp(
            1j * phase
        )
    )

    return pupil


# =========================================================
# 6. Best-focus reference intensity
# =========================================================

pupil_best_focus = make_physical_defocus_pupil(
    freq,
    cutoff,
    wavelength,
    defocus_um=0.0
)

field_best_focus = np.fft.ifft(
    spectrum
    *
    pupil_best_focus
)

intensity_best_focus_raw = (
    np.abs(field_best_focus) ** 2
)

reference_peak = np.max(
    intensity_best_focus_raw
)

print(
    "Reference peak =",
    reference_peak
)


# =========================================================
# 7. Imaging with physical defocus
#
# Important:
# all defocus conditions use the SAME reference_peak
# =========================================================

def imaging_with_physical_defocus(
    spectrum,
    freq,
    cutoff,
    wavelength,
    defocus_um,
    reference_peak
):

    pupil = make_physical_defocus_pupil(
        freq,
        cutoff,
        wavelength,
        defocus_um
    )

    filtered_spectrum = (
        spectrum
        *
        pupil
    )

    field = np.fft.ifft(
        filtered_spectrum
    )

    intensity_raw = (
        np.abs(field) ** 2
    )

    intensity = (
        intensity_raw
        /
        reference_peak
    )

    return intensity


# =========================================================
# 8. Edge interpolation
# =========================================================

def interpolate_edge(
    x1,
    x2,
    I1,
    I2,
    threshold
):

    if I2 == I1:

        return (
            x1 + x2
        ) / 2

    x_edge = (
        x1
        +
        (
            threshold - I1
        )
        /
        (
            I2 - I1
        )
        *
        (
            x2 - x1
        )
    )

    return x_edge


# =========================================================
# 9. CD and EPE measurement
# =========================================================

def measure_cd_epe(
    x,
    intensity,
    threshold,
    target_left_edge,
    target_right_edge
):

    resist_binary = (
        intensity
        >=
        threshold
    ).astype(float)

    transition = np.diff(
        resist_binary
    )

    left_edges = np.where(
        transition == 1
    )[0]

    right_edges = np.where(
        transition == -1
    )[0]

    center_index = np.argmin(
        np.abs(x)
    )

    left_candidates = (
        left_edges[
            left_edges
            <
            center_index
        ]
    )

    right_candidates = (
        right_edges[
            right_edges
            >
            center_index
        ]
    )

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

    x_left = interpolate_edge(
        x[left_index],
        x[left_index + 1],
        intensity[left_index],
        intensity[left_index + 1],
        threshold
    )

    x_right = interpolate_edge(
        x[right_index],
        x[right_index + 1],
        intensity[right_index],
        intensity[right_index + 1],
        threshold
    )

    printed_cd = (
        x_right
        -
        x_left
    )

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
# 10. Target geometry
# =========================================================

target_left_edge = (
    -line_width / 2
)

target_right_edge = (
    line_width / 2
)

target_cd = (
    line_width
)


# =========================================================
# 11. CD vs physical defocus
# =========================================================

defocus_um_scan = np.linspace(
    -0.30,
    0.30,
    61
)

cd_physical_scan = []


for defocus_um in defocus_um_scan:

    intensity = imaging_with_physical_defocus(
        spectrum,
        freq,
        cutoff,
        wavelength,
        defocus_um,
        reference_peak
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

    cd_physical_scan.append(
        printed_cd
    )


cd_physical_scan = np.array(
    cd_physical_scan
)


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    defocus_um_scan * 1000,
    cd_physical_scan * 1000,
    marker="o"
)

plt.axvline(
    0,
    linestyle="--",
    label="Best Focus"
)

plt.xlabel(
    "Defocus (nm)"
)

plt.ylabel(
    "Printed CD (nm)"
)

plt.title(
    "CD vs Physical Defocus"
)

plt.grid()

plt.legend()

plt.show()


# =========================================================
# 12. Imaging with physical defocus + dose
# =========================================================

def imaging_with_physical_defocus_and_dose(
    spectrum,
    freq,
    cutoff,
    wavelength,
    defocus_um,
    reference_peak,
    dose
):

    intensity = imaging_with_physical_defocus(
        spectrum,
        freq,
        cutoff,
        wavelength,
        defocus_um,
        reference_peak
    )

    effective_intensity = (
        dose
        *
        intensity
    )

    return effective_intensity


# =========================================================
# 13. Dose calibration at best focus
# =========================================================

dose_scan = np.linspace(
    0.5,
    4.0,
    141
)

cd_vs_dose = []


for dose in dose_scan:

    intensity = (
        imaging_with_physical_defocus_and_dose(
            spectrum,
            freq,
            cutoff,
            wavelength,
            defocus_um=0.0,
            reference_peak=reference_peak,
            dose=dose
        )
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
    "Dose Calibration"
)

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


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    dose_scan,
    cd_vs_dose * 1000
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

plt.grid()

plt.legend()

plt.show()


# =========================================================
# 14. Physical Bossung plot
# =========================================================

dose_list = [
    0.9 * nominal_dose,
    nominal_dose,
    1.1 * nominal_dose
]


defocus_um_bossung = np.linspace(
    -0.30,
    0.30,
    61
)


plt.figure(
    figsize=(9, 6)
)


for dose in dose_list:

    cd_curve = []

    for defocus_um in defocus_um_bossung:

        intensity = (
            imaging_with_physical_defocus_and_dose(
                spectrum,
                freq,
                cutoff,
                wavelength,
                defocus_um,
                reference_peak,
                dose
            )
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

        cd_curve.append(
            printed_cd
        )

    cd_curve = np.array(
        cd_curve
    )

    plt.plot(
        defocus_um_bossung * 1000,
        cd_curve * 1000,
        marker="o",
        label=f"Dose = {dose:.3f}"
    )


plt.axhline(
    target_cd * 1000,
    linestyle="--",
    label="Target CD"
)


plt.xlabel(
    "Defocus (nm)"
)

plt.ylabel(
    "Printed CD (nm)"
)

plt.title(
    "Physical Bossung Plot"
)

plt.grid()

plt.legend()

plt.show()


# =========================================================
# 15. Process window settings
# =========================================================

target_cd_nm = (
    target_cd
    *
    1000
)

cd_tolerance_nm = 5.0

cd_lower_nm = (
    target_cd_nm
    -
    cd_tolerance_nm
)

cd_upper_nm = (
    target_cd_nm
    +
    cd_tolerance_nm
)


print()
print("=" * 60)

print(
    "Process Window"
)

print("=" * 60)

print(
    "CD specification =",
    cd_lower_nm,
    "to",
    cd_upper_nm,
    "nm"
)


# =========================================================
# 16. Process-window scan
# =========================================================

dose_scan_pw = np.linspace(
    0.85 * nominal_dose,
    1.15 * nominal_dose,
    61
)


defocus_um_pw = np.linspace(
    -0.80,
    0.80,
    121
)


cd_map = np.full(
    (
        len(defocus_um_pw),
        len(dose_scan_pw)
    ),
    np.nan
)


# =========================================================
# 17. Calculate CD map
# =========================================================

for i, defocus_um in enumerate(
    defocus_um_pw
):

    for j, dose in enumerate(
        dose_scan_pw
    ):

        intensity = (
            imaging_with_physical_defocus_and_dose(
                spectrum,
                freq,
                cutoff,
                wavelength,
                defocus_um,
                reference_peak,
                dose
            )
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

        cd_map[
            i,
            j
        ] = (
            printed_cd
            *
            1000
        )


# =========================================================
# 18. Pass / Fail map
# =========================================================

pass_map = (
    (cd_map >= cd_lower_nm)
    &
    (cd_map <= cd_upper_nm)
)


# =========================================================
# 19. Plot CD map
# =========================================================

plt.figure(
    figsize=(9, 6)
)

plt.imshow(
    cd_map,
    extent=[
        dose_scan_pw.min(),
        dose_scan_pw.max(),
        defocus_um_pw.min() * 1000,
        defocus_um_pw.max() * 1000
    ],
    origin="lower",
    aspect="auto"
)

plt.colorbar(
    label="Printed CD (nm)"
)

plt.xlabel(
    "Dose"
)

plt.ylabel(
    "Defocus (nm)"
)

plt.title(
    "CD Map"
)

plt.show()


# =========================================================
# 20. Plot Process Window
# =========================================================

plt.figure(
    figsize=(9, 6)
)

plt.imshow(
    pass_map.astype(float),
    extent=[
        dose_scan_pw.min(),
        dose_scan_pw.max(),
        defocus_um_pw.min() * 1000,
        defocus_um_pw.max() * 1000
    ],
    origin="lower",
    aspect="auto",
    vmin=0,
    vmax=1
)

plt.xlabel(
    "Dose"
)

plt.ylabel(
    "Defocus (nm)"
)

plt.title(
    "Physical Process Window"
)

plt.show()


# =========================================================
# 21. Exposure Latitude
# =========================================================

best_focus_index = np.argmin(
    np.abs(
        defocus_um_pw
    )
)

pass_at_best_focus = (
    pass_map[
        best_focus_index,
        :
    ]
)

valid_doses = (
    dose_scan_pw[
        pass_at_best_focus
    ]
)


if len(valid_doses) > 0:

    dose_min = (
        valid_doses.min()
    )

    dose_max = (
        valid_doses.max()
    )

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
    print(
        "Exposure Latitude"
    )

    print(
        "------------------"
    )

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

else:

    print(
        "No valid dose range at best focus."
    )


# =========================================================
# 22. Depth of Focus
# =========================================================

nominal_dose_index = np.argmin(
    np.abs(
        dose_scan_pw
        -
        nominal_dose
    )
)

pass_at_nominal_dose = (
    pass_map[
        :,
        nominal_dose_index
    ]
)

valid_defocus_um = (
    defocus_um_pw[
        pass_at_nominal_dose
    ]
)


if len(valid_defocus_um) > 0:

    defocus_min_um = (
        valid_defocus_um.min()
    )

    defocus_max_um = (
        valid_defocus_um.max()
    )

    dof_um = (
        defocus_max_um
        -
        defocus_min_um
    )

    print()
    print(
        "Depth of Focus"
    )

    print(
        "--------------"
    )

    print(
        "Defocus range =",
        defocus_min_um * 1000,
        "to",
        defocus_max_um * 1000,
        "nm"
    )

    print(
        "DOF =",
        dof_um * 1000,
        "nm"
    )

else:

    print(
        "No valid defocus range at nominal dose."
    )