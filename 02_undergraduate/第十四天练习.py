import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
focus_points = np.array([-0.30, -0.15, 0.00, 0.15, 0.30])
dose_points  = np.array([0.85, 0.95, 1.05, 1.15, 1.25])

cd_matrix = np.array([
    [48, 45, 42, 40, 39],
    [46, 43, 39, 37, 36],
    [45, 41, 37, 35, 34],
    [46, 43, 39, 37, 36],
    [48, 45, 42, 40, 39]
])
inter=RegularGridInterpolator((focus_points,dose_points),cd_matrix,method='linear')
