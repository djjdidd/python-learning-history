import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize
from scipy.interpolate import interp1d
x = np.linspace(-2.0, 2.0, 2000)
dx = x[1] - x[0]

mask = np.where(np.abs(x) <= 0.25, 1.0, 0.0)

sigma = 0.12#sigma 变大表示 光学系统模糊更强 ，所以成像会 更模糊、更平滑
psf = np.exp(-x**2 / (2 * sigma**2))
psf = psf / np.sum(psf)

aerial_image = signal.fftconvolve(mask, psf, mode='same')

threshold = 0.40
def printed_pattern(dose):
    return (dose*aerial_image>=threshold).astype(float)
def width(binary):
    return np.sum(binary)*dx
target_width = 0.40
def objective(width):
    return (width-target_width)**2
dose_list=np.array([0.8,0.9,1.0,1.1,1.2])
width_list=[]
ob_list=[]
for i in dose_list:
    binary=printed_pattern(i)
    w=width(binary)
    ob=objective(w)
    width_list.append(w)
    ob_list.append(ob)
    print("dose=",i,"width=",w,"ob=",ob)
width_list=np.array(width_list)
ob_list=np.array(ob_list)


