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
dose_list=np.array([0.6,0.7,0.8,0.9,1.0,1.1,1.2])
width_list=[]
ob_list=[]
w1=1
dose_index=0
for i,dose in enumerate(dose_list):
    binary=printed_pattern(dose)
    w=width(binary)
    ob=objective(w)
    width_list.append(w)
    ob_list.append(ob)
    print("dose=",i,"width=",w,"ob=",ob)
    if objective(w1)<objective(w):
        w1=w1
    else:
        w1=w
        dose_index=i
print("best width=",w1)   
print("best dose=",dose_list[dose_index])
print("best ob=",objective(w1))
width_list=np.array(width_list)
ob_list=np.array(ob_list)
plt.figure(figsize=(12,5))
plt.scatter(dose_list,width_list,label="width")
plt.plot(dose_list,width_list,label="width")
plt.xlabel("dose")
plt.ylabel("width")
plt.title("width vs dose")
plt.grid(True)
plt.legend()
plt.show()
