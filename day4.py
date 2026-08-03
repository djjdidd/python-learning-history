#剂量如何改变线宽——打印宽度测量与 dose 分析
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
x=np.linspace(-2.0,2.0,2000)
dx=x[1]-x[0]
mask=np.where(np.abs(x)<=0.25,1.0,0.0)
sigma=0.08
psf=np.exp(-x**2/(2*sigma**2))
psf=psf/np.sum(psf)
aerial_image=signal.fftconvolve(mask,psf,mode='same')
threshold=0.30
dose_list=[0.8,1.0,1.2]
width_list=[]
plt.figure(figsize=(10,8))
for i,dose in enumerate(dose_list):
    image_with_dose=dose*aerial_image
    printed_pattern=np.where(image_with_dose>=threshold,1.0,0.0)
    width=np.sum(printed_pattern)*dx
    width_list.append(width)
    plt.subplot(3,1,i+1)
    plt.plot(x, image_with_dose, label=f'dose = {dose}')
    plt.plot(x, printed_pattern, label='printed pattern')
    plt.axhline(threshold, color='red', linestyle='--', label='threshold')
    plt.title(f'Dose = {dose}, Printed Width = {width:.4f}')
    plt.xlabel("Position")
    plt.ylabel("value")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(6,4))
plt.plot(dose_list,width_list,marker='o')
plt.xlabel=("dose")
plt.ylabel=("printed width")
plt.grid(True)
plt.show()
print("不同剂量下的打印线宽：")
for dose,width in zip(dose_list,width_list):
    print(f"dose={dose},width={width}")