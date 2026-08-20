
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
x=np.linspace(-1.0,1.0,201)
y=np.linspace(-1.0,1.0,201)
X,Y=np.meshgrid(x,y)
left_rect=((X>=-0.45)&(X<=-0.20)&(np.abs(Y)<=0.25))
right_rect=((X>=0.20)&(X<=0.45)&(np.abs(Y)<=0.25))
central=((X>=-0.2)&(X<=0.2)&(np.abs(Y)<=0.05))
mask_2d=np.where(left_rect|right_rect|central,1.0,0.0)
sigma=0.08
psf_2d=np.exp(-(X**2+Y**2)/(2*sigma**2))
psf_2d=psf_2d/np.sum(psf_2d)
aerial_2d=convolve2d(mask_2d,psf_2d,mode='same')
dose=0.5
threshold=0.10
printed_2d=(dose*aerial_2d>=threshold).astype(float)
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(mask_2d,extent=[x.min(),x.max(),y.min(),y.max()],origin='lower')
plt.title("2D mask")
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(aerial_2d,extent=[x.min(),x.max(),y.min(),y.max()],origin='lower')
plt.title("2D aerial image")
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(printed_2d,extent=[x.min(),x.max(),y.min(),y.max()],origin='lower')
plt.title("2D printed pattern")
plt.colorbar()
plt.tight_layout()
plt.show()