import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

focus_data = np.array([-0.20, -0.10, 0.00, 0.10, 0.20, 0.30])
hole_data  = np.array([118.0, 111.0, 106.0, 108.0, 114.0, 123.0])
def linear_model(focus,a,b):
    return a*focus+b
def qua_model(focus,a,b,c):
    return a*focus**2+b*focus+c
params_lin,_=curve_fit(linear_model,focus_data,hole_data)
params_qua,_=curve_fit(qua_model,focus_data,hole_data)
focus_fine=np.linspace(-0.20,0.30,300)
hole_lin=linear_model(focus_fine,*params_lin)
hole_qua=qua_model(focus_fine,*params_qua)

plt.figure(figsize=(8,6))
plt.scatter(focus_data,hole_data,label='original data')
plt.plot(focus_fine,hole_lin,color='green',label='linear fit')
plt.plot(focus_fine,hole_qua,label='quadratic fit')
plt.xlabel('focus')
plt.ylabel('hole diameter')
plt.title("focus-hole image")
plt.legend()
plt.grid(True)
plt.show()