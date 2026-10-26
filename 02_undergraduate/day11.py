#曲线拟合入门——从离散数据到参数化模型
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
dose_data=np.array([0.80,0.90,1.00,1.10,1.20])
cd_data=np.array([42.0,39.0,36.0,34.0,33.0])
def linear_model(x,a,b):
    return a*x+b
def quadratic_model(x,a,b,c):
    return a*x**2+b*x+c
params_lin,_=curve_fit(linear_model,dose_data,cd_data)
params_quad,_=curve_fit(quadratic_model,dose_data,cd_data)
cd_fit_lin=linear_model(dose_data,*params_lin)
cd_fit_quad=quadratic_model(dose_data,*params_quad)
dose_fine=np.linspace(0.8,1.2,200)
cd_fine_lin=linear_model(dose_fine,*params_lin)
cd_fine_quad=quadratic_model(dose_fine,*params_quad)
print("线性模型参数：",params_lin)
print("二次模型参数：",params_quad)
plt.figure(figsize=(8,5))
plt.scatter(dose_data,cd_data,label='data')
plt.plot(dose_fine,cd_fine_lin,label='linear fit')
plt.plot(dose_fine,cd_fine_quad,label='quadratic fit')
plt.xlabel("dose")
plt.ylabel("cd")
plt.title("dose-cd fitting")
plt.legend()
plt.grid(True)
plt.show()