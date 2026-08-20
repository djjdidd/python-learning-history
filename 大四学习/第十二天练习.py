import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

focus_data = np.array([-0.20, -0.10, 0.00, 0.10, 0.20, 0.30])
cd_data    = np.array([128.0, 120.0, 115.0, 117.0, 123.0, 132.0])
def linear_model(dose,a,b):
    return a*dose+b
def qua_model(dose,a,b,c):
    return a*dose**2+b*dose+c
def resi_lin(params,x,y):
    a,b=params
    y_pred=linear_model(x,a,b)
    return y_pred-y
def resi_qua(params,x,y):
    a,b,c=params
    y_pred=qua_model(x,a,b,c)
    return y_pred-y
initial_lin=([1.0,1.0])
initial_qua=([1.0,1.0,1.0])
result_lin=least_squares(resi_lin,x0=initial_lin,args=(focus_data,cd_data))
result_qua=least_squares(resi_qua,x0=initial_qua,args=(focus_data,cd_data))
a1,b1=result_lin.x
a2,b2,c2=result_qua.x
print("一次函数残差拟合参数为：")
print(f"a={a1:.4f}")
print(f"b={b1:.4f}")
print("二次函数残差拟合参数为：")
print(f"a={a2:.4f}")
print(f"b={b2:.4f}")
print(f"c={c2:.4f}")
focus_fine = np.linspace(-0.20, 0.30, 300)
cd_fit1 = linear_model(focus_fine, a1, b1)
cd_fit2=qua_model(focus_fine,a2,b2,c2)
cd_data1=linear_model(focus_data,a1,b1)
cd_data2=qua_model(focus_data,a2,b2,c2)
def rmse(y_true, y_pred):
    return  np.sqrt(np.mean((y_pred-y_true)**2))

rmse_linear = rmse(cd_data,cd_data1)
rmse_quadratic = rmse(cd_data,cd_data2)

print("线性模型 RMSE =", rmse_linear)
print("二次模型 RMSE =", rmse_quadratic)

plt.figure(figsize=(8, 5))
plt.scatter(focus_data, cd_data, label='original data')
plt.plot(focus_fine, cd_fit1, label='linear least_squares fit')
plt.plot(focus_fine, cd_fit2, label='quadratic least_squares fit')
plt.xlabel("focus")
plt.ylabel("CD")
plt.title("focus-CD Fitting by least_squares")
plt.grid(True)
plt.legend()
plt.show()