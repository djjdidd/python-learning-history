 #最小化一个函数
from scipy.optimize import minimize
import numpy as np
 
def objective(x):
    return x[0]**2 + x[1]**2 # 最小化 f(x) = x1² + x2²
 
result = minimize(objective, x0=[1, 1], method='BFGS')
print("最优解:", result.x)
print("最小值:", result.fun)

#曲线拟合
from scipy.optimize import curve_fit
# 假设有实验数据 y ≈ a * exp(-b * x) + c
x_data = np.linspace(0, 4, 500)
y_true = 2.5 * np.exp(-1.3 * x_data) + 0.5
y_noise = y_true + 0.2 * np.random.normal(size=x_data.size)
# 定义模型函数
def model(x, a, b, c):
    return a * np.exp(-b * x) + c
# 拟合参数
params, cov = curve_fit(model, x_data, y_noise)
print("拟合参数 a, b, c:", params)

#计算定积分
from scipy.integrate import quad
result, error = quad(lambda x: x**2, 0, 1)
print("积分结果:", result)  # 应为 1/3 ≈ 0.333...

#求常微分方程
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# 定义 ODE：dy/dt = v, dv/dt = -k/m * y - c/m * v
def mass_spring_damper(t, z, m=1, k=1, c=0.3):
    y, v = z
    dydt = v
    dvdt = -(k/m)*y - (c/m)*v
    return [dydt, dvdt]
# 初始条件：位置=1，速度=0
sol = solve_ivp(mass_spring_damper, [0, 20], [1, 0], t_eval=np.linspace(0, 20, 200))
plt.plot(sol.t, sol.y[0], label='Position')
plt.plot(sol.t, sol.y[1], label='Velocity')
plt.legend()
plt.title("Damped Oscillator")
plt.show()

#快速傅里叶变换
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
# 生成含噪声的正弦信号
t = np.linspace(0, 1, 500)
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
signal += 0.2 * np.random.normal(size=t.shape)
# FFT 变换
Y = fft(signal)
freq = np.fft.fftfreq(t.shape[-1], d=t[1]-t[0])
plt.plot(freq[:250], np.abs(Y)[:250])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Frequency Spectrum')
plt.show()