import sympy as sp
import numpy as np

# ========== 微积分部分 ==========
# 1.1 导数计算
x = sp.symbols('x')
f = x**2 + 3*x + 5  # 定义函数 f(x) = x^2 + 3*x + 5
f_prime = sp.diff(f, x)  # 计算导数
print(f"函数 f(x) = {f} 的导数为: f'(x) = {f_prime}")

# 1.2 定积分计算
integral_result = sp.integrate(f, (x, 0, 1))  # 计算定积分
print(f"函数 f(x) = {f} 在区间 [0, 1] 上的定积分结果是: {integral_result}")

# ========== 线性代数部分 ==========
# 2.1 矩阵运算
A = np.array([[1, 2], [3, 4]])  # 定义矩阵 A
B = np.array([[5, 6], [7, 8]])  # 定义矩阵 B

# 矩阵加法
matrix_sum = A + B
# 矩阵乘法
matrix_product = np.dot(A, B)
# 矩阵转置
matrix_transpose = np.transpose(A)

print(f"矩阵 A + B:\n{matrix_sum}")
print(f"矩阵 A * B:\n{matrix_product}")
print(f"矩阵 A 的转置:\n{matrix_transpose}")

# 2.2 特征值与特征向量计算
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"矩阵 A 的特征值为: {eigenvalues}")
print(f"矩阵 A 的特征向量为: \n{eigenvectors}")