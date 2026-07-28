import numpy as np


def poly_least_squares(x_list, y_list, k=3):
    """
    多项式最小二乘拟合

    参数:
        x_list: 自变量数据列表/数组
        y_list: 因变量数据列表/数组
        k: 拟合多项式的最高次数（默认3，即拟合到x³项）

    返回:
        coeffs: 系数数组 [a_k, a_{k-1}, ..., a_1, a_0]
                对应模型: y = a_k·x^k + a_{k-1}·x^{k-1} + ... + a_1·x + a_0

    原理:
        构造设计矩阵 X = [x^k, x^{k-1}, ..., x, 1]
        解正规方程: (XᵀX)·θ = XᵀY  →  θ = (XᵀX)⁻¹·XᵀY
    """
    x = np.array(x_list, dtype=float)
    y = np.array(y_list, dtype=float)
    n = len(x)

    # 构造设计矩阵 X: 每一列对应 x^k, x^{k-1}, ..., x, 1
    # 共 k+1 列，对应 k+1 个待求参数
    X = np.zeros((n, k + 1))
    for j in range(k + 1):
        power = k - j  # 从高次到低次: k, k-1, ..., 0
        X[:, j] = x ** power

    Y = y.reshape(-1, 1)

    # 解正规方程: θ* = (XᵀX)⁻¹ XᵀY
    # 用 np.linalg.solve 比直接求逆 (XᵀX)⁻¹ 数值更稳定
    XtX = X.T @ X
    XtY = X.T @ Y
    theta = np.linalg.solve(XtX, XtY)

    coeffs = theta.flatten()
    return coeffs


# ==================== 使用示例 ====================

x = [1800, 1900, 2100, 2200, 2300, 2400, 2500, 2600,
     2650, 2700, 2720, 2650, 2600, 2500, 2300, 2200,
     2000, 1850, 1820, 1800, 1750, 1500, 1000, 900]

y = [32, 60, 75, 85, 90, 98, 100, 102,
     108, 112, 115, 116, 118, 120, 118, 105,
     80, 60, 50, 30, 26, 20, 8, 5]

# 调用函数
coeffs = poly_least_squares(x, y, k=3)
print(coeffs)
# 输出: [-8.04750803e-08  4.55230960e-04 -7.35492252e-01  3.62275830e+02]
# 即: y = -8.05e-08·x³ + 4.55e-04·x² - 0.735·x + 362.28