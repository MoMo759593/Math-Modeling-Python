from scipy import stats

def f_test_regression(x_list, y_list, beta0, beta1, alpha=0.05):
    """
    一元线性回归方程的显著性 F 检验
    H0: beta1 = 0（x 与 y 之间不存在线性关系）

    参数:
        x_list, y_list : 观测数据（等长实数列表）
        beta0, beta1   : 已由最小二乘法算出的回归系数
        alpha          : 显著性水平，默认 0.05
    返回:
        包含各平方和、自由度、F 统计量、p 值、临界值及结论的字典
    """
    n = len(x_list)
    if n != len(y_list):
        raise ValueError("x_list 与 y_list 长度必须一致")
    if n < 3:
        raise ValueError("样本量至少为 3，否则残差自由度为 0")

    y_mean = sum(y_list) / n
    y_hat = [beta0 + beta1 * x for x in x_list]

    S_T = sum((y - y_mean) ** 2 for y in y_list)              # 总平方和
    S_R = sum((yh - y_mean) ** 2 for yh in y_hat)             # 回归平方和
    S_E = sum((y - yh) ** 2 for y, yh in zip(y_list, y_hat))  # 残差平方和

    df_R, df_E = 1, n - 2                                     # 自由度
    F = (S_R / df_R) / (S_E / df_E)
    p_value = 1 - stats.f.cdf(F, df_R, df_E)                  # 右尾 p 值
    F_crit = stats.f.ppf(1 - alpha, df_R, df_E)               # 临界值

    return {
        "S_T": S_T, "S_R": S_R, "S_E": S_E,
        "df_R": df_R, "df_E": df_E,
        "F": F,
        "p_value": p_value,
        "alpha": alpha,
        "F_critical": F_crit,
        "reject_H0": F > F_crit,
        "conclusion": ("拒绝 H0：回归方程显著，x 对 y 有显著线性影响"
                       if F > F_crit else
                       "不拒绝 H0：回归方程不显著，线性关系不成立")
    }

x = [1, 2, 3, 4, 5]
y = [2.1, 4.2, 5.9, 8.1, 9.8]

# 先用最小二乘法估计参数
x_m, y_m = sum(x)/len(x), sum(y)/len(y)
b1 = sum((xi-x_m)*(yi-y_m) for xi, yi in zip(x, y)) / sum((xi-x_m)**2 for xi in x)
b0 = y_m - b1*x_m

result = f_test_regression(x, y, b0, b1)
print(result)