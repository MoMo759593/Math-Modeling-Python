def lagrange_coefficients(x, y):
    """
    计算拉格朗日插值多项式的系数。

    给定 n 个数据点 (x_i, y_i)，构造唯一的 n-1 次插值多项式：
        p(t) = sum_{i=0}^{n-1} [ y_i * l_i(t) ]
    其中 l_i(t) = prod_{j!=i} (t - x_j) / (x_i - x_j) 为拉格朗日基函数。

    参数:
        x: list of float, 插值节点的横坐标，长度 n
        y: list of float, 插值节点的纵坐标，长度 n

    返回:
        list of float: 长度为 n 的系数列表，按降幂排列：
              coeffs[0] * t^(n-1) + coeffs[1] * t^(n-2) + ... + coeffs[n-1]
    """
    n = len(x)
    if len(y) != n:
        raise ValueError("x 和 y 的长度必须相等")

    def poly_mul(a, b):
        """两个多项式相乘，系数均按降幂排列"""
        res = [0.0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                res[i + j] += ai * bj
        return res

    # 最终系数，初始化为 0（n-1 次多项式共有 n 个系数）
    coeffs = [0.0] * n

    for i in range(n):
        # 计算第 i 个拉格朗日基函数的分母：prod_{j!=i} (x_i - x_j)
        denom = 1.0
        for j in range(n):
            if j != i:
                denom *= (x[i] - x[j])

        # 计算基函数的分子多项式：prod_{j!=i} (t - x_j)
        # 从常数 1 开始，逐个乘以 (t - x_j)
        poly = [1.0]
        for j in range(n):
            if j != i:
                # (t - x_j) 的降幂系数为 [1, -x_j]
                poly = poly_mul(poly, [1.0, -x[j]])

        # 将第 i 个基函数的系数乘以 y_i/denom，累加到总系数
        factor = y[i] / denom
        for k in range(n):
            coeffs[k] += factor * poly[k]

    return coeffs


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 示例 1：线性插值 p(t) = t
    x1 = [0, 1]
    y1 = [0, 1]
    c1 = lagrange_coefficients(x1, y1)
    print(f"示例1 系数: {c1}")  # [1.0, 0.0]  =>  1*t + 0

    # 示例 2：二次插值，三点恰好在抛物线 y = t^2 上
    x2 = [0, 1, 2]
    y2 = [0, 1, 4]
    c2 = lagrange_coefficients(x2, y2)
    print(f"示例2 系数: {c2}")  # [1.0, 0.0, 0.0]  =>  1*t^2 + 0*t + 0

    # 示例 3：一般三点插值
    x3 = [1, 2, 3]
    y3 = [2, 3, 5]
    c3 = lagrange_coefficients(x3, y3)
    print(f"示例3 系数: {c3}")


    # 验证：p(1)=2, p(2)=3, p(3)=5
    def eval_poly(coeffs, t):
        n = len(coeffs)
        return sum(coeffs[k] * t ** (n - 1 - k) for k in range(n))


    print(f"  验证 p(1)={eval_poly(c3, 1):.4f}, p(2)={eval_poly(c3, 2):.4f}, p(3)={eval_poly(c3, 3):.4f}")