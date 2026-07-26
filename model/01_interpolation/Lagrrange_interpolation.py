def lagrange_interpolation(x_list, y_list):
    """
    拉格朗日插值。输入节点序列，返回一个可调用函数 p(x)。

    参数
    ----
    x_list : list of float
        互异的插值节点，长度 n
    y_list : list of float
        对应节点处的函数值，长度 n

    返回
    ----
    p : callable
        接受标量或列表/数组，返回拉格朗日插值多项式在该点的取值
    """
    n = len(x_list)
    if len(y_list) != n:
        raise ValueError("x_list 和 y_list 长度必须相等")
    if n == 0:
        raise ValueError("输入不能为空列表")

    # 预计算每个基函数的分母：denom[i] = prod_{j!=i} (x_i - x_j)
    denom = []
    for i in range(n):
        d = 1.0
        for j in range(n):
            if j != i:
                diff = x_list[i] - x_list[j]
                if abs(diff) < 1e-15:
                    raise ValueError(f"节点重复: x[{i}] = x[{j}] = {x_list[i]}")
                d *= diff
        denom.append(d)

    def p(x):
        """
        拉格朗日插值多项式求值。
        支持标量、list、tuple。
        """
        is_sequence = isinstance(x, (list, tuple))
        if is_sequence:
            x_vals = list(x)
            result = [0.0] * len(x_vals)
            for idx, xv in enumerate(x_vals):
                s = 0.0
                for i in range(n):
                    # 计算基函数 l_i(xv) 的分子：prod_{j!=i} (xv - x_j)
                    num = 1.0
                    for j in range(n):
                        if j != i:
                            num *= (xv - x_list[j])
                    s += y_list[i] * num / denom[i]
                result[idx] = s
            return result
        else:
            s = 0.0
            for i in range(n):
                num = 1.0
                for j in range(n):
                    if j != i:
                        num *= (x - x_list[j])
                s += y_list[i] * num / denom[i]
            return s

    return p


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 例 1：三点在抛物线 y = x^2 上
    x = [0, 1, 2]
    y = [0, 1, 4]
    p = lagrange_interpolation(x, y)

    print("--- 例 1: y = x^2 ---")
    print(f"p(0)   = {p(0)}")  # 0
    print(f"p(1)   = {p(1)}")  # 1
    print(f"p(2)   = {p(2)}")  # 4
    print(f"p(1.5) = {p(1.5)}")  # 2.25
    print(f"p([0, 0.5, 1, 1.5, 2]) = {p([0, 0.5, 1, 1.5, 2])}")

    # 例 2：线性插值
    x2, y2 = [1, 3], [2, 6]
    p2 = lagrange_interpolation(x2, y2)
    print("\n--- 例 2: 线性 ---")
    print(f"p2(1) = {p2(1)}, p2(2) = {p2(2)}, p2(3) = {p2(3)}")

    # 例 3：正弦函数采样
    import math

    x3 = [0, math.pi / 4, math.pi / 2]
    y3 = [math.sin(v) for v in x3]
    p3 = lagrange_interpolation(x3, y3)
    print("\n--- 例 3: sin(x) 三点插值 ---")
    print(f"p3(0)       = {p3(0):.6f}  (真值 0.000000)")
    print(f"p3(pi/6)    = {p3(math.pi / 6):.6f}  (真值 0.500000)")
    print(f"p3(pi/4)    = {p3(math.pi / 4):.6f}  (真值 0.707107)")
    print(f"p3(pi/2)    = {p3(math.pi / 2):.6f}  (真值 1.000000)")