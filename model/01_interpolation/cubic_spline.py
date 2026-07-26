def cubic_spline_interpolation(x_list, y_list, bc_type='natural',
                               left_slope=None, right_slope=None):
    """
    三次样条插值（三弯矩法）。

    参数
    ----
    x_list : list of float
        严格递增的节点序列，长度 >= 2
    y_list : list of float
        对应节点处的函数值，长度与 x_list 相等
    bc_type : str, 默认 'natural'
        边界条件类型：
        - 'natural' : 自然边界，S''(x_0) = S''(x_n) = 0
        - 'clamped' : 固定边界，需同时提供 left_slope 和 right_slope
                      分别表示 S'(x_0) 和 S'(x_n)
    left_slope : float, optional
        bc_type='clamped' 时，左端点一阶导数
    right_slope : float, optional
        bc_type='clamped' 时，右端点一阶导数

    返回
    ----
    s : callable
        接受标量或列表/元组，返回三次样条在该点的取值。
        若 x 超出 [x_0, x_n] 范围，抛出 ValueError。
    """
    n_nodes = len(x_list)
    if len(y_list) != n_nodes:
        raise ValueError("x_list 和 y_list 长度必须相等")
    if n_nodes < 2:
        raise ValueError("至少需要 2 个节点")

    n = n_nodes - 1  # 区间数

    # 检查严格递增
    for i in range(n):
        if x_list[i + 1] <= x_list[i]:
            raise ValueError("x_list 必须严格递增")

    # 步长和函数值差分
    h = [x_list[i + 1] - x_list[i] for i in range(n)]
    dy = [y_list[i + 1] - y_list[i] for i in range(n)]

    # ---------- 1. 计算内部节点的 μ, λ, d ----------
    # mu[i], lam[i], d_vec[i] 对应 i = 1, 2, ..., n-1
    mu = [0.0] * n_nodes
    lam = [0.0] * n_nodes
    d_vec = [0.0] * n_nodes

    for i in range(1, n):
        mu[i] = h[i - 1] / (h[i - 1] + h[i])
        lam[i] = h[i] / (h[i - 1] + h[i])
        d_vec[i] = 6.0 / (h[i - 1] + h[i]) * (dy[i] / h[i] - dy[i - 1] / h[i - 1])

    # ---------- 2. 根据边界条件组装三对角方程组并求解 M ----------
    def _thomas(a, b, c, d):
        """追赶法（Thomas 算法）解三对角方程组，长度均为 m"""
        m = len(d)
        cp = [0.0] * m
        dp = [0.0] * m

        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]

        for i in range(1, m):
            denom = b[i] - a[i] * cp[i - 1]
            if i < m - 1:
                cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

        x = [0.0] * m
        x[m - 1] = dp[m - 1]
        for i in range(m - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x

    if bc_type == 'natural':
        # M_0 = 0, M_n = 0，解内部 M_1..M_{n-1}
        if n == 1:
            # 只有两个节点，自然边界直接令 M = [0, 0]
            M = [0.0, 0.0]
        else:
            m = n - 1  # 未知数个数
            a_tri = [0.0] * m
            b_tri = [0.0] * m
            c_tri = [0.0] * m
            d_tri = [0.0] * m

            # i = 1 对应 k = 0
            b_tri[0] = 2.0
            c_tri[0] = lam[1]
            d_tri[0] = d_vec[1]

            for k in range(1, m - 1):
                i = k + 1
                a_tri[k] = mu[i]
                b_tri[k] = 2.0
                c_tri[k] = lam[i]
                d_tri[k] = d_vec[i]

            # i = n-1 对应 k = m-1
            a_tri[m - 1] = mu[n - 1]
            b_tri[m - 1] = 2.0
            c_tri[m - 1] = 0.0
            d_tri[m - 1] = d_vec[n - 1]

            M_inner = _thomas(a_tri, b_tri, c_tri, d_tri)
            M = [0.0] + M_inner + [0.0]

    elif bc_type == 'clamped':
        if left_slope is None or right_slope is None:
            raise ValueError("clamped 边界必须提供 left_slope 和 right_slope")

        m = n + 1  # 未知数 M_0..M_n
        a_tri = [0.0] * m
        b_tri = [0.0] * m
        c_tri = [0.0] * m
        d_tri = [0.0] * m

        # 方程 0: 2*M_0 + M_1 = 6/h_0 * ((y_1-y_0)/h_0 - left_slope)
        b_tri[0] = 2.0
        c_tri[0] = 1.0
        d_tri[0] = 6.0 / h[0] * (dy[0] / h[0] - left_slope)

        # 方程 1..n-1（内部节点）
        for i in range(1, n):
            a_tri[i] = mu[i]
            b_tri[i] = 2.0
            c_tri[i] = lam[i]
            d_tri[i] = d_vec[i]

        # 方程 n: M_{n-1} + 2*M_n = 6/h_{n-1} * (right_slope - (y_n-y_{n-1})/h_{n-1})
        a_tri[n] = 1.0
        b_tri[n] = 2.0
        c_tri[n] = 0.0
        d_tri[n] = 6.0 / h[n - 1] * (right_slope - dy[n - 1] / h[n - 1])

        M = _thomas(a_tri, b_tri, c_tri, d_tri)

    else:
        raise ValueError("bc_type 必须是 'natural' 或 'clamped'")

    # ---------- 3. 计算每段系数 ----------
    # S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
    a_coef = y_list[:n]  # a_i = y_i
    b_coef = [0.0] * n
    c_coef = [0.0] * n
    d_coef = [0.0] * n

    for i in range(n):
        c_coef[i] = M[i] / 2.0
        d_coef[i] = (M[i + 1] - M[i]) / (6.0 * h[i])
        b_coef[i] = dy[i] / h[i] - h[i] * (2.0 * M[i] + M[i + 1]) / 6.0

    # ---------- 4. 求值函数 ----------
    def _find_interval(xv):
        """二分查找 xv 所在区间 [x_i, x_{i+1}]"""
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if xv < x_list[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def s(x):
        is_seq = isinstance(x, (list, tuple))
        x_vals = list(x) if is_seq else [x]
        res = []

        for xv in x_vals:
            if xv < x_list[0] or xv > x_list[-1]:
                raise ValueError(
                    f"x={xv} 超出插值区间 [{x_list[0]}, {x_list[-1]}]"
                )
            i = _find_interval(xv)
            dx = xv - x_list[i]
            # 秦九韶式求值：y_i + dx*(b_i + dx*(c_i + dx*d_i))
            res.append(
                a_coef[i] + dx * (b_coef[i] + dx * (c_coef[i] + dx * d_coef[i]))
            )

        return res if is_seq else res[0]

    return s


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 例 1：自然边界，4 个节点在 y = x^2 上
    # 理论上三次样条应精确重构二次函数
    x = [0, 1, 2, 3]
    y = [0, 1, 4, 9]
    s = cubic_spline_interpolation(x, y, bc_type='natural')

    print("--- 例 1: 自然边界，y = x^2 ---")
    print(f"s(0)   = {s(0)}")  # 0
    print(f"s(1)   = {s(1)}")  # 1
    print(f"s(2)   = {s(2)}")  # 4
    print(f"s(3)   = {s(3)}")  # 9
    print(f"s(1.5) = {s(1.5)}")  # 2.25
    print(f"s(0.5) = {s(0.5)}")  # 0.25
    print(f"批量求值: {s([0, 0.5, 1, 1.5, 2, 2.5, 3])}")

    # 例 2：固定边界，指定端点斜率
    # y = x^2 的导数为 2x，在 x=0 处斜率为 0，x=3 处斜率为 6
    s2 = cubic_spline_interpolation(x, y, bc_type='clamped',
                                    left_slope=0, right_slope=6)
    print("\n--- 例 2: 固定边界（left=0, right=6）---")
    print(f"s2(0)   = {s2(0)}")
    print(f"s2(1.5) = {s2(1.5)}")
    print(f"s2(3)   = {s2(3)}")

    # 例 3：非均匀节点 + 正弦函数
    import math

    x3 = [0, math.pi / 6, math.pi / 3, math.pi / 2]
    y3 = [math.sin(v) for v in x3]
    s3 = cubic_spline_interpolation(x3, y3, bc_type='natural')
    print("\n--- 例 3: sin(x) 自然边界 ---")
    print(f"s3(0)       = {s3(0):.6f}  (真值 0.000000)")
    print(f"s3(pi/12)   = {s3(math.pi / 12):.6f}  (真值 {math.sin(math.pi / 12):.6f})")
    print(f"s3(pi/4)    = {s3(math.pi / 4):.6f}  (真值 {math.sin(math.pi / 4):.6f})")
    print(f"s3(pi/2)    = {s3(math.pi / 2):.6f}  (真值 1.000000)")