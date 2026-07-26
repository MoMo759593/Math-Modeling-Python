def quadratic_spline_interpolation(x_list, y_list, left_slope=None):
    """
    二次样条插值。

    参数
    ----
    x_list : list of float
        严格递增的节点序列，长度 >= 2
    y_list : list of float
        对应节点处的函数值，长度与 x_list 相等
    left_slope : float, optional
        左端点 x_0 处的一阶导数 S'(x_0)。
        若为 None（默认），则令第一段为直线（c_0 = 0）。

    返回
    ----
    s : callable
        接受标量或列表/元组，返回二次样条在该点的取值。
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

    # 步长
    h = [x_list[i + 1] - x_list[i] for i in range(n)]
    # 函数值差分
    dy = [y_list[i + 1] - y_list[i] for i in range(n)]

    # 每段 S_i(x) = y_i + b_i*(x-x_i) + c_i*(x-x_i)^2
    b = [0.0] * n
    c = [0.0] * n

    # ========== 边界条件（补全缺的 1 个方程）==========
    if left_slope is None:
        # 默认：第一段退化为直线 => c_0 = 0
        # 由 S_0(x_1) = y_1 :  y_0 + b_0*h_0 = y_1  =>  b_0 = dy_0 / h_0
        b[0] = dy[0] / h[0]
    else:
        # 用户给定左端点导数
        b[0] = float(left_slope)

    # ========== 递推求解所有系数 ==========
    # 对 i = 0, 1, ..., n-2：
    #   (1) 函数值连续:  c_i = (dy_i/h_i - b_i) / h_i
    #   (2) 一阶导连续:  b_{i+1} = b_i + 2*c_i*h_i
    for i in range(n - 1):
        c[i] = (dy[i] / h[i] - b[i]) / h[i]
        b[i + 1] = b[i] + 2 * c[i] * h[i]

    # 最后一段的 c_{n-1} 直接由函数值条件算出
    c[n - 1] = (dy[n - 1] / h[n - 1] - b[n - 1]) / h[n - 1]

    # ========== 求值函数 ==========
    def _find_interval(xv):
        """二分查找 xv 落在哪个区间 [x_i, x_{i+1}]"""
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
            # S_i(x) = y_i + b_i*dx + c_i*dx^2
            res.append(y_list[i] + b[i] * dx + c[i] * dx * dx)

        return res if is_seq else res[0]

    return s


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 例 1：4 个节点，恰好在抛物线 y = x^2 上
    x = [0, 1, 2, 3]
    y = [0, 1, 4, 9]

    # 默认边界：第一段为直线
    s = quadratic_spline_interpolation(x, y)

    print("--- 例 1: 默认边界（第一段为直线）---")
    print(f"s(0)   = {s(0)}")  # 精确过点: 0
    print(f"s(1)   = {s(1)}")  # 精确过点: 1
    print(f"s(2)   = {s(2)}")  # 精确过点: 4
    print(f"s(3)   = {s(3)}")  # 精确过点: 9
    print(f"s(1.5) = {s(1.5)}")  # 插值
    print(f"s(0.5) = {s(0.5)}")  # 第一段是直线，所以 s(0.5)=0.5
    print(f"批量求值: {s([0, 0.5, 1, 1.5, 2, 2.5, 3])}")

    # 例 2：指定左端点斜率为 0
    s2 = quadratic_spline_interpolation(x, y, left_slope=0)
    print("\n--- 例 2: 左端点斜率 = 0 ---")
    print(f"s2(0)   = {s2(0)}")
    print(f"s2(0.5) = {s2(0.5)}")  # 不再是 0.5，因为第一段不是直线了
    print(f"s2(1)   = {s2(1)}")
    print(f"s2(1.5) = {s2(1.5)}")
