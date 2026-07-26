import numpy as np

def equidistant_bspline(x_list, y_list, k=3):
    x = np.asarray(x_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    n = len(x) - 1

    if len(y) != n + 1:
        raise ValueError("x_list 和 y_list 长度必须相同")
    if n + 1 < k + 1:
        raise ValueError(f"数据点数量({n+1})必须至少为 k+1 = {k+1}")
    if not np.allclose(np.diff(x), x[1] - x[0]):
        raise ValueError("x_list 必须是等距的")

    # ---- 构造 not-a-knot 节点向量 ----
    inner = x[2:n-1] if n >= k + 1 else np.array([])
    t = np.concatenate([np.full(k+1, x[0]), inner, np.full(k+1, x[-1])])
    m = len(t) - k - 1          # 控制点数量 = n + 1

    # ---- de Boor 求值（对给定控制点）----
    def de_boor(ctrl, xv):
        if xv <= t[k]:          return float(ctrl[0])
        if xv >= t[-k-1]:       return float(ctrl[-1])

        mu = np.searchsorted(t, xv, side='right') - 1
        mu = max(k, min(mu, len(t) - k - 2))
        d = ctrl[mu-k:mu+1].copy().astype(float)

        for r in range(1, k+1):
            for j in range(k, r-1, -1):
                denom = t[mu+j-r+1] - t[mu-k+j]
                alpha = (xv - t[mu-k+j]) / denom if abs(denom) > 1e-14 else 0.0
                d[j] = (1-alpha) * d[j-1] + alpha * d[j]
        return float(d[k])

    # ---- 构建插值矩阵 A[i,j] = N_{j,k}(x_i) ----
    A = np.zeros((m, m))
    for j in range(m):
        c_unit = np.zeros(m);  c_unit[j] = 1.0
        for i in range(n+1):
            A[i, j] = de_boor(c_unit, x[i])

    c = np.linalg.solve(A, y)   # 求解控制点

    # ---- 包装求值函数 ----
    def S(xv):
        xv_arr = np.asarray(xv, dtype=float)
        scalar = (xv_arr.ndim == 0)
        if scalar:  xv_arr = np.array([xv_arr])
        res = np.array([de_boor(c, v) for v in xv_arr])
        return res[0] if scalar else res

    S.knots = t
    S.control_points = c
    S.degree = k
    return S

x = np.linspace(0, 10, 11)      # 等距节点
y = np.sin(x)                    # 对应值

S = equidistant_bspline(x, y)    # 构建 B 样条

# 标量求值
print(S(2.5))                    # 0.598...

# 数组求值
print(S(np.array([0.5, 1.2, 3.7, 5.0])))
# [ 0.50174335  0.92496506 -0.52963865 -0.95892427]

# 查看内部结构
print(S.knots)           # 节点向量
print(S.control_points)  # 控制点（de Boor 点）