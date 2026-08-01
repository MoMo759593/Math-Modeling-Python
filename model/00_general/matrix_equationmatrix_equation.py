def solve_linear_system(A, b):
    """
    通用高斯消元法解 n 元线性方程组 Ax = b

    参数:
        A: n×n 系数矩阵，list套list，如 [[2,1,-1], [-3,-1,2], [-2,1,2]]
        b: 右侧常数项，单list，如 [8, -11, -3]

    返回:
        list: 解向量 [x1, x2, ..., xn]
        或 None: 方程组无唯一解（无解或无穷多解）
    """
    n = len(A)

    # 基本校验
    if len(b) != n:
        raise ValueError(f"维度不匹配：A是{n}×{n}，b的长度是{len(b)}")
    for row in A:
        if len(row) != n:
            raise ValueError("系数矩阵必须是方阵（每行长度相等）")

    # 构造增广矩阵 [A | b]
    aug = [A[i][:] + [b[i]] for i in range(n)]

    # ========== 前向消元：化为上三角矩阵 ==========
    for col in range(n):
        # 1. 选主元：在当前列下方找绝对值最大的行（部分主元法）
        pivot_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[pivot_row][col]):
                pivot_row = row

        # 2. 如果主元近似为0，矩阵奇异，无唯一解
        if abs(aug[pivot_row][col]) < 1e-12:
            return None

        # 3. 交换当前行与主元行
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        # 4. 消去下方所有行的当前列元素
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            # 从当前列开始消（左边的已经是0了，不用算）
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    # ========== 回代求解 ==========
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        # 先减去已知的右边部分
        known_sum = sum(aug[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (aug[i][n] - known_sum) / aug[i][i]

    return x


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 示例1：3元方程组
    A1 = [
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ]
    b1 = [8, -11, -3]

    sol1 = solve_linear_system(A1, b1)
    print("=" * 40)
    print("示例1：3元方程组")
    print(f"系数矩阵 A: {A1}")
    print(f"常数项 b:   {b1}")
    print(f"解 x:       {sol1}")

    # 验算
    if sol1:
        print("验算 Ax:")
        for i in range(len(A1)):
            val = sum(A1[i][j] * sol1[j] for j in range(len(A1)))
            print(f"  方程{i + 1}: {val:.6f} ≈ {b1[i]}")

    # 示例2：2元方程组
    A2 = [[3, 2], [1, -1]]
    b2 = [7, 1]
    sol2 = solve_linear_system(A2, b2)
    print("\n" + "=" * 40)
    print("示例2：2元方程组")
    print(f"A: {A2}, b: {b2}")
    print(f"解 x: {sol2}")

    # 示例3：4元方程组
    A3 = [
        [2, -1, 3, 1],
        [1, 2, -1, 2],
        [3, 1, 2, -1],
        [1, -1, 1, 3]
    ]
    b3 = [10, 5, 11, 8]
    sol3 = solve_linear_system(A3, b3)
    print("\n" + "=" * 40)
    print("示例3：4元方程组")
    print(f"解 x: {sol3}")

    # 示例4：奇异矩阵（无解/无穷多解）
    A4 = [[1, 2], [2, 4]]
    b4 = [3, 7]
    sol4 = solve_linear_system(A4, b4)
    print("\n" + "=" * 40)
    print("示例4：奇异矩阵（无唯一解）")
    print(f"A: {A4}, b: {b4}")
    print(f"解 x: {sol4}  ← 返回None，表示无唯一解")