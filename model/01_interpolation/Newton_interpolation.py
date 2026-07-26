def newton_interpolation(x_list, y_list):
    """
    牛顿插值法

    参数:
        x_list: 已知节点的 x 坐标列表
        y_list: 已知节点的 y 坐标列表

    返回:
        一个函数 p(x)，输入任意 x 返回插值结果
    """
    n = len(x_list)
    if len(y_list) != n:
        raise ValueError("x_list 和 y_list 长度必须相同")
    if n == 0:
        raise ValueError("输入列表不能为空")

    # 复制一份，避免修改原始数据
    x = list(x_list)
    y = list(y_list)

    # 计算各阶差商，只保留需要的系数
    divided_diff = [y[0]]  # f[x0], f[x0,x1], f[x0,x1,x2], ...

    for k in range(1, n):
        # 从后往前算，避免覆盖
        for i in range(n - 1, k - 1, -1):
            y[i] = (y[i] - y[i - 1]) / (x[i] - x[i - k])
        divided_diff.append(y[k])

    def p(t):
        """牛顿插值多项式（秦九韶形式，O(n) 计算）"""
        result = divided_diff[n - 1]
        for k in range(n - 2, -1, -1):
            result = result * (t - x[k]) + divided_diff[k]
        return result

    return p

# 已知数据点
x = [0, 1, 2, 3, 4]
y = [1, 2, 4, 8, 16]   # 近似 2^x

# 得到插值函数
p = newton_interpolation(x, y)

# 任意插值
print(p(1.5))   # 2.8359375
print(p(2.5))   # 5.6484375
print(p(3.7))   # 任意你想算的 x