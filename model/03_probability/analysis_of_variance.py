def anova_sum_squares(X_list):
    """
    单因素方差分析：计算组间平方和 S_A 与组内平方和 S_E

    参数:
        X_list: 包含若干组数据的列表，如 [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], ...]
                每个子列表代表一个水平（组）的观测值

    返回:
        dict: 包含 S_A, S_E, S_T, 以及各组均值、总均值等信息
    """
    if not X_list or any(len(group) == 0 for group in X_list):
        raise ValueError("输入列表不能为空，且每组数据不能为空")

    # 各组样本量
    n_list = [len(group) for group in X_list]
    # 总样本量
    N = sum(n_list)
    # 组数（水平数）
    k = len(X_list)

    # 各组均值
    group_means = [sum(group) / len(group) for group in X_list]
    # 总均值（所有数据的总平均）
    all_data = [x for group in X_list for x in group]
    total_mean = sum(all_data) / N

    # 组间平方和 S_A = Σ n_i * (x̄_i - x̄)^2
    S_A = sum(n_list[i] * (group_means[i] - total_mean) ** 2 for i in range(k))

    # 组内平方和 S_E = Σ Σ (x_ij - x̄_i)^2
    S_E = sum(
        sum((x - group_means[i]) ** 2 for x in X_list[i])
        for i in range(k)
    )

    # 总平方和 S_T = S_A + S_E（用于验证）
    S_T = sum((x - total_mean) ** 2 for x in all_data)

    return {
        "S_A": S_A,  # 组间平方和（因素效应）
        "S_E": S_E,  # 组内平方和（随机误差）
        "S_T": S_T,  # 总平方和
        "group_means": group_means,  # 各组均值
        "total_mean": total_mean,  # 总均值
        "N": N,  # 总样本量
        "k": k,  # 组数
        "n_list": n_list  # 各组样本量
    }


# ========== 示例用法 ==========
if __name__ == "__main__":
    # 示例：3个组，每组5个观测值
    X_list = [
        [24, 27, 26, 25, 24],  # 组1
        [31, 33, 32, 30, 31],  # 组2
        [40, 41, 39, 42, 40],  # 组3
    ]

    result = anova_sum_squares(X_list)

    print("=" * 40)
    print("单因素方差分析结果")
    print("=" * 40)
    print(f"组数 k = {result['k']}, 总样本量 N = {result['N']}")
    print(f"各组样本量: {result['n_list']}")
    print(f"各组均值: {[round(m, 4) for m in result['group_means']]}")
    print(f"总均值: {result['total_mean']:.4f}")
    print("-" * 40)
    print(f"组间平方和 S_A = {result['S_A']:.4f}")
    print(f"组内平方和 S_E = {result['S_E']:.4f}")
    print(f"总平方和   S_T = {result['S_T']:.4f}")
    print(f"验证: S_A + S_E = {result['S_A'] + result['S_E']:.4f}")
    print("-" * 40)
    print(f"组间均方 MS_A = {result['S_A'] / (result['k'] - 1):.4f}")
    print(f"组内均方 MS_E = {result['S_E'] / (result['N'] - result['k']):.4f}")
    print(f"F 统计量 = {(result['S_A'] / (result['k'] - 1)) / (result['S_E'] / (result['N'] - result['k'])):.4f}")