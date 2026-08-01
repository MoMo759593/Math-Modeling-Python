def correlation_analysis(x_list, y_list, sample=True):
    """
    计算协方差、标准差和相关系数

    参数:
        x_list: X 的观测值列表
        y_list: Y 的观测值列表
        sample: True 用样本标准（除以 n-1），False 用总体标准（除以 n）

    返回:
        dict: 包含协方差、标准差、相关系数等
    """
    if len(x_list) != len(y_list):
        raise ValueError("x_list 和 y_list 长度必须相同")
    if len(x_list) == 0:
        raise ValueError("输入列表不能为空")
    if sample and len(x_list) < 2:
        raise ValueError("样本计算需要至少 2 个数据点")

    n = len(x_list)
    divisor = n - 1 if sample else n

    # 均值
    mean_x = sum(x_list) / n
    mean_y = sum(y_list) / n

    # 协方差 Cov(X,Y)
    cov_xy = sum((x_list[i] - mean_x) * (y_list[i] - mean_y) for i in range(n)) / divisor

    # 标准差 S_X, S_Y
    var_x = sum((x - mean_x) ** 2 for x in x_list) / divisor
    var_y = sum((y - mean_y) ** 2 for y in y_list) / divisor
    s_x = var_x ** 0.5
    s_y = var_y ** 0.5

    # 相关系数 r
    if s_x == 0 or s_y == 0:
        r = float('nan')  # 标准差为0时相关系数无意义
    else:
        r = cov_xy / (s_x * s_y)

    return {
        "n": n,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "Cov(X,Y)": cov_xy,
        "S_X": s_x,
        "S_Y": s_y,
        "r": r,
        "type": "样本" if sample else "总体"
    }


# ========== 示例用法 ==========
if __name__ == "__main__":
    # 示例数据：身高(cm) 与 体重(kg)
    x_list = [160, 165, 170, 175, 180, 185]
    y_list = [55, 60, 65, 70, 75, 82]

    result = correlation_analysis(x_list, y_list, sample=True)

    print("=" * 45)
    print("相关性分析结果（样本统计量）")
    print("=" * 45)
    print(f"样本量 n = {result['n']}")
    print(f"X 均值 = {result['mean_x']:.4f}, Y 均值 = {result['mean_y']:.4f}")
    print("-" * 45)
    print(f"协方差 Cov(X,Y) = {result['Cov(X,Y)']:.4f}")
    print(f"标准差 S_X = {result['S_X']:.4f}")
    print(f"标准差 S_Y = {result['S_Y']:.4f}")
    print("-" * 45)
    print(f"相关系数 r = {result['r']:.4f}")
    print("=" * 45)

    # 相关系数解读
    r_abs = abs(result['r'])
    if r_abs >= 0.9:
        strength = "极强相关"
    elif r_abs >= 0.7:
        strength = "强相关"
    elif r_abs >= 0.5:
        strength = "中等相关"
    elif r_abs >= 0.3:
        strength = "弱相关"
    else:
        strength = "极弱/无相关"
    direction = "正相关" if result['r'] > 0 else "负相关"
    print(f"解读: {strength}（{direction}）")