def variance_population(x_list):
    """
    总体方差（除以 n）
    公式: S² = E[(X - E(X))²] = (1/n) * Σ(xi - μ)²
    """
    if not x_list or len(x_list) == 0:
        raise ValueError("输入列表不能为空")

    n = len(x_list)
    mean = sum(x_list) / n
    return sum((x - mean) ** 2 for x in x_list) / n


def variance_sample(x_list):
    """
    样本方差（除以 n-1，无偏估计）
    公式: S² = (1/(n-1)) * Σ(xi - x̄)²
    """
    if len(x_list) < 2:
        raise ValueError("样本方差需要至少 2 个数据点")

    n = len(x_list)
    mean = sum(x_list) / n
    return sum((x - mean) ** 2 for x in x_list) / (n - 1)


# ========== 示例用法 ==========
if __name__ == "__main__":
    data = [1.5, 2.0, 3.5, 4.0, 5.5]

    print(f"数据: {data}")
    print(f"数据个数 n = {len(data)}")
    print(f"平均值 E(x) = {sum(data) / len(data):.4f}")
    print(f"总体方差 S²(x) = {variance_population(data):.4f}")
    print(f"样本方差 S²(x) = {variance_sample(data):.4f}")
    print(f"总体标准差 = {variance_population(data) ** 0.5:.4f}")
    print(f"样本标准差 = {variance_sample(data) ** 0.5:.4f}")