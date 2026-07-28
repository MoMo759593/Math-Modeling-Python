import numpy as np


def ahp_eigen_method(matrix):
    """
    特征根法求AHP权重（含一致性检验）
    输入: list of list 或 np.ndarray，k阶正互反矩阵
    返回: tuple (w, lambda_max, CI, CR, is_consistent)
        w: 权重向量
        lambda_max: 最大特征值
        CI: 一致性指标
        CR: 一致性比率
        is_consistent: bool，是否通过一致性检验（CR < 0.1）
    """
    A = np.array(matrix, dtype=np.float64)
    n = A.shape[0]

    # 求特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # 取最大实特征值及其索引
    lambda_max = np.max(np.real(eigenvalues))
    idx = np.argmax(np.real(eigenvalues))

    # 取对应特征向量，保留实部，归一化
    v = np.real(eigenvectors[:, idx])
    w = v / v.sum()

    # 一致性检验
    CI = (lambda_max - n) / (n - 1)

    RI_table = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    RI = RI_table.get(n, 1.49)  # n>9时取近似值
    CR = CI / RI

    is_consistent = CR < 0.1

    return w, lambda_max, CI, CR, is_consistent

A = [
    [1,    2,    6   ],
    [0.5,  1,    4   ],
    [1/6,  0.25, 1   ]
]

print("\n=== 特征根法 ===")
w3, lmax, ci, cr, ok = ahp_eigen_method(A)
print(f"权重: {w3}")
print(f"λ_max={lmax:.4f}, CI={ci:.4f}, CR={cr:.4f}, 通过检验={ok}")