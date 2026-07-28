import numpy as np

def ahp_root_method(matrix):
    """
    求根法（几何平均法）求AHP权重
    输入: list of list 或 np.ndarray，k阶正互反矩阵
    返回: np.ndarray，长度为k的权重向量
    """
    A = np.array(matrix, dtype=np.float64)
    n = A.shape[0]

    # 步骤1：每行求几何平均
    geo_mean = np.prod(A, axis=1) ** (1 / n)

    # 步骤2：归一化
    w = geo_mean / geo_mean.sum()

    return w

A = [
    [1,    2,    6   ],
    [0.5,  1,    4   ],
    [1/6,  0.25, 1   ]
]

print("\n=== 求根法 ===")
w2 = ahp_root_method(A)
print(w2)