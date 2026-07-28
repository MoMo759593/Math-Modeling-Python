import numpy as np

def ahp_sum_method(matrix):
    """
    和法（算术平均法）求AHP权重
    输入: list of list 或 np.ndarray，k阶正互反矩阵
    返回: np.ndarray，长度为k的权重向量（各分量之和为1）
    """
    A = np.array(matrix, dtype=np.float64)
    n = A.shape[0]

    # 步骤1：列归一化
    col_sum = A.sum(axis=0)
    A_norm = A / col_sum

    # 步骤2：行求和，再归一化
    row_sum = A_norm.sum(axis=1)
    w = row_sum / row_sum.sum()

    return w

A = [
    [1,    2,    6   ],
    [0.5,  1,    4   ],
    [1/6,  0.25, 1   ]
]

print("=== 和法 ===")
w1 = ahp_sum_method(A)
print(w1)