import numpy as np
import pandas as pd

def ahp_eigen(matrix):
    """特征根法求权重 + 一致性检验"""
    M = np.array(matrix, dtype=np.float64)
    n = M.shape[0]
    eigvals, eigvecs = np.linalg.eig(M)
    lambda_max = np.max(np.real(eigvals))
    idx = np.argmax(np.real(eigvals))
    v = np.real(eigvecs[:, idx])
    w = v / v.sum()
    CI = (lambda_max - n) / (n - 1)
    RI_table = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    RI = RI_table.get(n, 1.49)
    CR = CI / RI
    return w, lambda_max, CI, CR, CR < 0.1

def build_criteria_matrix(n=8):
    """构造准则层比较矩阵：第i行从第i列开始是1,2,3,..."""
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            A[i, j] = j - i + 1
            if i != j:
                A[j, i] = 1.0 / A[i, j]
    return A

def main(csv_path='house_score.csv'):
    df = pd.read_csv(csv_path)
    criteria_names = ['任职时间', '工作时间', '职级', '职称',
                      '配偶情况', '学历', '出生年月', '奖励加分']
    D = df[criteria_names].values.astype(np.float64)  # 40×8

    # Step 1: 准则层
    A = build_criteria_matrix(8)
    W1, lambda_A, CI_A, CR_A, ok_A = ahp_eigen(A)

    # Step 2: 方案层（直接评分数据列归一化）
    W2 = D / D.sum(axis=0)

    # Step 3: 组合权重
    W = (W2 @ W1.reshape(-1, 1)).flatten()

    # Step 4: 排名
    ranking = np.argsort(-W)
    print(f"{'排名':<6} {'人员':<8} {'权重':<10}")
    for rank, idx in enumerate(ranking, 1):
        print(f"{rank:<6} {int(df.iloc[idx]['人员']):<8} {W[idx]:.6f}")

    return W1, W2, W, ranking

if __name__ == '__main__':
    main()