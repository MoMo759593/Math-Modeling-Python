#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
层次分析法（AHP）住宅分配排序
基于40人原始数据，从7个准则维度进行综合排序
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========================
# 1. 读取数据
# ========================
# 请将 house.csv 放在与本脚本同一目录下
df = pd.read_csv('house.csv', dtype={
    'employment_time': str,
    'working_time': str,
    'deat_of_birth': str
})

# ========================
# 2. 解析时间格式（如 1991.6 → 1991年6月）
# ========================
def parse_year_month(s):
    """解析 '1954.9' 或 '1957.11' 为 (年, 月)"""
    if pd.isna(s):
        return np.nan, np.nan
    s = str(s).strip()
    parts = s.split('.')
    year = int(parts[0])
    month = int(parts[1])
    return year, month

for col in ['employment_time', 'working_time', 'deat_of_birth']:
    df[[f'{col}_y', f'{col}_m']] = df[col].apply(lambda x: pd.Series(parse_year_month(x)))
    df[f'{col}_months'] = df[f'{col}_y'] * 12 + df[f'{col}_m']

# ========================
# 3. 量化各准则得分
# ========================

# C1: 职称职务得分 = 职称分 × 级别分
title_map = {'初级': 2, '中级': 4, '高级': 6}
rank_map = {8: 2, 9: 3}
df['C1_职称职务'] = df['professional_title'].map(title_map) * df['rank'].map(rank_map)

# C2: 工龄得分（参加工作越早，工龄越长，得分越高）
df['C2_工龄'] = df['working_time_months'].max() - df['working_time_months'] + 1

# C3: 来院时间得分（来院越早，得分越高）
df['C3_来院时间'] = df['employment_time_months'].max() - df['employment_time_months'] + 1

# C4: 学历得分
edu_map = {'中专': 1, '大专': 2, '本科': 3, '硕士': 4, '博士': 5, '博士后': 6}
df['C4_学历'] = df['education'].map(edu_map)

# C5: 配偶状态得分
spouse_map = {'院外': 1, '院内职工': 2, '院内干部': 3}
df['C5_配偶状态'] = df['spouse_status'].map(spouse_map)

# C6: 年龄得分（出生越早，年龄越大，得分越高）
df['C6_年龄'] = df['deat_of_birth_months'].max() - df['deat_of_birth_months'] + 1

# C7: 奖励分（直接使用原始分）
df['C7_奖励分'] = df['reward_points']

# ========================
# 4. AHP 判断矩阵与权重计算
# ========================
# 准则顺序: [职称职务, 工龄, 来院时间, 学历, 配偶状态, 年龄, 奖励分]
A = np.array([
    [1,    2,    2,    4,    3,    2,    5],    # 职称职务
    [1/2,  1,    2,    3,    2,    2,    4],    # 工龄
    [1/2,  1/2,  1,    2,    2,    1,    3],    # 来院时间
    [1/4,  1/3,  1/2,  1,    1/2,  1/2,  2],    # 学历
    [1/3,  1/2,  1/2,  2,    1,    1,    3],    # 配偶状态
    [1/2,  1/2,  1,    2,    1,    1,    3],    # 年龄
    [1/5,  1/4,  1/3,  1/2,  1/3,  1/3,  1]     # 奖励分
])

criteria_names = ['职称职务', '工龄', '来院时间', '学历', '配偶状态', '年龄', '奖励分']

# 特征值法求权重
eigenvalues, eigenvectors = np.linalg.eig(A)
max_index = np.argmax(eigenvalues.real)
weights = eigenvectors[:, max_index].real
weights = weights / weights.sum()  # 归一化

# 一致性检验
n = A.shape[0]
lambda_max = eigenvalues[max_index].real
CI = (lambda_max - n) / (n - 1)
RI_table = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
RI = RI_table[n]
CR = CI / RI

print("=" * 60)
print("【准则层判断矩阵】")
print(pd.DataFrame(A, index=criteria_names, columns=criteria_names).round(3))
print(f"\n最大特征值 λ_max = {lambda_max:.4f}")
print(f"一致性指标 CI = {CI:.4f}")
print(f"随机一致性指标 RI = {RI:.4f}")
print(f"一致性比率 CR = {CR:.4f}")
print(f"一致性检验: {'通过 ✓' if CR < 0.1 else '不通过 ✗'} (CR < 0.1)")
print("\n【准则层权重】")
for name, w in zip(criteria_names, weights):
    print(f"  {name:8s}: {w:.4f} ({w*100:.2f}%)")

# ========================
# 5. 方案层得分计算与排序
# ========================
score_cols = ['C1_职称职务', 'C2_工龄', 'C3_来院时间', 'C4_学历', 'C5_配偶状态', 'C6_年龄', 'C7_奖励分']
norm_cols = []

# min-max 标准化到 [0, 1]
for col in score_cols:
    norm_col = col + '_norm'
    min_v = df[col].min()
    max_v = df[col].max()
    df[norm_col] = (df[col] - min_v) / (max_v - min_v) if max_v != min_v else 0.5
    norm_cols.append(norm_col)

# 加权求和得综合得分
df['综合得分'] = sum(df[norm_col] * w for norm_col, w in zip(norm_cols, weights))

# 排序
df_sorted = df.sort_values('综合得分', ascending=False).reset_index(drop=True)
df_sorted['排序'] = range(1, len(df_sorted) + 1)

# ========================
# 6. 输出排序结果
# ========================
print("\n" + "=" * 80)
print("【40人住宅分配 AHP 排序结果】")
print("=" * 80)
output_cols = ['排序', 'personnel', 'professional_title', 'rank', 'working_time',
               'employment_time', 'education', 'spouse_status', 'deat_of_birth',
               'reward_points', '综合得分']
print(df_sorted[output_cols].to_string(index=False))

# ========================
# 7. 保存结果到 CSV
# ========================
output_df = df_sorted[['排序', 'personnel', 'rank', 'professional_title', 'working_time',
                         'employment_time', 'education', 'spouse_status', 'deat_of_birth',
                         'reward_points', 'C1_职称职务', 'C2_工龄', 'C3_来院时间',
                         'C4_学历', 'C5_配偶状态', 'C6_年龄', 'C7_奖励分', '综合得分']].copy()

output_df.columns = ['排序', '人员编号', '级别', '职称', '参加工作时间', '来院时间',
                     '学历', '配偶状态', '出生年月', '奖励分', '职称职务原始分', '工龄原始分',
                     '来院时间原始分', '学历原始分', '配偶状态原始分', '年龄原始分',
                     '奖励分原始分', 'AHP综合得分']

output_df.to_csv('ahp_housing_result.csv', index=False, encoding='utf-8-sig')
print("\n结果已保存至: ahp_housing_result.csv")

# ========================
# 8. 可视化
# ========================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 图1: 准则权重饼图
ax1 = axes[0, 0]
colors = plt.cm.Set3(np.linspace(0, 1, len(criteria_names)))
ax1.pie(weights, labels=criteria_names, autopct='%1.1f%%',
        colors=colors, startangle=90, explode=[0.02]*len(weights))
ax1.set_title('准则层权重分布', fontsize=14, fontweight='bold')

# 图2: 40人综合得分排序条形图
ax2 = axes[0, 1]
y_pos = np.arange(len(df_sorted))
bars = ax2.barh(y_pos, df_sorted['综合得分'],
                color=plt.cm.RdYlGn(df_sorted['综合得分'] / df_sorted['综合得分'].max()))
ax2.set_yticks(y_pos)
ax2.set_yticklabels([f"#{row['排序']} 人员{int(row['personnel'])}" for _, row in df_sorted.iterrows()], fontsize=8)
ax2.invert_yaxis()
ax2.set_xlabel('综合得分', fontsize=12)
ax2.set_title('40人住宅分配综合得分排序', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for i in range(5):
    ax2.text(df_sorted.iloc[i]['综合得分'] + 0.01, i, f"{df_sorted.iloc[i]['综合得分']:.3f}",
             va='center', fontsize=8, color='darkred', fontweight='bold')

# 图3: 前10名各准则得分对比
ax3 = axes[1, 0]
top10 = df_sorted.head(10)
x = np.arange(10)
width = 0.1
for i, (col, name, w) in enumerate(zip(norm_cols, criteria_names, weights)):
    ax3.bar(x + i*width, top10[col], width, label=f'{name}({w*100:.1f}%)', alpha=0.8)
ax3.set_xticks(x + width * 3)
ax3.set_xticklabels([f"人员{int(p)}" for p in top10['personnel']], rotation=45, ha='right')
ax3.set_ylabel('标准化得分', fontsize=12)
ax3.set_title('前10名各准则标准化得分对比', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right', fontsize=8, ncol=2)
ax3.set_ylim(0, 1.2)

# 图4: 综合得分分布直方图
ax4 = axes[1, 1]
ax4.hist(df['综合得分'], bins=12, color='steelblue', edgecolor='white', alpha=0.7, density=True)
ax4.axvline(df['综合得分'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'均值={df["综合得分"].mean():.3f}')
ax4.axvline(df['综合得分'].median(), color='green', linestyle='--', linewidth=2,
            label=f'中位数={df["综合得分"].median():.3f}')
ax4.set_xlabel('综合得分', fontsize=12)
ax4.set_ylabel('密度', fontsize=12)
ax4.set_title('综合得分分布', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ahp_housing_sorting.png', dpi=150, bbox_inches='tight')
plt.show()
print("图表已保存至: ahp_housing_sorting.png")