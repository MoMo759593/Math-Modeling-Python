"""
黄河小浪底调水调沙问题 — Python实现
=====================================
题目要求：
1. 给出估算任意时刻的排沙量及总排沙量的方法
2. 确定排沙量与流水量的变化关系

实现内容：
- 三次B样条插值（k=3）估算任意时刻排沙量
- 对B样条函数积分求总排沙量
- 三次多项式最小二乘拟合排沙量与流水量的关系
"""

import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.integrate import quad
import matplotlib.pyplot as plt

# ============================================================
# 1. 数据读取与预处理（修复后的CSV，直接读取）
# ============================================================

df = pd.read_csv('yellow_river.csv')

# 构建时间轴（小时），从0开始，每12小时一个点
t_data = np.arange(0, len(df) * 12, 12)
df['t'] = t_data

# 计算排沙量 S = discharge * sand_content
# discharge: m³/s, sand_content: kg/m³, S: kg/s
df['sediment_discharge'] = df['discharge'] * df['sand_content']

print("=" * 60)
print("【数据概览】")
print("=" * 60)
print(df[['date', 'time', 't', 'discharge', 'sand_content', 'sediment_discharge']])
print(f"\n时间范围: {t_data[0]} ~ {t_data[-1]} 小时")
print(f"数据点数: {len(df)}")


# ============================================================
# 2. 三次B样条插值：估算任意时刻排沙量及总排沙量
# ============================================================

S_data = df['sediment_discharge'].values  # 排沙量 (kg/s)

# 三次B样条插值（k=3）
bspline = make_interp_spline(t_data, S_data, k=3)

def S_hour(t):
    """排沙量函数，t单位为小时，返回kg/s"""
    return bspline(t)

# 生成密集时间点用于绘图
t_fine = np.linspace(t_data[0], t_data[-1], 1000)
S_fine = bspline(t_fine)

# 总排沙量 = ∫ S(t) dt，t单位为秒
# S单位是kg/s，dt_hours 需转换为秒：dt = 3600 * dt_hours
total_sediment_kg, err = quad(lambda th: S_hour(th) * 3600, t_data[0], t_data[-1])

print("\n" + "=" * 60)
print("【第一部分】三次B样条插值与总排沙量")
print("=" * 60)
print(f"\n三次B样条插值参数:")
print(f"  节点数 (knots): {len(bspline.t)}")
print(f"  样条阶数 (degree): 3")
print(f"  系数数: {len(bspline.c)}")
print(f"\n总排沙量 = {total_sediment_kg:.2f} kg")
print(f"总排沙量 = {total_sediment_kg / 1e3:.2f} 吨")
print(f"总排沙量 = {total_sediment_kg / 1e7:.2f} 万吨")
print(f"总排沙量 = {total_sediment_kg / 1e8:.4f} 亿吨")
print(f"积分误差估计: {err:.4e}")

# 任意时刻排沙量估算示例
test_times = [6, 18, 30, 100, 150, 200]
test_labels = ['6.29 14:00', '6.29 22:00', '6.30 14:00', '7.3 4:00', '7.5 2:00', '7.7 8:00']
print(f"\n任意时刻排沙量估算示例:")
for label, t_val in zip(test_labels, test_times):
    print(f"  {label:12s} (t={t_val:3d}h): S = {S_hour(t_val):10.2f} kg/s")


# ============================================================
# 3. 三次多项式最小二乘拟合：排沙量与流水量的关系
# ============================================================

Q_data = df['discharge'].values  # 水流量 (m³/s)

# 使用 numpy.polyfit 进行三次多项式最小二乘拟合
# coeffs = [a3, a2, a1, a0]，对应 S = a3*Q^3 + a2*Q^2 + a1*Q + a0
coeffs = np.polyfit(Q_data, S_data, deg=3)
p3 = np.poly1d(coeffs)

# 拟合优度
S_pred = p3(Q_data)
SS_res = np.sum((S_data - S_pred) ** 2)
SS_tot = np.sum((S_data - np.mean(S_data)) ** 2)
R2 = 1 - SS_res / SS_tot
RMSE = np.sqrt(np.mean((S_data - S_pred) ** 2))

print("\n" + "=" * 60)
print("【第二部分】三次多项式最小二乘拟合")
print("=" * 60)
print(f"\n拟合模型: S = a₃·Q³ + a₂·Q² + a₁·Q + a₀")
print(f"\n拟合系数:")
print(f"  a₃ = {coeffs[0]:.6e}")
print(f"  a₂ = {coeffs[1]:.6e}")
print(f"  a₁ = {coeffs[2]:.6e}")
print(f"  a₀ = {coeffs[3]:.6e}")
print(f"\n拟合效果: R² = {R2:.6f}, RMSE = {RMSE:.2f}")
print(f"\n拟合方程:")
print(f"  S(Q) = ({coeffs[0]:.4e})·Q³ + ({coeffs[1]:.4e})·Q² + ({coeffs[2]:.4e})·Q + ({coeffs[3]:.4e})")


# ============================================================
# 4. 可视化
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 图1：排沙量随时间变化（B样条插值）
ax1 = axes[0]
ax1.scatter(t_data, S_data, c='red', s=80, zorder=5, label='观测数据点', edgecolors='black')
ax1.plot(t_fine, S_fine, 'b-', linewidth=2, label='三次B样条插值')
ax1.set_xlabel('时间 t (小时)', fontsize=12)
ax1.set_ylabel('排沙量 S (kg/s)', fontsize=12)
ax1.set_title('排沙量随时间变化（三次B样条插值）', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 图2：排沙量与流水量的关系（三次多项式拟合）
ax2 = axes[1]
Q_fine = np.linspace(Q_data.min(), Q_data.max(), 500)
S_fit = p3(Q_fine)
ax2.scatter(Q_data, S_data, c='red', s=80, zorder=5, label='观测数据点', edgecolors='black')
ax2.plot(Q_fine, S_fit, 'b-', linewidth=2.5, label=f'三次多项式拟合 (R²={R2:.4f})')
ax2.set_xlabel('水流量 Q (m³/s)', fontsize=12)
ax2.set_ylabel('排沙量 S (kg/s)', fontsize=12)
ax2.set_title('排沙量与流水量的关系（三次多项式最小二乘拟合）', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

textstr = (f'S(Q) = {coeffs[0]:.2e}·Q³ + {coeffs[1]:.2e}·Q²\n'
           f'        + {coeffs[2]:.2e}·Q + {coeffs[3]:.2e}\n'
           f'R² = {R2:.4f}')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax2.text(0.98, 0.35, textstr, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('yellow_river_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n图表已保存为 yellow_river_analysis.png")