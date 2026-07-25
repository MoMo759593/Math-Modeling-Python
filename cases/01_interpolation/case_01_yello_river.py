"""
案例一：黄河小浪底调水调沙问题
章节：插值与拟合方法
方法：分段线性插值、三次样条插值、数值积分（梯形法）
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置，避免 PyCharm 后端崩溃
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from scipy.integrate import trapezoid

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 原始数据 ====================

# 日期：6.29 ~ 7.10（共12天，每天8:00和20:00两个观测点）
dates = pd.date_range(start='2004-06-29', periods=12, freq='D')

# 水流量 (m3/s)
flow = np.array([
    1800, 1900, 2100, 2200, 2300, 2400, 2500, 2600, 2650, 2700, 2720, 2650,
    2600, 2500, 2300, 2200, 2000, 1850, 1820, 1800, 1750, 1500, 1000, 900
], dtype=float)

# 含沙量 (kg/m3)
sediment_conc = np.array([
    32, 60, 75, 85, 90, 98, 100, 102, 108, 112, 115, 116,
    118, 120, 118, 105, 80, 60, 50, 30, 26, 20, 8, 5
], dtype=float)

# 排沙量 = 水流量 * 含沙量 (kg/s)
discharge = flow * sediment_conc

# 时间轴（小时，以6.29 8:00为起点0，每12小时一个点，共24个点）
time_hours = np.arange(0, 12 * 24, 12)

# ==================== 2. 插值与拟合 ====================

t_fine = np.linspace(time_hours.min(), time_hours.max(), 500)

# 2.1 分段线性插值
f_linear = interp1d(time_hours, discharge, kind='linear')
discharge_linear = f_linear(t_fine)

# 2.2 三次样条插值
cs = CubicSpline(time_hours, discharge)
discharge_spline = cs(t_fine)

# 2.3 多项式拟合（3次）
coeffs = np.polyfit(time_hours, discharge, 3)
p = np.poly1d(coeffs)
discharge_poly = p(t_fine)

# ==================== 3. 数值积分估算总排沙量 ====================

# 排沙量单位 kg/s，时间单位小时，总排沙量 = integral(discharge dt) * 3600 / 1e8 (亿吨)
total_linear = trapezoid(discharge_linear, t_fine) * 3600 / 1e8
total_spline = trapezoid(discharge_spline, t_fine) * 3600 / 1e8
total_poly = trapezoid(discharge_poly, t_fine) * 3600 / 1e8
total_raw = trapezoid(discharge, time_hours) * 3600 / 1e8

print("=" * 50)
print("黄河小浪底调水调沙 - 总排沙量估算")
print("=" * 50)
print(f"原始数据梯形法: {total_raw:.4f} 亿吨")
print(f"线性插值积分:   {total_linear:.4f} 亿吨")
print(f"三次样条积分:   {total_spline:.4f} 亿吨")
print(f"三次多项式拟合: {total_poly:.4f} 亿吨")
print("=" * 50)

# ==================== 4. 可视化 ====================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- 子图1：水流量与含沙量 ---
ax1 = axes[0, 0]
ax1_twin = ax1.twinx()
l1 = ax1.plot(time_hours, flow, 'b-o', label='水流量', markersize=5)
l2 = ax1_twin.plot(time_hours, sediment_conc, 'r-s', label='含沙量', markersize=5)
ax1.set_xlabel('时间 (小时, 以6.29 8:00为0)')
ax1.set_ylabel('水流量 (m3/s)', color='b')
ax1_twin.set_ylabel('含沙量 (kg/m3)', color='r')
ax1.set_title('原始观测数据')
ax1.grid(True, alpha=0.3)
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

# --- 子图2：排沙量插值对比 ---
ax2 = axes[0, 1]
ax2.scatter(time_hours, discharge / 1e3, color='black', s=60, zorder=5, label='观测值')
ax2.plot(t_fine, discharge_linear / 1e3, '--', label='线性插值', alpha=0.8)
ax2.plot(t_fine, discharge_spline / 1e3, '-', label='三次样条', linewidth=2)
ax2.plot(t_fine, discharge_poly / 1e3, '-.', label='三次多项式拟合', alpha=0.8)
ax2.set_xlabel('时间 (小时)')
ax2.set_ylabel('排沙量 (x10^3 kg/s)')
ax2.set_title('排沙量插值与拟合对比')
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- 子图3：排沙量变化趋势 ---
ax3 = axes[1, 0]
ax3.fill_between(t_fine, 0, discharge_spline / 1e3, alpha=0.3, color='green')
ax3.plot(t_fine, discharge_spline / 1e3, 'g-', linewidth=2, label='三次样条插值')
ax3.scatter(time_hours, discharge / 1e3, color='red', s=50, zorder=5, label='观测点')
ax3.set_xlabel('时间 (小时)')
ax3.set_ylabel('排沙量 (x10^3 kg/s)')
ax3.set_title('排沙量变化趋势（样条插值）')
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- 子图4：总排沙量估算对比 ---
ax4 = axes[1, 1]
methods = ['原始梯形', '线性插值', '三次样条', '多项式拟合']
values = [total_raw, total_linear, total_spline, total_poly]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
bars = ax4.bar(methods, values, color=colors, edgecolor='black', alpha=0.8)
ax4.set_ylabel('总排沙量 (亿吨)')
ax4.set_title('不同方法总排沙量估算对比')
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax4.annotate(f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.suptitle('黄河小浪底调水调沙问题 - 插值与拟合分析', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存图片到当前文件所在目录下的 result 文件夹
save_dir = os.path.join(os.path.dirname(__file__), 'result')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'case_01_yellow_river.png')
plt.savefig(save_path, dpi=200, bbox_inches='tight')

print(f"\n结果图已保存至: {save_path}")
print("请前往上述路径查看图片")