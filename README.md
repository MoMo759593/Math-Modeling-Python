# Math-Modeling-Python
本仓库以数学建模方法及其应用（第三版）为参考，分享一些数学建模案例的Python实现.  
各板块案例，均在对应case中列有原始数据.csv.  

## 📑 目录

- [引言](#引言)
- [1. 插值与拟合方法](#1-插值与拟合方法)
- [2. 层次分析方法](#2-层次分析方法)
- [3. 概率统计方法](#3-概率统计方法)
- [4. 回归分析方法](#4-回归分析方法)
- [5. 综合评价方法](#5-综合评价方法)
- [6. 线性规划方法](#6-线性规划方法)
- [7. 整数规划方法](#7-整数规划方法)
- [8. 排队论方法](#8-排队论方法)
- [9. 灰色系统分析方法](#9-灰色系统分析方法)

---

## 引言
本仓库会系统性地梳理每一章解决的问题、方法的优缺点、模型，以及案例分析.  

概念补充：  
1. 基函数：类比向量空间的基向量.例如，对于多项式，x^0,x^1……x^n即为基函数. 

---

## 1-插值与拟合方法

插值：构造函数，精确通过所有已知数据点.适用于少而精的数据.  
**常用：等距B样条函数.**  
拟合：追求整体最优趋势.适用于多而杂的数据.

**1.一般插值方法**：拉格朗日插值、牛顿插值  
   特点：全局多项式插值.对n+1个节点，有最高此项不超过n的多项式.  
   [拉格朗日插值](model/01_interpolation/Lagrrange_interpolation.py)：构造基函数L.  
   [牛顿插值](model/01_interpolation/Newton_interpolation.py)：基于差商递推，每次只加末项.  
   作用：估计已知离散值间任意数值的值(如求中值).  
   解决问题场景：数量少、计算量小的离散场景.  
   缺点：节点越多，断点越震荡，逼近效果差.因此，只适用于节点数量小的情况.
   
**2.样条函数插值方法**：二次样条函数插值、三次样条函数插值  
   特点：采用分段低次多项式和节点处光滑拼接，能克服一般插值方法的缺点.  
   
   [二次样条函数插值](model/01_interpolation/quadratic_splinequadratic_spline.py)：在每两个相邻的x区间里，拟合到x的二次方，拼接.其函数值、一阶导数连续，但二阶导数不连续，可能会有“折角”感.  
   其一般式为：<img width="334" height="23" alt="image" src="https://github.com/user-attachments/assets/8e2b2b69-c924-4ecd-83ff-36d9b900c368" />  
   三个数值的计算方式为：  
   1. <img width="171" height="26" alt="image" src="https://github.com/user-attachments/assets/4909a75e-84bf-4981-b59d-a4fba4b73a7f" />
   2. <img width="140" height="24" alt="image" src="https://github.com/user-attachments/assets/c03f448c-9b89-4aee-9349-21f33b5da257" />
   3. <img width="80" height="23" alt="image" src="https://github.com/user-attachments/assets/e04aaadf-75dd-43a8-9105-4d992fb620bb" />  
   导数值用商差解决.  
   
   [三次样条函数插值](model/01_interpolation/cubic_spline.py)：是二次样条函数的递推.二阶导数连续，更光滑.  
   其一般式为：<img width="310" height="39" alt="image" src="https://github.com/user-attachments/assets/d65007ad-a1b0-440e-97c2-963028f254d2" />  
   四个数值由左右端点值和左右端点二阶导值确定.  
   
   作用：构造通过所有数据的光滑曲线.  
   解决问题场景：数据多，要求光滑曲线的场景.  
   
**3.B样条函数插值方法**：磨光函数、等距B样条函数、一维等距B样条函数插值、二维等距B样条函数插值  
   特点：用局部支撑的基函数，而非用分段多项式函数.  
   
   磨光函数：对折线函数多次积分进行“磨光”，生成B样条基函数.  
   其公式为：<img width="236" height="49" alt="image" src="https://github.com/user-attachments/assets/5ba994c5-058a-4310-92b8-252c7b4c6b56" />  
   意义是：对已知区间里的N(t)函数，已经有它抛光k-1次的函数，可以这样抛光得k次函数.  
   
   [**等距B样条**](model/01_interpolation/Equidistant_B_spline.py)：节点分布均匀，基函数形式统一，计算方便.**便于程序实现.**  
   用法：输入已知数据x_list和y_list，可以得到等距B样条函数S(x)，对于任意输入的x，都可以得到y.  
   
   一维/二维等距B样条：分别处理曲线、曲面问题.  
   
   作用：数值更稳定，局部修改不影响全局.  
   解决问题场景：数据量大、稳定性要求高、需要局部调整，比普通三次样条更稳定.  
   
[**4.最小二乘拟合方法**](model/01_interpolation/L_S_FL_S_F.py)：线性最小二乘拟合方法、最小二乘拟合函数的求解、一般线性最小二乘拟合方法、非线性最小二乘拟合方法  
   特点：不一定过每个点，力求整体偏差平方和最小.不妨设y_0为原始数据，y_1为近似函数.则有S = ∑[(y_0-y_1)^2]最小.  
   解决问题场景：数据量大、噪声大的场景.  
   **具体操作**：  
   1. 借用线性主部概念.令 y_1 = ax + b.假设偏差 e = y_0 - y_1.其中，y_0是真函数，y_1是近似函数.则有S(a,b)=∑(e^2).
   2. 对所有系数求偏导，即S_a和S_b，有：<img width="257" height="128" alt="image" src="https://github.com/user-attachments/assets/88e442fd-31da-4fc0-a7eb-beb56e732b27" />

   3. 整理后可得：<img width="209" height="60" alt="image" src="https://github.com/user-attachments/assets/7e00c433-0051-446c-b83c-856fbac75d7b" />

   4. 解得a、b即可.
   5. 若要提升精度，则可设y = ax^2 + bx + c.次数可依次提高.

[**案例一 黄河小浪底调水调沙问题**](cases/01_interpolation/case_01_yellow_river.py)    
题目所求：
1. 给出估算任意时刻的排沙量及总排沙量的方法.
2. 确定排沙量与流水量的变化关系.

思路分析：  
1. 注意到，时间间隔均为12小时.则可令每个点时间为{x_i(t_i)}，排沙量{y_i}.用等距B样条插值，则可得到任意时刻的排啥量S(x) = S(x(t)).令S(x)在定义域内积分后，则可得到总排沙量.
2. 用多项式进行最小二乘拟合.

---

## 2-层次分析方法

定量与定性相结合，适用于复杂问题.多见于综合评估、制定计划、资源分配、选优排序等问题.  

**1. 层次分析的一般方法**：层次结构图、比较矩阵、相对权重向量确定（包含和、求根和特征根法）、一致性检验、计算组合权重和组合一致性检验  

   层次结构图：化繁为简，将问题分为目标层、决策层和方案层.  
   **比较矩阵**：量化工具.第一行第一列固定为1.元素i相对于元素j的重要程度为a_i_j.  
   **相对权重向量确定**：  
   1. [和法](model/02_ahp/sum_relative_weight.py). 每一列求和后，各元素和和求比值，即归一化.每一行求和.每行和归一化，得到最终权重.
   2. [求根法](model/02_ahp/root_finding_relative_weight.py). 每一行n个元素求积，开n次根，即几何平均.每行和归一化，得到最终权重.
   3. [**特征根法**](model/02_ahp/eigenvalue_relative_weight.py). 对于已有的比较矩阵A，设定长度n的单位列向量v^0.则有Av^0，取出其最大分量λ_max(即最大特征值，取收敛的.**不要归一化后的最大值**，要原矩阵的.)，令av^0每项除以k，得到v^1.递归，会收敛.收敛结果v'归一化，得到权重.

   一致性检验：解决主观判断可能带来的逻辑混乱或决策失真问题.  
   其算法为：<img width="304" height="107" alt="image" src="https://github.com/user-attachments/assets/f598a245-4231-44c7-935b-b8e64b916b8c" />  
   当 CR < 0.1 时，通过检测.  

   计算组合权重与组合一致性检验：若干权重W的组合权重为若个权重W的积，组合一致性为若干组合一致性的和.
   
**2. 一类选优排序问题**  
   一种模型.具体操作流程：  
   1. 确定准则层对目标层权重矩阵W_1,并进行一致性检测.
   2. 确定方案层对准则层权重矩阵W_2,并进行一致性检测.
   3. 计算组合权重W，并进行一致性检验.得到最终方案.

[**案例一 合理分配住房问题**](cases/02_ahp/ahp_house.py)  
题目所求：40人的合理排队次序.  
思路分析：  
   1. 给出主管评判标注，数据权重化.
   2. 计算准则间的比较矩阵A，得到准则层对目标层的权重矩阵W_1.
   3. 以单一准则,得到8个40阶方阵，计算出8个权重B_i，拼凑成方案层对准则层的权重矩阵W_2.
   4. 计算组合权重W = W_2 * W_1.其中，W_2为40行8列，W_1为8行1列.则，W为40行1列.进行组合一致性检查，组合一致性为W_1和W_2中8个B_i的合.
   5. 依照W中四十个数值的权重大小排列，完毕.

---

## 3-概率统计方法  

本章节的五个小结具有鲜明的逻辑链，依次提供了：  
   1. 概率论与数理统计理论，包括常用数学模型.
   2. 实际问题数学化.如何把现实数据变为好处理的数学数据.
   3. 如何从现实数据得到模型参数，并检查他们的合理性.
   4. 筛选有效影响因素.
   5. 研究变量和变量间关系.

**1. 概率分布与数字特征**：一维随机变量与分布函数、多维随机变量与分布函数、随机变量的数学期望与方差(数学期望、方差)  
   特点：理论基础.用来描述统计规律、提炼关键指标.  
   一维随机变量与分布函数：**分布函数**.对于某随机变量，其呈现f(t)，则分布函数<img width="199" height="31" alt="image" src="https://github.com/user-attachments/assets/5d4949e6-ba99-4f9a-9546-293bd6cc9bb1" />  
   其意义为：随机变量不超过x的概率大小.同时，f(x)需满足在正负无穷上积分为1，即归一化.  
   多为随机变量与分布函数：以二维为例，视f(x,y)为f(t)，则F(x)变为F(x,y)二重积分，计算即可.但要保证，F(x,y) = F(x)F(y)，即x、y独立.同时，f(x,y)在其定义域D是满足积分为1，即归一化.多维以此类推.  
   随机变量的数学期望与方差：  
   数学期望E：<img width="151" height="28" alt="image" src="https://github.com/user-attachments/assets/fb0b0347-34ce-4dc4-9b48-67f90b55fbbc" />，<img width="188" height="29" alt="image" src="https://github.com/user-attachments/assets/b754b57e-aa2e-445f-9a8c-bdb060efd55e" />  
   其中，x可视为一种g(x)，计算即可.  
   方差D：<img width="161" height="26" alt="image" src="https://github.com/user-attachments/assets/d4c5ee63-d557-47d1-8305-8d2dc1e0f309" />  
   其他常用分布：两点分布、二项分布、泊松分布、均匀分布、正态分布等.
   
**2. 样本与统计量**：常用统计量.  
   常用统计量：平均值、中位数、分位数、标准差、方差、极差、偏度、峰度、k阶原点矩、k阶中心矩.  
   平均值：估计总体期望.受极值影响大.  
   中位数：抗极值.适用于稳健性分析.  
   分位数：划定区域，选取合理区间.做异常检测、风险评估等.  
   方差：方差越大，越离散;方差越小，越聚集.  
   标准差：方差的平方根.与原始数据量纲一致，便于理解.  
   极差：最大值和最小值的差，作用较弱.  
   偏度：衡量左右程度.约等于0时，基本对称.若＞0，平均值＞中位数，右偏；反之，左偏.  
   丰度：衡量中心尖锐程度和尾部厚度.约等于3时，相当于正态分布.若＞3，极端值出现概率比正态分布高.  
   k阶原点矩：计算数据的加权平均，对极端值敏感.  
   k阶中心矩阵：k=1，矩为0；k＝2，矩为方差；k＝3，为偏度；k＝4，为峰度.  
   
**3. 参数估计法**：点估计法(矩估计法、最大似然估计法、估计量的评价)、区间估计法(正态总体期望的置信区间、正态总体方差的区间估计)  
   特点：模型定量化，**从定性到定量**的关键环节.能标定模型参数、评估估计精度、比较方案优劣等.  
   点估计法：给位置参数一个具体数值作为猜测.便于计算.是区间估计的基础.  
   区间估计法：给出一个区间，以一定置信水平保证真是参数落在这个区间内.  
   点估计法计算：  
   1. 矩估计法.用样本矩近似总体矩，简历方程求解参数.其中，k阶原点矩本身就有充分的含义.  
      计算：  
      1. 写出总体k阶原点矩<img width="187" height="29" alt="image" src="https://github.com/user-attachments/assets/65969033-bb31-47ba-a3b6-25b1afdc5428" />
      2. 计算样本k阶原点矩<img width="114" height="26" alt="image" src="https://github.com/user-attachments/assets/780a7d9d-963a-4212-bf2b-ff5c5082feff" />
      3. 令二者相等，得到方程组.
      4. 解方程组，得到参数估计.  
      例：<img width="489" height="416" alt="image" src="https://github.com/user-attachments/assets/6034b0d5-94bf-4eed-bde6-f3472988bd26" />  
   2. 最大似然估计法.参数θ能使观测值可能最大.  
      计算：  
      1. 构造似然函数L.其中，{X_i}是已知的.<img width="374" height="29" alt="image" src="https://github.com/user-attachments/assets/00ffc429-917e-4281-bf69-b4dfe3b9402a" />
      2. 取对数lnL.
      3. 对lnL求θ偏导，使偏导=0.
      4. 解得θ.  
      例：<img width="514" height="389" alt="image" src="https://github.com/user-attachments/assets/51d86f7a-c867-47ae-a41b-da604700a99e" />  
   3. 估计量的评价.判断点估计好坏.  
      计算：  
      1. 无偏性.在前面，我们已经通过最大似然估计发得到θ，不妨称之为θ_1.若E(θ_1) = θ_0,则无偏.
      2. 有效性.对于两个无偏估计值θ_1和θ_2，若前者的方差比后者小于1，则前者更有效.
      3. 一致性.n趋近于无穷时，θ_1趋近于θ_0，则一致.  
     
   区域估法:用于检验前者θ的精确程度、可信度.会需带入计算即可.
   
**4. 方差分析法**  
   用于检验前者算出来的参数所代表的因素有没有真实影响.可用于筛选有效因素、比较方案优劣、修正系统偏差.  
   核心参数：S_T = S_A + S_E.其中，S_T为总平方和，描述数据偏差；S_A为组件平方和，也称因素A平方和；S_E为组内平方和，也称误差平方和.  
   现在，假设我们研究的元素A有m个等级，每个等级都进行了n次测验.则设因素自由度dfA = m-1，组内自由度为dfE = m(n-1).  
   则有均方MS为<img width="170" height="49" alt="image" src="https://github.com/user-attachments/assets/59226f47-e8e1-481d-8104-ce711266c045" />  
   进而，假设统计量F为MS_A和MS_E的比值.若F很大，则认为A因素有显著影响.  
   
**5. 相关分析法**：相关系数、相关性检验  
   作用：研究两个变量X和Y间是否相关.  
   相关系数：定义协方差公式：Cov(X,Y) = E[(X - E(x)) * (Y - E(y))] = E(XY) - E(x) * E(Y).  
   则有相关性系数<img width="242" height="64" alt="image" src="https://github.com/user-attachments/assets/1d76b619-2873-47ca-8549-2d080bf8d91e" /> = <img width="108" height="47" alt="image" src="https://github.com/user-attachments/assets/7451329f-14c4-4762-960b-096ce1c0a824" />  
   其中，S_X是x方差开根.  
   r的取值有如下含义：<img width="348" height="193" alt="image" src="https://github.com/user-attachments/assets/a2bd7607-1d52-4b26-844a-51733d66702b" />  
   同时，经验上有：<img width="167" height="101" alt="image" src="https://github.com/user-attachments/assets/ed09592b-8b1f-4fbb-bb63-ba4485c2704a" />  
   相关性检验：用于检查r的真实性.  
   计算<img width="155" height="50" alt="image" src="https://github.com/user-attachments/assets/19b23507-24f7-4aee-9df9-824059f933d1" />  
   其中，n为总样本数.查表后，t若大于临界值，则显著相关.  

**案例一 足球门的危险区域问题**  
已知条件：标准球场长104m，宽69m；球门高2.44m，宽7.32m.射门球速约10m/s.  
题目所求：  
1. 球员在不同位置射门对球门的威胁度分析，绘制危险区域.
2. 在有一名守门员防守的情况下，进一步研究威胁度和危险区域.  

解题思路：
1. 核心假设：球飞向球门，呈二维正态分布.定义威胁度：<img width="195" height="48" alt="image" src="https://github.com/user-attachments/assets/97cc53ea-1a36-4305-bafa-83e9ee5d59d9" />
2. 建立坐标系：  
   1. 原点O：球门底边中点.xOy平面，地面；yOz平面，球门所在平面.
   2. <img width="320" height="323" alt="image" src="https://github.com/user-attachments/assets/62632dcd-8709-47ec-86d4-baf51512b61e" />
3. 对问题1.设射向球门所在平面yOz上一点(Y,Z)，则有<img width="298" height="55" alt="image" src="https://github.com/user-attachments/assets/f7401319-4b9f-4493-a883-da5017649751" />  
   其中，方差和球员素质k、射门距离d有关，且和射门偏角θ有关，即<img width="86" height="47" alt="image" src="https://github.com/user-attachments/assets/f667bcc5-cf89-44d0-8e91-021aaf8ca6fc" />  
   再设命中概率<img width="213" height="53" alt="image" src="https://github.com/user-attachments/assets/e9943310-96ad-4131-bc14-8d443e98ab77" />  
   回代如威胁度函数计算.球员素质k需要用数据反解.第一题完毕.
4. 对问题2.仅需修正威胁度定义<img width="224" height="43" alt="image" src="https://github.com/user-attachments/assets/3084b967-4ac6-4647-a2ff-5227f5b18440" />  

案例一是一个复杂的A类题目，笔者在此仅简单给出了思路.详细推导思路需要回归书本.  

**案例二 最优评卷问题**  

---

## 4-回归分析方法  

1. 一元线性回归
2. 多元线性回归
3. 回归模型的选择方法
4. 回归模型的正交化设计方法
5. 多重共线性与有偏估计方法

案例一 沼气的生成问题

---
