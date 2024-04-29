import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick
import pandas as pd
from matplotlib.ticker import FuncFormatter
def to_percent(temp, position):
    return '%1.0f'%(10*temp)
fmt = '%1.f%%'
xticks = mtick.FormatStrFormatter(fmt)

#x=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.4]
#y1=[0.3431,0.3445,0.3493,0.3512,0.3529,0.3501,0.3463,0.3452]
#y2=[0.3482,0.3491,0.3531,0.3563,0.3617,0.3549,0.3521,0.3511]
#y1=[0.1704,0.1732,0.1734,0.1749,0.1765,0.1753,0.1742,0.1724]
#y2=[0.2975,0.3036,0.3034,0.3052,0.3076,0.3067,0.3056,0.3024]
#y1=[0.26,0.2613,0.2622,0.2616,0.2606,0.2565,0.2554,0.2550]
#y2=[0.3146,0.3161,0.3177,0.317,0.316,0.3109,0.3067,0.3062]

#x=[0,0.02,0.025,0.03,0.04,0.06,0.08]
#y1=[0.35,0.3516,0.3522,0.3526,0.3501,0.3491,0.3417]
#y2=[0.3572,0.3604,0.3605,0.3604,0.3595,0.3583,0.3515]
#y1=[0.1749,0.1756,0.1759,0.1756,0.1755,0.1754,0.1750]
#y2=[0.3054,0.3065,0.3075,0.3069,0.308,0.3074,0.3059]
#y1=[0.2565,0.2571,0.2574,0.2595,0.2594,0.2585,0.2567]
#y2=[0.31,0.3125,0.3125,0.3144,0.3132,0.3124,0.3122]

#x=[0,1.E-09,1.E-08,1.E-07,1.E-06,2.E-06]
#y1=[0.3346,0.3357,0.3435,0.3526,0.3504,0.3495]
#y2=[0.3453,0.3459,0.3525,0.3604,0.3586,0.3579]
#y1=[0.1677,0.1694,0.1746,0.1756,0.1732,0.169]
#y2=[0.2943,0.2962,0.3043,0.3069,0.3039,0.2944]
#y1=[0.2549,0.2556,0.2591,0.2621,0.2527,0.2515]
#y2=[0.3115,0.3113,0.3162,0.316,0.3095,0.3054]


x=[0,0.02,0.05,0.1,0.2,0.5]
#y1=[0.35,0.3519,0.3529,0.3509,0.3452,0.3439]
#y2=[0.3572,0.3608,0.3617,0.3608,0.3544,0.3525]
#y1=[0.1749,0.1756,0.1765,0.1754,0.1726,0.171]
#y2=[0.3054,0.3054,0.3076,0.3068,0.3025,0.299]

y1=[0.0460,0.0463,0.0473,0.0470,0.0467,0.0464]
y2=[0.0950,0.0953,0.0972,0.0970,0.0954,0.0952]
y1=[4.61,4.63,4.73,4.7,4.67,4.64]
y2=[9.5,9.53,9.72,9.70,9.54,9.52]
'''y1=[0.0423,0.0426,0.0427,0.0432,0.0440,0.0428]
y2=[0.0855,0.0858,0.0868,0.0876,0.0888,0.0870]

y1=[4.26,4.27,4.29,4.32,4.40,4.28]
y2=[8.61,8.64,8.68,8.76,8.88,8.70]'''


x_index = list(range(1,len(x)+1))
fig, ax1 = plt.subplots()
line1, = ax1.plot(x_index, y1, color=sns.xkcd_rgb["pale red"], linestyle='-', label='NDCG@10')
p1 = ax1.scatter(x_index, y1, color=sns.xkcd_rgb["pale red"], marker='s', s=40, label='NDCG@10')

ax2 = ax1.twinx()
line2, = ax2.plot(x_index, y2, color=sns.xkcd_rgb["medium green"], linestyle='-', label='HR@10')
p2 = ax2.scatter(x_index, y2, color=sns.xkcd_rgb["medium green"], marker='o', s=40, label='HR@10')

#ax1.set_xlabel("α", fontsize=10)
ax1.set_title("Toys", fontsize=16)
#ax1.set_xticks(np.array(x))

ax1.set_ylabel("NDCG@10(%)", fontsize=15)
ax1.set_yticks(np.arange(4.6,5.0,0.1))
#ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1,decimals=1))


ax2.set_ylabel("HR@10(%)", fontsize=15)
ax2.set_yticks(np.arange(9.4,9.8,0.1))
#ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1,decimals=1))

# 双Y轴标签颜色设置
ax1.yaxis.label.set_color(line1.get_color())
ax2.yaxis.label.set_color(line2.get_color())

# 双Y轴刻度颜色设置
ax1.tick_params(axis='y', colors=line1.get_color(),labelsize=18)
ax2.tick_params(axis='y', colors=line2.get_color(),labelsize=18)
ax1.tick_params(labelsize=18)
# 图例设置
ax1.grid()

plt.xticks(x_index,x)



plt.legend(handles=[p1, p2],fontsize='x-large',loc="lower center")
plt.tight_layout()
plt.show()


