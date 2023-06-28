import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'  # set the font
plt.rcParams.update({'font.size':8, 'font.family': 'serif'})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

"""生成数据并设置绘图参数"""
x=range(10,35,5)
x2=range(1,11,2)
M_time=[5777.691891,6420.792704,6596.475717,6737.360264, 6831.511583108425]
tau_time=[6835.695459,6803.517247,6706.639754,6610.682458,6578.078698]

# 设置两种绘图颜色
c1 = 'tab:red'
c2 = 'tab:blue'
# 设置字体大小
fontsize = 12
# 设置画布大小
width, height = 16, 14  # 单位为cm；因为保存图片时使用 bbox_inches = 'tight' 可能使图片尺寸略微放大，所以此处宽度设置得略小
# 设置刻度线在坐标轴内
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
"""绘图"""
lns = []  # 用于存储绘图句柄以合并图例的list
# 创建画布并设置大小
fig = plt.figure(figsize=(8, 6))

# fig.set_size_inches(width / 2.54, height / 2.54)  # 因为画布输入大小为厘米，此处需转换为英寸，所以除以2.54
# 通过 add_subplot 方式创建两个坐标轴，相当于在同一个子图上叠加了两对坐标系
ax = fig.add_subplot(111, label="1")
ax2 = fig.add_subplot(111, label="2", frame_on=False)
# 绘制图1并将绘图句柄返回，以便添加合并图例
lns1 = ax.plot(x, M_time, color=c1, linewidth=2.5,marker='.',ms=8,label="M",zorder=5)
ax.tick_params(labelsize=24)

lns = lns1
lns2 = ax2.plot(x2, tau_time, color=c2,linewidth=2.5,marker='.',ms=8, label="tau",zorder=5)
ax2.tick_params(labelsize=24)
lns += lns2
"""图形美化"""
# 调整第二对坐标轴的label和tick位置，以实现双X轴双Y轴效果
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.xaxis.set_label_position('top')
ax2.yaxis.set_label_position('right')
# 设置坐标轴标注
ax.set_xlabel("# of Picked devices", color=c1, fontdict={'weight': 'normal', 'size': 28})
ax.set_ylabel("Latency", color=c1, fontdict={'weight': 'normal', 'size': 28})
ax2.set_xlabel('Different Lag Tolerance', color=c2, fontdict={'weight': 'normal', 'size': 28})
ax2.set_ylabel('Latency', color=c2, fontdict={'weight': 'normal', 'size': 28})

# 设置图表标题
# fig.suptitle("Title", fontsize=fontsize + 2)
# 设置坐标轴刻度颜色
ax.tick_params(axis='x', colors=c1)
ax.tick_params(axis='y', colors=c1)

ax2.tick_params(axis='x', colors=c2)
ax2.tick_params(axis='y', colors=c2)
ax2.set_xticks(range(1,11,2))

# 设置坐标轴线颜色
ax.spines["left"].set_color(c1)  # 修改左侧颜色
ax.spines["right"].set_color(c2)  # 修改右侧颜色
ax.spines["top"].set_color(c2)  # 修改上边颜色
ax.spines["bottom"].set_color(c1)  # 修改下边颜色

# plt.legend(loc='upper right',fontsize=24)
# 添加图例
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc=0, fontsize=fontsize)
plt.grid(c='#d2c9eb', linestyle='--', zorder=0)
plt.savefig('result/'+ 'diffrent_M_tau_time.pdf', format='pdf', bbox_inches='tight')
plt.tight_layout()
plt.show()