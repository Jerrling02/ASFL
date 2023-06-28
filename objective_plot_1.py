#coding=utf-8
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import ticklabel_format


def draw_comm_reduce2(data, x_label):
    matplotlib.rcParams['font.family'] = 'Times New Roman'  # set the font
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    hatch_list = ['/', '+', 'x', '\\']
    color_list = ['yellowgreen', 'steelblue', 'darkorange', 'dimgray']

    plt.plot(1)

    plt.grid(c='#d2c9eb', linestyle='--', zorder=0)

    x_len = np.arange(len(x_label))
    # total_width, n = 0.9, 3
    # width = total_width / n
    # xticks 就是三个柱子的开始位置
    # xticks = x_len - (total_width - width) / 2

    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    bar_width = x_len / len(data)  # 设置条形的宽度
    for i in range(len(data)):
        plt.bar(x_label[i], height=data[i], color=color_list[i],
                width=bar_width*0.9, edgecolor='white', hatch=hatch_list[i], zorder=2)

    # 折线图
    plt.plot(x_label, data, "dimgray", marker='o', ms=8, label="a")

    # 为两条坐标轴设置名称
    # plt.xticks(range(len(hatch_par)), hatch_par, fontsize=32)
    plt.ylabel('Objective Values', fontdict={'weight': 'normal', 'size': 28})
    plt.xticks(fontsize=28)  # xtickets自定义的话，字体就设置大一点
    plt.yticks(fontsize=24)
    plt.ylim(0, 0.8)  # cora:0.3, citeseer:0.1, pubmed:0.25
    plt.savefig('result/objective_N.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def draw_runtime(comp_list, comm_list, x_label):

    matplotlib.rcParams['font.family'] = 'Times New Roman'  # set the font
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42


    # 定义柱形的宽度
    x_len = np.arange(len(x_label))
    bar_width = x_len / len(comp_list)  # 设置条形的宽度

    b1 = plt.bar(x_label, comp_list, 0.35, zorder=10)

    b2 = plt.bar(x_label, comm_list, 0.35, bottom=comp_list, zorder=10)

    plt.legend([b1, b2], ['computation', 'communication'],
               fontsize=24, loc='best')

    plt.xticks(rotation=20)
    # plt.xticks(range(len(hatch_par)), hatch_par, fontsize=32)
    plt.ylabel('Objective Values', fontdict={'weight': 'normal', 'size': 28})
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=24)
    plt.grid(c='#d2c9eb', linestyle='--', zorder=0)
    # plt.ylim(0, 1.6)  # cora:0.3, citeseer:0.1, pubmed:0.25
    # plt.savefig('results/' + dataset_name + '_runtime_ratio.pdf', format='pdf', bbox_inches='tight')
    plt.show()

dataset_name = 'cora'  # "cora", "citeseer", "pubmed"

if __name__ == '__main__':
    # cora communication reduce
    data = [0.689792753, 0.668314378, 0.637401019]
    x_label = ['N2', 'N3', 'N4']

    draw_comm_reduce2(data, x_label)

    # cora runtime
    # comp_list = [2786569, 2763246, 3349782, 2599782]
    # comm_list = [1054577.63671875, 768524.69921875, 226550.29296875, 456701.66015625]
    #
    # x_label = ['Random  ', 'Adaptive', 'DIGEST', 'FGL-CICS']
    # base_total = comp_list[2] + comm_list[2]
    # comp_list = [item/base_total for item in comp_list]
    # comm_list = [item/base_total for item in comm_list]
    # draw_runtime(comp_list, comm_list, x_label)