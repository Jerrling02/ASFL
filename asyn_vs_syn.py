import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

matplotlib.rcParams['font.family'] = 'Times New Roman'  # set the font
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42




import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator







# color_list = ['yellowgreen', 'steelblue', 'darkorange', 'dimgray']
# color_list1 =['#e600e6','#00cccb','dimgray','yellowgreen']
color_list1 = [ '#9900ff', '#AAAAAA', 'tab:blue', 'tab:green','tab:red']


def draw_double_y(n,y1,y2):

    x=range(0,n)
    marker_list=[' ', '^']
    color_list_2=['tab:blue','tab:red']
    # color_list_2=['#69b3a2',(0.2, 0.6, 0.9, 1)]
    fig = plt.figure(figsize=(8, 6))
    label_list=['Sync. accuracy','Asyn. accuracy','Sync. latency','Asyn. latency']
    ax1 = fig.add_subplot(111)
    lns=[]
    for i in range(len(y1)):
        lns1=ax1.plot(x, y1[i],
                 label=label_list[i], linewidth=2.5, marker=marker_list[i],ms=6,color=color_list_2[0],zorder=0)
        lns+=lns1
    ax1.set_ylabel('Accuracy',c=color_list_2[0],fontdict={'weight': 'normal', 'size': 28})
    ax1.set_xlabel('# of Global Rounds', fontdict={'weight': 'normal', 'size': 28})  # X轴标签

    ax2 = ax1.twinx()  # this is the important function
    for i in range(len(y2)):
        lns2=ax2.plot(x, y2[i],
                 label=label_list[i+len(y1)], linewidth=2.5, marker=marker_list[i],ms=6,color=color_list_2[1],zorder=0)
        lns += lns2
    labs = [label.get_label() for label in lns]
    ax2.legend(lns, labs,fontsize=24,loc='upper right')
    ax2.set_xlim([0, 31])
    ax2.set_ylabel('Latency (s)',c=color_list_2[1],fontdict={'weight': 'normal', 'size': 28})
    ax2.set_xlabel('# of Global Rounds')
    ax1.tick_params(axis='y', labelsize=24, labelcolor=color_list_2[0], color=color_list_2[0])
    ax1.tick_params(labelsize=24)

    ax2.tick_params(labelsize=24)
    ax2.tick_params(axis='y', labelsize=24, labelcolor=color_list_2[1], color=color_list_2[1])
    #设置图例
    plt.grid(c='#d2c9eb', linestyle='--', zorder=0)
    # 设置坐标轴线颜色
    ax2.spines["left"].set_color(color_list_2[0])  # 修改左侧颜色
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['right'].set_linewidth(1.5)
    ax2.spines["right"].set_color(color_list_2[1])  # 修改右侧颜色

    # plt.legend(loc='upper right', fontsize=24)
    plt.savefig('result/' + 'syn_vs_asyn.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def res_plot2(epoch, val_acces, type,label_list):

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams.update({'font.size':8, 'font.family': 'serif'})
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    # matplotlib.rc('font', size=24)
    print(label_list,len(label_list))
    marker_list = ['*', 'd', 'x', '>', 's', '^']
    # marker_list = [' ', ' ', ' ', ' ', ' ', ' ']
    epoches = np.arange(1, epoch+1)
    # 画出训练结果
    plt.figure(figsize=(8, 6))
    # plt.plot(epoches, val_acces[0], 'k', label=label_list[0], linewidth=2)
    for index in range(0, len(val_acces)):
        plt.plot(epoches, val_acces[index],
                 label=label_list[index], linewidth=2.5,marker=marker_list[index],ms=6, color=color_list1[index])

    plt.xlabel('# of Global Rounds', fontdict={'weight': 'normal', 'size': 28})  # X轴标签
    plt.ylabel('Accuracy', fontdict={'weight': 'normal', 'size': 28})  # Y轴标签


    plt.grid(c='#d2c9eb', linestyle='--', zorder=0)


    # cora x,y轴设置
    # plt.xticks([0, 10, 20, 30, 40, 50])
    # plt.yticks([0.2, 0.4, 0.6, 0.8])

    # citeseer x，y轴设置
    # plt.yticks([0.35, 0.4, 0.45, 0.5, 0.55])

    # pubmed x，y轴设置
    plt.xlim(1, 30)
    # 添加这部分代码
    # plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    ax=plt.gca()
    ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    ax.yaxis.get_offset_text().set_fontsize(24)  # 设置1e6的大小与位置
    plt.xticks(fontsize=24)  # xtickets自定义的话，字体就设置大一点
    plt.yticks(fontsize=24)
    plt.legend(loc='lower right',fontsize=24)


    plt.savefig('result/'+ 'diffrent_'+type + '_acc.pdf', format='pdf', bbox_inches='tight')
    plt.show()




if __name__ == '__main__':
    val_acces = []
    # cora的数据
    # '''
    label_list = ['Syn', 'Asyn']
    # acc Syn
    syn_acc = [0.5552,0.8577,0.9317,0.9459,0.9458,0.9512,0.9559,0.9586,0.9572,0.9599,0.965,0.9639,0.9642,0.9665,0.9671,0.9665,0.9694,0.9668,0.9665,0.968,0.9687,0.9691,0.9696,0.9705,0.9708,0.9694,0.9716,0.9707,0.9718,0.9713]
    syn_t = [28.00011468,48.74343967,70.37183881,90.82919717,111.7922907,134.1853685,154.5578902,175.3603923,196.6216719,217.1372647,237.7966449,258.4139555,279.478338,302.4241257,323.2056208,344.435354,365.3337569,385.4126391,405.8182828,426.4673216,446.4614804,465.9715822,487.297523,507.4772356,528.3220286,549.8770342,570.1615579,592.6685653,613.1532371,633.5472102]
    asyn_t = [15.20321035,30.80906534,51.02286577,52.02025676,57.1745007,58.21687388,59.85625815,63.24804091,75.2344389,93.02438164,101.1636598,103.491888,106.1097138,107.5493958,109.8146291,110.8961663,114.5330188,124.9214382,127.0422232,143.0249536,151.3668702,156.6852546,160.6663888,161.3609111,168.2756712,170.7974367,174.6689503,176.1566412,177.5875928,194.8327379]
    asyn_acc = [0.7335,0.7568,0.7592,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595,0.7595]

    # syn_acc_5=syn_acc[0:30:5]+[syn_acc[29]]
    # asyn_acc_5=asyn_acc[0:30:5]+[asyn_acc[29]]
    acc=[syn_acc,asyn_acc]

    # syn_t_5 = syn_t[0:30:5]+[syn_t[29]]
    # asyn_t_5 = asyn_t[0:30:5]+[asyn_t[29]]
    Time = [syn_t,asyn_t]
    # draw_double_y(30,acc,Time )
    # res_plot2(30, acc, "acc",label_list)


    
    #compare M and tau

    M_10_acc = [0.630088299, 0.566729892, 0.534911506, 0.534911506, 0.517982959, 0.517982959, 0.516511403, 0.516511403,
                0.516511403, 0.492850978, 0.492850978, 0.48292064, 0.48292064, 0.479706345, 0.479706345, 0.479706345,
                0.479706345, 0.479706345, 0.479706345, 0.479706345, 0.479706345, 0.473971813, 0.473971813, 0.473971813,
                0.473971813, 0.473971813, 0.473971813, 0.473971813, 0.473971813, 0.473971813]
    M_15_acc = [0.745048101, 0.558566385, 0.515681121, 0.439254986, 0.439254986, 0.405043429, 0.405043429, 0.386486117,
                0.386486117, 0.377509938, 0.376042188, 0.375650136, 0.375650136, 0.375650136, 0.375650136, 0.375650136,
                0.375650136, 0.375650136, 0.375108431, 0.375108431, 0.373602858, 0.373602858, 0.373602858, 0.373602858,
                0.372977644, 0.372977644, 0.372977644, 0.372977644, 0.372977644, 0.372977644]
    M_20_acc = [0.698012037, 0.564321232, 0.473723527, 0.393634866, 0.373359485, 0.369080519, 0.347015946, 0.347015946,
                0.332826333, 0.332826333, 0.316448122, 0.316448122, 0.312019431, 0.312019431, 0.305800184, 0.305800184,
                0.301964556, 0.301964556, 0.297454357, 0.297454357, 0.291976892, 0.291976892, 0.291976892, 0.291976892,
                0.291976892, 0.291976892, 0.291976892, 0.291976892, 0.291976892, 0.291976892]
    M_25_acc = [0.639810939, 0.499926008, 0.4368056, 0.380171181, 0.370290767, 0.341453798, 0.338470048, 0.31828614,
                0.316570158, 0.314498055, 0.310159206, 0.305337696, 0.305337696, 0.295565932, 0.295565932, 0.289533385,
                0.285724539, 0.281148472, 0.279099704, 0.274487919, 0.272902518, 0.266079788, 0.266079788, 0.259351763,
                0.258802991, 0.258802991, 0.251576283, 0.251576283, 0.244463847, 0.244463847]
    M_30_acc = [0.690619183, 0.526283974, 0.460167107, 0.389916138, 0.382108841, 0.357290933, 0.327853207, 0.324659774,
                0.31441078, 0.308960578, 0.290368145, 0.290368145, 0.265006409, 0.265006409, 0.240853664, 0.240853664,
                0.229282209, 0.229282209, 0.219511671, 0.219511671, 0.209896796, 0.209896796, 0.193834641, 0.193834641,
                0.184496719, 0.184496719, 0.177116851, 0.177116851, 0.170755703, 0.170755703]
    label_list = ['M=10', 'M=15', 'M=20', 'M=25', 'M=30']
    # M_10_acc_5 = M_10_acc[0:30:5] + [M_10_acc[29]]
    # M_15_acc_5 = M_15_acc[0:30:5] + [M_15_acc[29]]
    # M_20_acc_5 = M_20_acc[0:30:5] + [M_20_acc[29]]
    # M_25_acc_5 = M_25_acc[0:30:5] + [M_25_acc[29]]
    # M_30_acc_5 = M_30_acc[0:30:5] + [M_30_acc[29]]
    M_10_acc_5 = list(map(lambda item: 1 - item, M_10_acc))
    M_15_acc_5 = list(map(lambda item: 1 - item, M_15_acc))
    M_20_acc_5 = list(map(lambda item: 1 - item, M_20_acc))
    M_25_acc_5 = list(map(lambda item: 1 - item, M_25_acc))
    M_30_acc_5 = list(map(lambda item: 1 - item, M_30_acc))
    M_acc = [M_10_acc_5, M_15_acc_5,M_20_acc_5,M_25_acc_5,M_30_acc_5]
    # M_acc = [M_10_acc, M_15_acc, M_20_acc, M_25_acc, M_30_acc]

    tau_1_acc = [0.659970694, 0.548081177, 0.458049912, 0.380701741, 0.362574702, 0.331541723, 0.329316921, 0.299474566,
                 0.299474566, 0.280957804, 0.280957804, 0.267672512, 0.267672512, 0.267672512, 0.267672512, 0.257547509,
                 0.257547509, 0.244286442, 0.244286442, 0.22868435, 0.22868435, 0.21360413, 0.21360413, 0.210897108,
                 0.210897108, 0.204945018, 0.204945018, 0.191334892, 0.191334892, 0.179964759]
    tau_3_acc = [0.639810939, 0.499926008, 0.4368056, 0.380171181, 0.370290767, 0.341453798, 0.338470048, 0.31828614,
                 0.316570158, 0.314498055, 0.310159206, 0.305337696, 0.305337696, 0.295565932, 0.295565932, 0.289533385,
                 0.285724539, 0.281148472, 0.279099704, 0.274487919, 0.272902518, 0.266079788, 0.266079788, 0.259351763,
                 0.258802991, 0.258802991, 0.251576283, 0.251576283, 0.244463847, 0.244463847]
    tau_5_acc = [0.667827544, 0.541161615, 0.40065231, 0.396941052, 0.349428409, 0.335515843, 0.323423806, 0.323423806,
                 0.314722147, 0.314722147, 0.307929943, 0.307929943, 0.301996482, 0.301996482, 0.297071077, 0.297071077,
                 0.29217864, 0.29217864, 0.287127627, 0.287127627, 0.282680644, 0.282680644, 0.282680644, 0.279598001,
                 0.279598001, 0.275893062, 0.275893062, 0.272573132, 0.272198029, 0.270344841]
    tau_7_acc = [0.737057578, 0.638923886, 0.554396521, 0.490993003, 0.445356718, 0.421727278, 0.413705779,
                  0.403587844, 0.403587844, 0.397237956, 0.397237956, 0.393757892, 0.393757892, 0.393757892,
                  0.393757892, 0.393757892, 0.393757892, 0.393757892, 0.393757892, 0.393757892, 0.393757892,
                  0.393757892, 0.393757892, 0.393226967, 0.393226967, 0.393198222, 0.393198222, 0.393087623,
                  0.393087623, 0.393087623]
    tau_9_acc = [0.688106258, 0.636275917, 0.51578058, 0.491085393, 0.45337053, 0.452090471, 0.431694256, 0.431694256,
                  0.42201668, 0.42201668, 0.417043249, 0.405961975, 0.405961975, 0.405961975, 0.405961975, 0.405961975,
                  0.405961975, 0.405961975, 0.405961975, 0.405045601, 0.405045601, 0.405045601, 0.405045601,
                  0.405045601, 0.405045601, 0.405045601, 0.405045601, 0.405045601, 0.405045601, 0.405045601]
    label_list1 = ['$\\tau$=1', '$\\tau$=3', '$\\tau$=5', '$\\tau$=7', '$\\tau$=9']
    # tau_1_acc_5 = tau_1_acc[0:30:5] + [tau_1_acc[29]]
    # tau_3_acc_5 = tau_3_acc[0:30:5] + [tau_3_acc[29]]
    # tau_5_acc_5 = tau_5_acc[0:30:5] + [tau_5_acc[29]]
    # tau_7_acc_5 = tau_7_acc[0:30:5] + [tau_7_acc[29]]
    # tau_9_acc_5 = tau_9_acc[0:30:5] + [tau_9_acc[29]]

    tau_1_acc_5 = list(map(lambda item: 1-item, tau_1_acc))
    tau_3_acc_5 = list(map(lambda item: 1-item, tau_3_acc))
    tau_5_acc_5 = list(map(lambda item: 1-item, tau_5_acc))
    tau_7_acc_5 = list(map(lambda item: 1-item, tau_7_acc))
    tau_9_acc_5 = list(map(lambda item: 1-item, tau_9_acc))
    tau_acc = [tau_1_acc_5, tau_3_acc_5, tau_5_acc_5, tau_7_acc_5, tau_9_acc_5]
    # tau_acc = [tau_1_acc, tau_3_acc, tau_5_acc, tau_7_acc, tau_9_acc]

    # res_plot2(30, M_acc, "M", label_list)


    res_plot2(30, tau_acc, "tau", label_list1)

    M_time=[5777.691891,6420.792704,6596.475717,6737.360264, 6831.511583108425]
    tau_time=[]
    
    
    
    