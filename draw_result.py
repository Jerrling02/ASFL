import numpy as np
import matplotlib.pyplot as plt
import os

dir = "result"
list_path = os.listdir(dir)


def get_color_list(length):
    color_list = [
        '#19CAAD',
        '#8CC7B5',
        '#A0EEE1',
        '#BEE7E9',
        '#BEEDC7',
        '#D6D5B7',
        '#D1BA74',
        '#E6CEAC',
        '#ECAD9E',
        '#F4606C'
    ]
    start = 0
    ret_color_list = []
    for i in range(length):
        ret_color_list.append(color_list[start])
        start += 3
        start %= len(color_list)
    return ret_color_list


plot_color = get_color_list(len(list_path))


# for path, color in zip(list_path, plot_color):
#     result_file = os.path.join(dir, path)
#     tmp = os.path.splitext(path)
#     r = np.load(result_file)
#     if path == 'RL_reward_(\'N\', 0.2)_Reg_.npz':
#         max_episodes = 1
#         max_ep_step = 50
#         gap = 1
#         start = 0
#         end = -1
#         plot_reward = r['arr_1']
#         # plot_reward = r['arr_2']
#         np.savetxt('./result/reward_2.csv',plot_reward , delimiter=",")

for path, color in zip(list_path, plot_color):
    if path == 'RL_reward_(\'N\', 0.3)_CNN_414.npz':
        result_file = os.path.join(dir, path)
        tmp = os.path.splitext(path)
        r = np.load(result_file)

        plot_reward = r['arr_1']

        plot_fake_reward = r['arr_2']
        # start =2000
        # end = 2999
        # plot_reward = plot_reward[start:end]
        # # plot_reward=plot_reward[0:500]
        # plot_fake_reward = plot_fake_reward[start:end]
        # acc_result=[]
        # time_result=[]
        # for i in range(100):
        #     start=(800+i)*50-50
        #     end=(800+i)*50
        #     plot_reward1=plot_reward[start:end]
        #     # # plot_reward=plot_reward[0:500]
        #     plot_fake_reward1 = plot_fake_reward[start:end]
        #     acc_result.append(plot_reward1)
        #     time_result.append(plot_fake_reward1)
        # result=np.vstack((acc_result, time_result))
        # np.savetxt("./result/acc_time.csv", result, delimiter=',')
        print(plot_reward)
        plot_x = range(len(plot_reward))
        plt.subplot(2, 1, 1)
        plt.title("reward")
        plt.plot(plot_x, plot_reward, label=tmp[0], color=color)
        plt.legend()
        # plt.show()


        # lis1 = []
        # for j in plot_fake_reward:
        #     if j != 0:
        #         lis1.append(j)
        print("plot_fake_reward: ", plot_fake_reward)
        plot_x1 = range(len(plot_fake_reward))
        plt.subplot(2, 1, 2)
        plt.title("fake reward")
        plt.plot(plot_x1, plot_fake_reward, label=tmp[0], color=color)
        plt.legend()
        plt.show()



# for path, color in zip(list_path, plot_color):
#     if path == 'RL_0.5_logistic_20client.npz':
#         result_file = os.path.join(dir, path)
#         tmp = os.path.splitext(path)
#         r = np.load(result_file)
#
#         max_episodes = 1
#         max_ep_step = 50
#         gap = 1
#         start = 0
#         end = -1
#         plot_x = r['arr_0']
#         plot_acc = r['arr_1']
#         plot_cost = r['arr_2']
#         plot_reward = r['arr_3']
#
#         plot_x = plot_x[start:end:gap]
#         plot_acc = plot_acc[start:end:gap]
#         plot_cost = plot_cost[start:end:gap]
#         plot_reward = plot_reward[start:end:gap]
#
#         if path == "RL.npz":
#             if plot_x.size >= max_episodes * max_ep_step:
#                 start_ = plot_x.size - plot_x.size % max_ep_step - max_episodes * max_ep_step
#                 end_ = plot_x.size - plot_x.size % max_ep_step
#                 plot_x = plot_x[0:max_episodes*max_ep_step:gap]
#                 plot_acc = plot_acc[start_:end_:gap]
#                 plot_cost = plot_cost[start_:end_:gap]
#                 plot_reward = plot_reward[start_:end_:gap]
#
#         # plot_acc = -np.log(-(plot_acc - 1))
#
#         plt.subplot(311)
#         plt.title("reward")
#         plt.plot(plot_x, plot_reward, label=tmp[0], color=color)
#         plt.legend()
#
#         plt.subplot(312)
#         plt.title("accuracy")
#         plt.plot(plot_x, plot_acc, label=tmp[0], color=color)
#         plt.legend()
#
#         plt.subplot(313)
#         plt.title("communication cost")
#         plt.plot(plot_x, plot_cost, label=tmp[0], color=color)
#         plt.legend()
#         plt.show()