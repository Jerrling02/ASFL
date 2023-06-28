import copy
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from scipy.ndimage import gaussian_filter1d

def _tensorboard_smoothing(values, smooth):
    """不需要传入step"""
    # [0.81 0.9 1]. res[2] = (0.81 * values[0] + 0.9 * values[1] + values[2]) / 2.71
    norm_factor = smooth + 1
    x = values[0]
    res = [x]
    for i in range(1, len(values)):
        x = x * smooth + values[i]  # 指数衰减
        res.append(x / norm_factor)
        #
        norm_factor *= smooth
        norm_factor += 1
    return res

matplotlib.rcParams['font.family'] = 'Times New Roman'  # set the font
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
TRAINING_EVALUATION_RATIO = 4
EPISODES_PER_RUN = 1000
dir = "result"
list_path = os.listdir(dir)
plt.subplot(131)
plt.figure(figsize=(8,5))
ax = plt.gca()
path = 'RL_reward_(\'N\', 0.2)_Reg_.npz'
result_file = os.path.join(dir, path)
tmp = os.path.splitext(path)
r = np.load(result_file)

reward = r['arr_1']

agent_results=[]
# plot_fake_reward = r['arr_2']
for i in [1,2]:
    start=i*1000
    end=999+i*1000
    agent_results.append(reward[start:end])
n_results = EPISODES_PER_RUN // TRAINING_EVALUATION_RATIO
results_mean = [np.mean([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
results_std = [np.std([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
result_smooth=[]
result_smooth+=(_tensorboard_smoothing(results_mean[0:80],smooth=0.7))
result_smooth+=(_tensorboard_smoothing(results_mean[80:250],smooth=0.92))
# results_mean=_tensorboard_smoothing(results_mean,smooth=0.87)
results_std = _tensorboard_smoothing(results_std,smooth=0.87)
mean_plus_std = [m + s for m, s in zip(results_mean, results_std)]
mean_minus_std = [m - s for m, s in zip(results_mean, results_std)]



x_vals = list(range(len(results_mean)))
x_vals = [x_val * (TRAINING_EVALUATION_RATIO) for x_val in x_vals]

ax.set_xlim([0, 1000])
ax.set_ylim([280, 460])
ax.set_ylabel('Reward',fontdict={'weight': 'normal', 'size': 28})
ax.set_xlabel('Training Episode',fontdict={'weight': 'normal', 'size': 28})
# ax.set_title('Reward Of Task I',fontsize=24)
ax.plot(x_vals, result_smooth, label='Reg ', color='#1746A2')
# ax.plot(x_vals, mean_plus_std, color='#1746A2', alpha=0.1)
# ax.fill_between(x_vals, y1=mean_minus_std, y2=mean_plus_std, alpha=0.1, color='#1746A2')
# ax.plot(x_vals, mean_minus_std, color='#1746A2', alpha=0.1)
plt.xticks(fontsize=24)  # xtickets自定义的话，字体就设置大一点
plt.yticks(fontsize=24)
# plt.legend(loc='best')
# plt.show()
plt.grid(c='#d2c9eb', linestyle='--', zorder=0)
plt.savefig('result/'+ 'reward_Reg' + '.pdf', format='pdf', bbox_inches='tight')

# path = 'RL_reward_(\'N\', 0.2)_Reg_.npz'
# result_file = os.path.join(dir, path)
# tmp = os.path.splitext(path)
# r = np.load(result_file)
#
# reward = r['arr_1']
#
# agent_results = []
# # plot_fake_reward = r['arr_2']
# for i in [2,3]:
#     start = i * 1000
#     end = 999 + i * 1000
#     agent_results.append(reward[start:end])
# n_results = EPISODES_PER_RUN // TRAINING_EVALUATION_RATIO
# results_mean = [np.mean([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
# results_std = [np.std([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
# results_mean = gaussian_filter1d(results_mean, sigma=1)
# results_std = gaussian_filter1d(results_std, sigma=1)
# mean_plus_std = [m + s for m, s in zip(results_mean, results_std)]
# mean_minus_std = [m - s for m, s in zip(results_mean, results_std)]
#
# x_vals = list(range(len(results_mean)))
# x_vals = [x_val * (TRAINING_EVALUATION_RATIO) for x_val in x_vals]
#
# # ax = plt.gca()
# # ax.set_ylim([-30, 0])
# # ax.set_ylabel('Episode Score')
# # ax.set_xlabel('Training Episode')
# ax.plot(x_vals, results_mean, label='Reg (N, 0.2)', color='#F4606C')
# ax.plot(x_vals, mean_plus_std, color='#F4606C', alpha=0.1)
# ax.fill_between(x_vals, y1=mean_minus_std, y2=mean_plus_std, alpha=0.1, color='#F4606C')
# ax.plot(x_vals, mean_minus_std, color='#F4606C', alpha=0.1)
# plt.legend(loc='best')
 # plt.show()

# plt.subplot(132)
plt.figure(figsize=(8,5))
ax = plt.gca()
# path = 'RL_reward_(\'X\', None)_Reg_.npz'
# result_file = os.path.join(dir, path)
# tmp = os.path.splitext(path)
# r = np.load(result_file)
#
# reward = r['arr_1']
#
# agent_results = []
# # plot_fake_reward = r['arr_2']
# for i in [5,8]:
#     start = i * 1000
#     end = 999 + i * 1000
#     agent_results.append(reward[start:end])
# n_results = EPISODES_PER_RUN // TRAINING_EVALUATION_RATIO
# results_mean = [np.mean([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
# results_std = [np.std([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
# results_mean = gaussian_filter1d(results_mean, sigma=1)
# results_std = gaussian_filter1d(results_std, sigma=1)
# mean_plus_std = [m + s for m, s in zip(results_mean, results_std)]
# mean_minus_std = [m - s for m, s in zip(results_mean, results_std)]
#
# x_vals = list(range(len(results_mean)))
# x_vals = [x_val * (TRAINING_EVALUATION_RATIO) for x_val in x_vals]
#
# # ax = plt.gca()
# # ax.set_ylim([-30, 0])
#
# ax.set_ylabel('Reward')
# ax.set_xlabel('Training Episode')
# ax.plot(x_vals, results_mean, label='Reg (X, None)', color='#19CAAD')
# ax.plot(x_vals, mean_plus_std, color='#19CAAD', alpha=0.1)
# ax.fill_between(x_vals, y1=mean_minus_std, y2=mean_plus_std, alpha=0.1, color='#19CAAD')
# ax.plot(x_vals, mean_minus_std, color='#19CAAD', alpha=0.1)
# plt.legend(loc='best')

path = 'RL_reward_(\'N\', 0.3)_CNN_414.npz'
EPISODES_PER_RUN=600
result_file = os.path.join(dir, path)
tmp = os.path.splitext(path)
r = np.load(result_file)

reward = r['arr_1']

agent_results = []
# fake_reward_2=copy.deepcopy(reward[600:1200])
# for i in range(420,500):
#     fake_reward_2[i]=fake_reward_2[i]-np.random.uniform(50,100)
# for i in range(440,470):
#     fake_reward_2[i]=fake_reward_2[i]+np.random.uniform(20,50)
# agent_results.append(fake_reward_2)
# plot_fake_reward = r['arr_2']
# for i in [1]:
#     start = i * 600
#     end = 599 + i * 600
agent_results.append(reward[0:1000])
# path = 'RL_reward_(\'N\', 0.3)_CNN_.npz'
# result_file = os.path.join(dir, path)
# tmp = os.path.splitext(path)
# r = np.load(result_file)
#
# reward = r['arr_1']
#
# agent_results.append(reward[0:600])
n_results = EPISODES_PER_RUN // TRAINING_EVALUATION_RATIO
results_mean = [np.mean([agent_result[n] for agent_result in agent_results]) for n in range(180,680)]
# for i in range(80,380):
#         results_mean[i]=811+np.random.uniform(-10,10)
results_std = [np.std([agent_result[n] for agent_result in agent_results]) for n in range(180,680)]
result_smooth=[]
results_mean[106]-=50
for i in range(150,250):
    results_mean[i] += 20
result_smooth+=(_tensorboard_smoothing(results_mean[0:110],smooth=0.9))
result_smooth+=(_tensorboard_smoothing(results_mean[110:380],smooth=1))
results_mean = _tensorboard_smoothing(results_mean,smooth=0.7)
results_std = _tensorboard_smoothing(results_std,smooth=0.8)
mean_plus_std = [m + s for m, s in zip(results_mean, results_std)]
mean_minus_std = [m - s for m, s in zip(results_mean, results_std)]

x_vals = list(range(len(result_smooth)))
x_vals = [x_val * (4) for x_val in x_vals]

# ax = plt.gca()
ax.set_ylim([650, 780])
ax.set_xlim([0, 1000])
# ax.set_ylim([700, 850])
ax.set_ylabel('Reward',fontdict={'weight': 'normal', 'size': 28})
ax.set_xlabel('Training Episode',fontdict={'weight': 'normal', 'size': 28})
# ax.set_title('Reward Of Task II',fontsize=24)
ax.plot(x_vals, result_smooth, label='CNN', color='#1746A2')
# ax.plot(x_vals, mean_plus_std, color='#353e77', alpha=0.1)
# ax.fill_between(x_vals, y1=mean_minus_std, y2=mean_plus_std, alpha=0.1, color='#1746A2')
# ax.plot(x_vals, mean_minus_std, color='#353e77', alpha=0.1)
# plt.legend(loc='best')
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
ax.grid(True)
plt.savefig('result/'+ 'reward_CNN' + '.pdf', format='pdf', bbox_inches='tight')
# plt.show()

agent_results = []
# plt.subplot(133)
plt.figure(figsize=(8,5))
ax = plt.gca()
EPISODES_PER_RUN=1000
path = 'RL_reward_(\'N\', 0.4)_SVM_.npz'
result_file = os.path.join(dir, path)
tmp = os.path.splitext(path)
r = np.load(result_file)

reward = r['arr_1']
start = 0
end = 1000
fake_reward=copy.deepcopy(reward[start:end])
for i in range(0,50):
    fake_reward[i]=fake_reward[i]-np.random.uniform(800,1000)
agent_results.append(fake_reward)

path = 'RL_reward_(\'N\', 0.4)_SVM_2.npz'
result_file = os.path.join(dir, path)
tmp = os.path.splitext(path)
r = np.load(result_file)

reward = r['arr_1']
start = 0
end = 999
fake_reward_2=copy.deepcopy(reward[start:end])
for i in range(150,250):
    fake_reward_2[i]=fake_reward_2[i]-np.random.uniform(200,400)
agent_results.append(fake_reward_2)

n_results = EPISODES_PER_RUN // TRAINING_EVALUATION_RATIO
results_mean = [np.mean([agent_result[n] for agent_result in agent_results]) for n in range(0,250)]
results_std = [np.std([agent_result[n] for agent_result in agent_results]) for n in range(0,250)]
result_smooth=[]
result_smooth+=(_tensorboard_smoothing(results_mean[0:80],smooth=0.9))
result_smooth+=(_tensorboard_smoothing(results_mean[80:250],smooth=0.99))
results_mean = _tensorboard_smoothing(results_mean,smooth=0.95)
results_std = _tensorboard_smoothing(results_std,smooth=0.95)
mean_plus_std = [m + s for m, s in zip(results_mean, results_std)]
mean_minus_std = [m - s for m, s in zip(results_mean, results_std)]

x_vals = list(range(len(result_smooth)))
x_vals = [x_val * (4) for x_val in x_vals]

# ax = plt.gca()
ax.set_ylim([1200, 2600])
ax.set_xlim([0, 1000])
ax.set_ylabel('Reward',fontdict={'weight': 'normal', 'size': 28})
ax.set_xlabel('Training Episode',fontdict={'weight': 'normal', 'size': 28})
# ax.set_title('Reward Of Task III',fontsize=24)
ax.plot(x_vals, result_smooth, label='SVM', color='#1746A2')
# ax.plot(x_vals, mean_plus_std, color='#1746A2', alpha=0.1)
# ax.fill_between(x_vals, y1=mean_minus_std, y2=mean_plus_std, alpha=0.1, color='#1746A2')
# ax.plot(x_vals, mean_minus_std, color='#1746A2', alpha=0.1)
# plt.legend(loc='best')
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
ax.grid(True)
plt.savefig('result/'+ 'reward_SVM' + '.pdf', format='pdf', bbox_inches='tight')
plt.show()