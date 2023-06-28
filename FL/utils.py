# utils.py
# Utility functions
# @Author  : wwt
# @Date    : 2019-8-1

from datetime import datetime
import sys
import os
import random
import numpy as np
import torch
# import syft as sy
import matplotlib.pyplot as plt
from sklearn import datasets
import FL.FLLocalSupport as FLSup


def set_print_device(dev, f_handle=None):
    """
    # set current print device to dev
    :param dev: device name or file name
    :param f_handle: file handle
    :return: reference to the original output device (e.g., sys.__stdout__)
    """
    if dev == 'stdout':
        sys.stdout = sys.__stdout__
    elif f_handle:
        assert f_handle
        sys.stdout = f_handle


def show_settings(env_cfg, detail=False, detail_info=None):
    """
    Display experiment settings
    :param env_cfg: federated environment configuration
    :param detail: detailed env settings, round specified
    :param detail_info: detailed env settings, including (client_shard_sizes, clients_perf_vec, clients_crash_prob_vec)
    :return:
    """
    print('> Env settings')
    print('Mode: %s' % env_cfg.mode)
    print('n_clients=%d, selected_clients=%d, picked_clients=%d' % (env_cfg.n_clients, round(env_cfg.n_clients*env_cfg.pick_C),round(env_cfg.n_clients*env_cfg.pick_pct)))
    print('rounds=%d, n_local_epochs=%d, batch_size=%d, lag_tol=%d' % (env_cfg.n_rounds, env_cfg.n_epochs, env_cfg.batch_size,env_cfg.lag_tol))
    print('data_dist=%s, perf_dist=%s, crash_dist=%s' % (env_cfg.data_dist, env_cfg.perf_dist, env_cfg.crash_dist))
    if detail:
        show_env(detail_info[0], detail_info[1], detail_info[2])
    print('> Task settings')
    print('dataset: %s, task_type: %s' % (env_cfg.dataset, env_cfg.task_type))
    print('in_dim=%s, out_dim=%s, lr=%.6f' % (env_cfg.in_dim, env_cfg.out_dim, env_cfg.lr))
    print('optimizer=%s, loss_func=%s' % (env_cfg.optimizer, env_cfg.loss))


def show_env(client_shard_sizes, clients_perf_vec, clients_crash_prob_vec):
    """
    Display environment
    :param client_shard_sizes: sizes of shards
    :param clients_perf_vec: performances of clients
    :param clients_crash_prob_vec: probabilities of crash for clients
    :return:
    """
    print('> Env details')
    print('client_shard_sizes', client_shard_sizes)
    print('clients_perf_vec', clients_perf_vec)
    print('clients_crash_prob_vec', clients_crash_prob_vec)


def inspect_model(model):
    """
    inspect a pytorch model
    :param model: the model
    :return: model content
    """
    pms = []
    for param in model.parameters():
        pms.append(param.data)

    return pms


def log_stats(f_name, env_cfg, detail_env,
              time_trace,epoch_train_trace, epoch_test_trace, round_trace, acc_trace, make_trace, pick_trace, crash_trace,
              deprecate_trace, client_timers, client_futile_timers, global_timer, global_T_dist_timer,eu_ratio,
              sync_ratio, version_var, best_rd, best_loss,extra_args=None, log_loss_traces=True):
    """
    Save experiment results into a log file
    :param f_name: log file name
    :param env_cfg: federated environment configuration
    :param detail_env: detailed arguments like shards distribution
    :param epoch_train_trace: client train trace
    :param epoch_test_trace: client test trace
    :param round_trace: round trace
    :param acc_trace: accuracy trace
    :param make_trace: well-progressed clients trace
    :param pick_trace: client selection trace
    :param crash_trace: client crash trace
    :param deprecate_trace: deprecated client trace
    :param client_timers: client run time
    :param client_futile_timers: client futile run time
    :param global_timer: global run time
    :param global_T_dist_timer: global distribution time
    :param eu_ratio: Effective Update ratio
    :param sync_ratio: Sync. Ratio
    :param version_var: Version variance
    :param best_rd: round index at which best model is achieved
    :param best_loss: best model's global loss
    :param extra_args: extra arguments, for extended FL
    :param log_loss_traces: log the traces of training/test loss if True
    :return:
    """
    with open(f_name, 'a+') as f:
        set_print_device('to_file', f_handle=f)
        print('\n\n> Exp stats. at', datetime.now().strftime('%D-%H:%M'))
        show_settings(env_cfg, detail=True, detail_info=detail_env)
        #print('Clients run time:', client_timers)
        #print('Clients futile run time:', client_futile_timers)
        futile_pcts = np.array(client_futile_timers) / np.array(client_timers)
        # print('Clients futile percent (avg.=%.3f):' % np.mean(futile_pcts), futile_pcts)
        print('Clients futile percent (avg.=%.3f):' % np.mean(futile_pcts))
        print('EUR: %.6f' % eu_ratio)
        print('SR: %.6f' % sync_ratio)
        print('VV: %.6f' % version_var)
        print('Avg. round length:', global_timer/env_cfg.n_rounds)
        print('Avg. T_dist:', global_T_dist_timer/env_cfg.n_rounds)
        if log_loss_traces:
            print('> Loss traces')
            print('Client train trace:', epoch_train_trace)
            print('Client test trace:', epoch_test_trace)
        print('Round trace:', round_trace)
        print('accuracy trace:', acc_trace)
        print('> Pick&crash traces')
        #print('Make trace:', make_trace)
        #print('Pick trace:', pick_trace)
        #print('Crash trace:', crash_trace)
        #print('Deprecate trace(SAFA only):', deprecate_trace)
        print('Extra args(SAFA only):', extra_args)
        print('Best loss = %.6f at round #%d' % (best_loss, best_rd))
        print('Best accuracy:', np.max(acc_trace))
        print('time trace:',time_trace)

        # reset
        set_print_device('stdout')


def show_epoch_trace(trace, n_clients, plotting=False, cols=1):
    """
    Display the trace of training/test along epochs across rounds
    :param trace: the trace
    :param n_clients: # of clients#
    :param plotting: plot or not
    :param cols: plotting layout
    :return: na
    """
    client_traces = np.empty((n_clients, 0)).tolist()  # split into one trace per client
    for e in trace:
        # each element contains losses of n clients
        for c in range(n_clients):
            client_traces[c].append(e[c])

    print('> Showing traces')
    for c in range(n_clients):
        print('>   Client %d\'s trace: ' % c, client_traces[c])

    # plotting
    layout = (np.ceil(n_clients/cols), cols)
    if plotting:
        for c in range(n_clients):
            plt.subplot(layout[0], layout[1], c+1)  # fig no. 1 of [n*1] layout
            plt.plot(list(range(len(client_traces[c]))), client_traces[c])
            plt.title('Client %d' % c)
            plt.ylabel('Loss')
        plt.show()


def show_round_trace(trace, plotting=False, title_='XX'):
    """
    Display the trace of overall loss objective of the global model
    :param trace: the trace
    :param plotting: plot or not
    :param title_: figure title
    :return: na
    """
    print('> Showing round trace')
    print('>   ', trace)

    if plotting:
        plt.plot(list(range(len(trace))), trace)
        plt.title(title_)
        plt.xlabel('federated round #')
        plt.ylabel('Global loss')
        plt.show()


def normalize(data, expt=None):
    """
    Normalize data
    :param data: data to normalize (in np.array)
    :param expt: a list of col(s) to keep original value
    :return: normalized data
    """
    if not expt:
        return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    else:
        tmp = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))  # norm all cols
        tmp[:, expt] = data[:, expt]  # roll these cols back
        return tmp


def list_shuffle(l):
    """
    Shuffle a list
    :param l: the list
    :return: shuffled list
    """
    return random.sample(l, len(l))


def filter_matrix_symb(mat, col, value):
    """
    Filter a dataset (np.ndarray) by the value of a specified column
    :param mat:  the data matrix
    :param col:  the column
    :param value:  the value needed as the criterion
    :return:  filtered dataset
    """
    mask = []
    for r in range(len(mat)):
        if mat[r][col] == value:
            mask.append(r)  # mask this sample to keep
    return mat[mask]


def filter_matrix_value(mat, col, range_):
    """
    Filter a dataset (np.ndarray) by the range of a specified column
    :param mat:  the data matrix
    :param col:  the column
    :param range_:  the value needed as the criterion
    :return:  filtered dataset
    """
    mask = []
    for r in range(len(mat)):
        if range_[0] <= mat[r][col] <= range_[1]:
            mask.append(r)  # mask this sample to keep
    return mat[mask]


def filter_useless_cols(mat):
    """
    Remove cols that have only one value
    :param mat:  the data matrix
    :return:  filtered dataset
    """
    n_cols = mat.shape[1]
    mask = []
    for c in range(n_cols):
        if min(mat[:, c]) != max(mat[:, c]):
            mask.append(c)  # mask this col to keep
    return mat[:, mask]


def fetch_KddCup99_10pct_tcpdump(return_X_y=False):
    """
    Download KddCup99_percent10 via Scikit-learn and extract tcp-protocol samples to form a subset by filtering
    :return: tcp_data_mat, shape
    """
    X, y = datasets.fetch_kddcup99(subset=None, data_home='data/', shuffle=False, percent10=True, return_X_y=True)
    y = np.reshape(y, (-1, 1))
    data_mat = np.concatenate((X, y), axis=1)

    # tcp traces are what we need (190k total, ~40% normal)
    tcp_data_mat = filter_matrix_symb(data_mat, 1, b'tcp')
    # filter out extreme values
    tcp_data_mat = filter_matrix_value(tcp_data_mat, 4, [0, 3e4])  # src_bytes
    tcp_data_mat = filter_matrix_value(tcp_data_mat, 5, [0, 3e4])  # dst_bytes

    # filter out symbolic features
    mask = list(range(tcp_data_mat.shape[1]))
    mask.remove(1)  # symbolic field: protocol_type (e.g., tcp)
    mask.remove(2)  # symbolic field: service (e.g., http)
    mask.remove(3)  # symbolic field: flag (e.g., SF)
    tcp_data_mat = tcp_data_mat[:,mask]

    # filter out useless features
    tcp_data_mat = filter_useless_cols(tcp_data_mat)  # remove features whose stdev = 0

    # binarize labels, -1 normal, +1 abnormal
    labels = []
    for r in tcp_data_mat:
        if r[-1] == b'normal.':
            labels.append(-1)
        else:
            labels.append(1)

    # replace the label col
    tcp_data_mat = np.concatenate((tcp_data_mat[:, :-1].astype('float'), np.reshape(labels, (-1, 1))), axis=1)

    if return_X_y:
        return tcp_data_mat[:, :-1], tcp_data_mat[:, -1]
    else:
        return tcp_data_mat


def get_FL_datasets(data_train_x, data_train_y, data_test_x, data_test_y, env_cfg, clients, from_file=None):
    """
    Build federated data sets for a number of n_clients
    :param data_train_x: training data X to split
    :param data_train_y: training data Y to split
    :param data_test_x: test data X to split
    :param data_test_y: test data Y to split
    :param env_cfg: environment config file
    :param clients: client objects
    :param from_file: read a existing data partition scheme from local file instead of generating
    :return: FLFedDataset of training data, FLFedDataset of test data, and a list of sizes of shards
    """
    dev = env_cfg.device
    # device
    # data_train_x, data_train_y = data_train_x.to(dev), data_train_y.to(dev)
    # data_test_x, data_test_y = data_test_x.to(dev), data_test_y.to(dev)
    # metas
    train_size = len(data_train_x)
    test_size = len(data_test_x)
    data_size = train_size + test_size
    # prepare lists of local data shards
    client_train_data = []
    client_test_data = []
    client_shards_sizes = []

    # prepare split points
    split_points_train = [0]  # partition must start from 0
    split_points_test = [0]

    # Case 1: Even partition
    if env_cfg.data_dist[0] == 'E':
        eq_size_train = int(train_size / env_cfg.n_clients)  # even-size shards here
        eq_size_test = int(test_size / env_cfg.n_clients)  # even-size shards here
        for i in range(env_cfg.n_clients):
            split_points_train.append((i+1)*eq_size_train)
            split_points_test.append((i+1)*eq_size_test)
            # local data sizes, train/test combined
            client_shards_sizes.append(eq_size_train + eq_size_test)

    # Case 2: eXponential distribution, by partitioning with random split points
    elif env_cfg.data_dist[0] == 'X':
        rerand = True  # in case of illegal local data size
        while rerand:
            rerand = False
            client_shards_sizes = []
            # uniform split points, in percentage
            split_points_pct = np.append([0, 1], np.random.random_sample(size=env_cfg.n_clients-1))
            split_points_pct.sort()
            split_points_train = (split_points_pct * train_size).astype(int)
            split_points_test = (split_points_pct * test_size).astype(int)
            # validity check
            for i in range(env_cfg.n_clients):
                quota = split_points_train[i+1] - split_points_train[i] + split_points_test[i+1] - split_points_test[i]
                if quota < max(10, env_cfg.batch_size/2):  # check each shard size
                    rerand = True  # can't be too small
                    break
                else:
                    client_shards_sizes.append(quota)

    # Case 3: Local data sizes follow Normal distribution
    elif env_cfg.data_dist[0] == 'N':  # env_cfg.data_dist = ('N', rlt_sigma)
        mu = data_size / env_cfg.n_clients
        sigma = env_cfg.data_dist[1] * mu

        rerand = True
        while rerand:
            # directly generate sizes of shards, temporarily
            client_shards_sizes = np.random.randn(env_cfg.n_clients) * sigma + mu
            rerand = False
            # make it add up to data_size
            client_shards_sizes = client_shards_sizes * data_size / client_shards_sizes.sum()
            # validity check
            for s in client_shards_sizes:
                if s < max(20, env_cfg.batch_size):
                    rerand = True
                    break
        # now compose train and test partitions separately
        split_points_train = [0]
        last_point_train = 0
        split_points_test = [0]
        last_point_test = 0
        for s in client_shards_sizes:
            # for training
            split_points_train.append(last_point_train + int(s * env_cfg.train_pct))
            last_point_train += int(s * env_cfg.train_pct)
            # for test
            split_points_test.append(last_point_test + int(s * env_cfg.test_pct))
            last_point_test += int(s * env_cfg.test_pct)

        # round up to pre-determined sizes
        split_points_train[-1] = train_size
        split_points_test[-1] = test_size
        # recalibrate client data shards
        for i in range(env_cfg.n_clients):
            quota = split_points_train[i+1] - split_points_train[i] + split_points_test[i+1] - split_points_test[i]
            client_shards_sizes[i] = quota
        client_shards_sizes = client_shards_sizes.astype(int)

    else:
        print('Error> Invalid data distribution option')
        exit(0)


    # if from file
    if from_file:
        split_points_train = (np.loadtxt(from_file) * train_size).astype(int)
        split_points_test = (np.loadtxt(from_file) * test_size).astype(int)
        client_shards_sizes[0] = split_points_train[0] + split_points_test[0]
        for k in range(1, env_cfg.n_clients):
            train_shards = split_points_train[k] - split_points_train[k-1]
            test_shards = split_points_train[k] - split_points_train[k-1]
            client_shards_sizes.append(train_shards+test_shards)

    # split data and dispatch
    for i in range(env_cfg.n_clients):
        # prepare client data, train and test separately
        # Note: repeated tests show that barely slicing results in copying the computation graph and the entire
        # source data into n_clients pieces when sy.BaseDataset.send() is invoked, incurring excessive memory usage
        # Therefore, we use tensor.clone().requires_grad_(False) to avoid that.
        client_train_data.append(
            FLSup.FLBaseDataset(data_train_x[split_points_train[i]: split_points_train[i+1]],
                                data_train_y[split_points_train[i]: split_points_train[i+1]],dev=env_cfg.device))
        client_test_data.append(
            FLSup.FLBaseDataset(data_test_x[split_points_test[i]: split_points_test[i+1]],
                                data_test_y[split_points_test[i]: split_points_test[i+1]],dev=env_cfg.device))
        # allocate the BaseDataset to clients
        client_train_data[i].bind(clients[i])
        client_test_data[i].bind(clients[i])
    # pseudo distributed data sets
    fed_data_train = FLSup.FLFedDataset(client_train_data)
    fed_data_test = FLSup.FLFedDataset(client_test_data)

    return fed_data_train, fed_data_test, client_shards_sizes


def batch_sum_accuracy(y_hat, y, taskLoss):
    """
    Compute Accuracy = (TP+TN)/(TP+TN+FP+FN)
    :param y_hat: model decisions
    :param y: labels
    :param taskLoss: i.e., Reg, SVM, or CNN
    :return: batch_sum_acc and batch count
    """
    assert len(y) == len(y_hat)
    acc = torch.tensor(0.0)
    count = len(y)

    if taskLoss == 'mse':  # sum up (1 - relative error)
        y = y.view_as(y_hat)
        y_hat, y = y_hat.float(), y.float()
        acc += sum(1.0 - abs((y - y_hat))/torch.max(y_hat, y)).item()
    elif taskLoss == 'nllLoss':
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc += pred.eq(y.view_as(pred)).sum().item()
    elif taskLoss == 'svmLoss':
        y = y.view_as(y_hat)
        for res in y*y_hat:
            acc += torch.tensor(1.0) if res.item() > 0 else torch.tensor(0.0)

    return acc.detach().item(), count


