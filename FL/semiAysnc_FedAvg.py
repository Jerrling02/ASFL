# primal_FedAvg.py
# Pytorch+PySyft implementation of the semi-asynchronous Federated Averaging (SAFA) algorithm for Federated Learning,
# proposed by: wwt
# @Author  : wwt
# @Date    : 2019-8-1


import torch
import torch.nn as nn
import torch.optim as optim
import copy
import sys
import random
import os
import math
import numpy as np
# import syft as sy
from learning_tasks import svmLoss
import utils
import FLLocalSupport as FLSup


def get_cross_rounders(clients_est_round_T_train, max_round_interval):
    cross_rounder_ids = []
    for c_id in range(len(clients_est_round_T_train)):
        if clients_est_round_T_train[c_id] > max_round_interval:
            cross_rounder_ids.append(c_id)
    return cross_rounder_ids


def sort_ids_by_perf_desc(id_list, perf_list):
    """
    Sort a list of client ids according to their performance, in an descending ordered
    :param id_list: a list of client ids to sort
    :param perf_list: full list of all clients' perf
    :return: sorted id_list
    """
    # make use of a map
    cp_map = {}  # client-perf-map
    for id in id_list:
        cp_map[id] = perf_list[id]  # build the map with corresponding perf
    # sort by perf
    sorted_map = sorted(cp_map.items(), key=lambda x: x[1], reverse=True)  # a sorted list of tuples
    sorted_id_list = [sorted_map[i][0] for i in range(len(id_list))]  # extract the ids into a list
    return sorted_id_list


def select_clients_CFCFM(make_ids, last_round_pick_ids, clients_perf_vec, cross_rounders, quota):
    """
    Select clients to aggregate their models according to Compensatory First-Come-First-Merge principle.
    :param make_ids: ids of clients finishing their training this round
    :param last_round_pick_ids: ids of clients picked last round, low priority in this round
    :param clients_perf_vec: global clients' performances
    :param cross_rounders: ids of clients inherently cannot finish before max round interval
    :param quota: number of clients to draw this round
    :return: ids of selected clients
    """
    picks = []
    # leave cross-rounders undrafted
    in_time_make_ids = [m_id for m_id in make_ids if m_id not in cross_rounders]  # in-time make ids
    high_priority_ids = [h_id for h_id in in_time_make_ids if h_id not in last_round_pick_ids]  # compensatory priority
    low_priority_ids = [l_id for l_id in in_time_make_ids if l_id in last_round_pick_ids]
    print(high_priority_ids)
    print(low_priority_ids)
    # case 0: clients finishing in time not enough for fraction C, just gather them all
    if len(in_time_make_ids) <= quota:  # if not enough well-progress clients to meet the quota
        return copy.deepcopy(in_time_make_ids)
    # case 1: # of priority ids > quota
    if len(high_priority_ids) >= quota:
        sorted_priority_ids = sort_ids_by_perf_desc(high_priority_ids, clients_perf_vec)
        picks = sorted_priority_ids[0:int(quota)]
    # case 2: # of priority ids <= quota
    # the rest are picked by order of performance ("FCFM"), lowest batch overhead first
    else:
        picks += high_priority_ids  # they have priority
        # FCFM
        sorted_low_priority_ids = sort_ids_by_perf_desc(low_priority_ids, clients_perf_vec)
        for i in range(min(quota - len(picks), len(sorted_low_priority_ids))):
            picks.append(sorted_low_priority_ids[i])

    return picks


def train(models, picked_ids, env_cfg, cm_map, fdl, last_loss_rep, verbose=True):
    """
    Execute one EPOCH of training process of any machine learning model on all clients
    :param models: a list of model prototypes corresponding to clients
    :param picked_ids: participating client indices for local training
    :param env_cfg: environment configurations
    :param cm_map: the client-model map, as a dict
    :param fdl: an instance of FederatedDataLoader
    :param last_loss_rep: loss reports in last run
    :param verbose: display batch progress or not.
    :return: epoch training loss of each client, batch-summed
    """
    dev = env_cfg.device
    if len(picked_ids) == 0:  # no training happens
        return last_loss_rep
    # extract settings
    n_models = env_cfg.n_clients  # # of clients
    # initialize loss report, keep loss tracks for idlers, clear those for participants
    client_train_loss_vec = last_loss_rep
    for id in picked_ids:
        client_train_loss_vec[id] = 0.0
    # Disable printing
    if not verbose:
        sys.stdout = open(os.devnull, 'w')
    # initialize training mode
    for m in range(n_models):
        models[m].train()

    # Define loss based on task
    if env_cfg.loss == 'mse':  # regression task
        loss_func = nn.MSELoss(reduction='mean')  # cannot back-propagate with 'reduction=sum'
    elif env_cfg.loss == 'svmLoss':  # SVM task
        loss_func = svmLoss(reduction='mean')  # self-defined loss, have to use default reduction 'mean'
    elif env_cfg.loss == 'nllLoss':  # CNN mnist task
        loss_func = nn.NLLLoss()

    # one optimizer for each model (re-instantiate optimizers to clear any possible momentum
    optimizers = []
    for i in range(n_models):
        if env_cfg.optimizer == 'SGD':
            optimizers.append(optim.SGD(models[i].parameters(), lr=env_cfg.lr))
        elif env_cfg.optimizer == 'Adam':
            optimizers.append(optim.Adam(models[i].parameters(), lr=env_cfg.lr))
        else:
            print('Err> Invalid optimizer %s specified' % env_cfg.optimizer)

    # begin an epoch of training
    for batch_id, (inputs, labels, client) in enumerate(fdl):
        inputs, labels = inputs.to(dev), labels.to(dev)  # data to device
        model_id = cm_map[client.id]  # locate the right model index
        # neglect non-participants
        if model_id not in picked_ids:
            continue
        # mini-batch GD
        print('\n> Batch #', batch_id, 'on', client.id)
        print('>   model_id = ', model_id)

        # ts = time.time_ns() / 1000000.0  # ms
        model = models[model_id]
        optimizer = optimizers[model_id]
        # gradient descent procedure
        optimizer.zero_grad()
        y_hat = model(inputs)
        # loss
        loss = loss_func(y_hat, labels)
        loss.backward()
        # weights
        optimizer.step()
        # te = time.time_ns() / 1000000.0   # ms
        # print('> T_batch = ', te-ts)

        # display
        print('>   batch loss = ', loss.item())  # avg. batch loss
        client_train_loss_vec[model_id] += loss.detach().item()*len(inputs)  # sum up

    # Restore printing
    if not verbose:
        sys.stdout = sys.__stdout__
    # end an epoch-training - all clients have traversed their own local data once
    return client_train_loss_vec


def local_test(models, picked_ids, env_cfg,  cm_map, fdl, last_loss_rep):
    """
    Evaluate client models locally and return a list of loss/error
    :param models: a list of model prototypes corresponding to clients
    :param env_cfg: environment configurations
    :param picked_ids: selected client indices for local training
    :param cm_map: the client-model map, as a dict
    :param fdl: an instance of FederatedDataLoader
    :param last_loss_rep: loss reports in last run
    :return: epoch test loss of each client, batch-summed
    """
    if not picked_ids:  # no training happens
        return last_loss_rep
    dev = env_cfg.device
    # initialize loss report, keep loss tracks for idlers, clear those for participants
    client_test_loss_vec = last_loss_rep
    for id in picked_ids:
        client_test_loss_vec[id] = 0.0
    # Define loss based on task
    if env_cfg.loss == 'mse':  # regression task
        loss_func = nn.MSELoss(reduction='sum')
    elif env_cfg.loss == 'svmLoss':  # SVM task
        loss_func = svmLoss(reduction='sum')
    elif env_cfg.loss == 'nllLoss':  # CNN mnist task
        loss_func = nn.NLLLoss(reduction='sum')
        is_cnn = True

    # initialize evaluation mode
    for m in range(env_cfg.n_clients):
        models[m].eval()
    # local evaluation, batch-wise
    acc = 0.0
    count = 0.0
    with torch.no_grad():
        for batch_id, (inputs, labels, client) in enumerate(fdl):
            inputs, labels = inputs.to(dev), labels.to(dev)  # data to device
            model_id = cm_map[client.id]  # locate the right model index
            # neglect non-participants
            if model_id not in picked_ids:
                continue
            model = models[model_id]
            # inference
            y_hat = model(inputs)
            # loss
            loss = loss_func(y_hat, labels)
            client_test_loss_vec[model_id] += loss.detach().item()
            # accuracy
            b_acc, b_cnt = utils.batch_sum_accuracy(y_hat, labels, env_cfg.loss)
            acc += b_acc
            count += b_cnt

    print('> acc = %.6f' % (acc / count))
    return client_test_loss_vec


def global_test(model, env_cfg, cm_map, fdl):
    """
    Testing the aggregated global model by averaging its error on each local data
    :param model: the global model
    :param env_cfg: environment configurations
    :param cm_map: the client-model map, as a dict
    :param fdl: FederatedDataLoader
    :return: global model's loss on each client (as a vector), accuracy
    """
    dev = env_cfg.device
    test_sum_loss_vec = [0 for i in range(env_cfg.n_clients)]
    # Define loss based on task
    if env_cfg.loss == 'mse':  # regression task
        loss_func = nn.MSELoss(reduction='sum')
    elif env_cfg.loss == 'svmLoss':  # SVM task
        loss_func = svmLoss(reduction='sum')
    elif env_cfg.loss == 'nllLoss':  # CNN mnist task
        loss_func = nn.NLLLoss(reduction='sum')
        is_cnn = True

    # initialize evaluation mode
    print('> global test')
    model.eval()
    # local evaluation, batch-wise
    acc = 0.0
    count = 0
    for batch_id, (inputs, labels, client) in enumerate(fdl):
        inputs, labels = inputs.to(dev), labels.to(dev)  # data to device
        model_id = cm_map[client.id]
        # inference
        y_hat = model(inputs)
        # loss
        loss = loss_func(y_hat, labels)
        test_sum_loss_vec[model_id] += loss.detach().item()
        # compute accuracy
        b_acc, b_cnt = utils.batch_sum_accuracy(y_hat, labels, env_cfg.loss)
        acc += b_acc
        count += b_cnt

    print('>   acc = %.6f' % (acc/count))
    return test_sum_loss_vec, acc/count


def init_cache(glob_model, env_cfg):
    """
    Initiate cloud cache with the global model
    :param glob_model:  initial global model
    :param env_cfg:  env config
    :return: the cloud cache
    """
    cache = []
    for i in range(env_cfg.n_clients):
        cache.append(copy.deepcopy(glob_model))
    return cache


def update_cloud_cache(cache, models, the_ids):
    """
    Update the model cache residing on the cloud, it contains the latest non-aggregated models
    :param cache: the model cache
    :param models: latest local model set containing picked, undrafted and deprecated models
    :param the_ids: ids of clients to update cache
    :return:
    """
    # use deepcopy to decouple cloud cache and local models
    for id in the_ids:
        cache[id] = copy.deepcopy(models[id])


def update_cloud_cache_deprecated(cache, global_model, deprecated_ids):
    """
    Update entries of those clients lagging too much behind with the latest global model
    :param cache: the model cache
    :param global_model: the aggregated global model
    :param deprecated_ids: ids of clients to update cache
    :return:
    """
    # use deepcopy to decouple cloud cache and local models
    for id in deprecated_ids:
        cache[id] = copy.deepcopy(global_model)


def get_versions(ids, versions):
    """
    Show versions of specified clients, as a dict
    :param ids: clients ids
    :param versions: versions vector of all clients
    :return:
    """
    cv_map = {}
    for id in ids:
        cv_map[id] = versions[id]

    return cv_map


def update_versions(versions, make_ids, rd):
    """
    Update versions of local models that successfully perform training in the current round
    :param versions: version vector
    :param make_ids: well-progressed clients ids
    :param rd: round number
    :return: na
    """
    for id in make_ids:
        versions[id] = rd


def version_filter(versions, the_ids, base_v, lag_tolerant=1):
    """
    Apply a filter to client ids by checking their model versions. If the version is lagged behind the latest version
    (i.e., round number) by a number > lag_tolarant, then it will be filtered out.
    :param versions: client versions vector
    :param the_ids: client ids to check version
    :param base_v: latest base version
    :param lag_tolerant: maximum tolerance of version lag
    :return: non-straggler ids, deprecated ids
    """
    good_ids = []
    deprecated_ids = []
    for id in the_ids:
        if base_v - versions[id] <= lag_tolerant:
            good_ids.append(id)
        else:  # stragglers
            deprecated_ids.append(id)

    return good_ids, deprecated_ids


def distribute_models(global_model, models, make_ids):
    """
    Distribute the global model
    :param global_model: aggregated global model
    :param models: local models
    :param make_ids: ids of clients that will replace their local models with the global one
    :return:
    """
    for id in make_ids:
        models[id] = copy.deepcopy(global_model)


def safa_aggregate(models, picked_ids,local_shards_sizes, data_size):
    """
    The function implements aggregation step (Semi-Async. FedAvg), allowing cross-round com
    :param models: a list of local models
    :param picked_ids: selected client indices for local training
    :param local_shards_sizes: a list of local data sizes, aligned with the orders of local models, say clients.
    :param data_size: total data size
    :return: a global model
    """
    print('>   Aggregating (SAFA)...')
    # shape the global model
    global_model = copy.deepcopy(models[0])
    global_model_params = global_model.state_dict()
    for pname, param in global_model_params.items():
        global_model_params[pname] = 0.0
    # client_weights_vec = np.array(local_shards_sizes) / data_size  # client weights (i.e., n_k / n)
    # for m in range(len(models)):  # for each local model
    #     for pname, param in models[m].state_dict().items():
    #         global_model_params[pname] += param.data * client_weights_vec[m]  # sum up the corresponding param
    round_data_size = 0
    for id in picked_ids:
        round_data_size += local_shards_sizes[id]
    client_weights_vec = np.array(local_shards_sizes) / round_data_size  # client weights (i.e., n_k / n)
    for m in picked_ids:  # for each local model
        for pname, param in models[m].state_dict().items():
            global_model_params[pname] += param.data * client_weights_vec[m]  # sum up the corresponding param
    # load state dict back
    global_model.load_state_dict(global_model_params)
    return global_model
    # print('>   Aggregating (SAFA)...')
    # # shape the global model
    # global_model = copy.deepcopy(models[0])
    # global_model_params = global_model.state_dict()
    # for pname, param in global_model_params.items():
    #     global_model_params[pname] = 0.0
    # client_weights_vec = np.array(local_shards_sizes) / data_size  # client weights (i.e., n_k / n)
    # for m in range(len(models)):  # for each local model
    #     for pname, param in models[m].state_dict().items():
    #         global_model_params[pname] += param.data * client_weights_vec[m]  # sum up the corresponding param
    # # load state dict back
    # global_model.load_state_dict(global_model_params)
    # return global_model
    # print('>   Aggregating (SAFA)...')
    # global_model_params = []
    # client_weights_vec = np.array(local_shards_sizes) / data_size  # client weights (i.e., n_k / n)
    # for m in range(len(models)):  # for each local model
    #     p_pointer = 0
    #     for param in models[m].parameters():
    #         if m == 0:  # use the first model to shape the global one
    #             global_model_params.append(param.data * client_weights_vec[m])
    #         else:
    #             global_model_params[p_pointer] += param.data * client_weights_vec[m]  # sum up the corresponding param
    #         p_pointer += 1
    #
    # # create a global model instance for return
    # global_model = copy.deepcopy(models[0])
    # p_pointer = 0
    # for param in global_model.parameters():
    #     param.data.copy_(global_model_params[p_pointer])
    #     p_pointer += 1
    # return global_model


def run_FL_SAFA(env_cfg, glob_model, cm_map, data_size, fed_loader_train, fed_loader_test, client_shard_sizes,
                clients_perf_vec, clients_crash_prob_vec, crash_trace, progress_trace, response_time_limit, lag_t=1):
    """
    Run FL with SAFA algorithm
    :param env_cfg: environment config
    :param glob_model: the global model
    :param cm_map: client-model mapping
    :param data_size: total data size
    :param fed_loader_train: federated training set
    :param fed_loader_test: federated test set
    :param client_shard_sizes: sizes of clients' shards
    :param clients_perf_vec: batch overhead values of clients
    :param clients_crash_prob_vec: crash probs of clients
    :param crash_trace: simulated crash trace
    :param progress_trace: simulated progress trace
    :param response_time_limit: maximum round interval
    :param lag_t: tolerance of lag
    :return:
    """
    # init
    global_model = glob_model  # the global model
    models = [None for _ in range(env_cfg.n_clients)]  # local models
    client_ids = list(range(env_cfg.n_clients))
    distribute_models(global_model, models, client_ids)  # init local models
    # global cache, storing models to merge before aggregation and latest models after aggregation.
    cache = None  # cache will be initiated in the very first epoch

    # traces
    reporting_train_loss_vec = [0.0 for _ in range(env_cfg.n_clients)]
    reporting_test_loss_vec = [0.0 for _ in range(env_cfg.n_clients)]
    versions = np.array([-1 for _ in range(env_cfg.n_clients)])
    versions = np.array([-1 for _ in range(env_cfg.n_clients)])
    epoch_train_trace = []
    epoch_test_trace = []
    pick_trace = []
    make_trace = []
    undrafted_trace = []
    deprecated_trace = []
    round_trace = []
    acc_trace = []
    time_trace=[]

    # Global event handler
    event_handler = FLSup.EventHandler(['time', 'T_dist'])
    # Local counters
    # 1. Local timers - record work time of each client
    client_timers = [0.01 for _ in range(env_cfg.n_clients)]  # totally
    client_comm_timers = [0.0 for _ in range(env_cfg.n_clients)]  # comm. totally
    # 2. Futile counters - progression (i,e, work time) in vain caused by denial
    clients_est_round_T_train = np.array(client_shard_sizes) / env_cfg.batch_size * env_cfg.n_epochs / np.array(
        clients_perf_vec)
    cross_rounders = get_cross_rounders(clients_est_round_T_train, response_time_limit)
    picked_ids = []
    client_futile_timers = [0.0 for _ in range(env_cfg.n_clients)]  # totally
    eu_count = 0.0  # effective updates count
    sync_count = 0.0  # synchronization count
    version_var = 0.0
    # Best loss (global)
    best_rd = -1
    best_loss = float('inf')
    best_acc = -1.0
    best_model = None

    # begin training: global rounds
    for rd in range(env_cfg.n_rounds):
        print('\n> Round #%d' % rd)
        # reset timers
        client_round_timers = [0.0 for _ in range(env_cfg.n_clients)]  # local time in current round
        client_round_comm_timers = [0.0 for _ in range(env_cfg.n_clients)]  # local comm. time in current round
        picked_client_round_timers = [0.0 for _ in range(env_cfg.n_clients)]  # the picked clients to wait
        # randomly pick a specified fraction of clients to launch training
        quota = math.ceil(env_cfg.n_clients * env_cfg.pick_pct)  # the quota
        # simulate device or network failure
        crash_ids = crash_trace[rd]
        make_ids = [c_id for c_id in range(env_cfg.n_clients) if c_id not in crash_ids]
        # compensatory first-come-first-merge selection, last-round picks are considered low priority
        picked_ids = select_clients_CFCFM(make_ids, picked_ids, clients_perf_vec, cross_rounders, quota)
        # also record well-progressed but undrafted ones
        undrafted_ids = [c_id for c_id in make_ids if c_id not in picked_ids]
        # tracing
        make_trace.append(make_ids)
        pick_trace.append(picked_ids)
        undrafted_trace.append(undrafted_ids)
        print('> Clients crashed: ', crash_ids)
        print('> Clients undrafted: ', undrafted_ids)
        print('> Clients picked: ', picked_ids)  # first-come-first-merge

        # distributing step
        # distribute the global model to the edge in a discriminative manner
        print('>   @Cloud> distributing global model')
        good_ids, deprecated_ids = version_filter(versions, client_ids, rd - 1, lag_tolerant=lag_t)  # find deprecated
        latest_ids, straggler_ids = version_filter(versions, good_ids, rd - 1, lag_tolerant=0)  # find latest/straggled
        # case 1: deprecated clients
        distribute_models(global_model, models, deprecated_ids)  # deprecated clients are forced to sync. (sync.)
        update_cloud_cache_deprecated(cache, global_model, deprecated_ids)  # replace deprecated entries in cache
        deprecated_trace.append(deprecated_ids)
        print('>   @Cloud> Deprecated clients (forced to sync.):', get_versions(deprecated_ids, versions))
        update_versions(versions, deprecated_ids, rd-1)  # no longer deprecated
        # case 2: latest clients
        distribute_models(global_model, models, latest_ids)  # up-to-version clients will sync. (sync.)
        # case 3: non-deprecated stragglers
        # Moderately straggling clients remain unsync.
        # for 1. saving of downloading bandwidth, and 2. reservation of their potential progress (async.)
        sync_count += len(deprecated_ids) + len(latest_ids)  # count sync. overheads

        # Local training step
        for epo in range(env_cfg.n_epochs):  # local epochs (same # of epochs for each client)
            print('\n> @Devices> local epoch #%d' % epo)
            # invoke mini-batch training on selected clients, from the 2nd epoch
            if rd + epo == 0:  # 1st epoch all-in to get start points
                bak_make_ids = copy.deepcopy(make_ids)
                make_ids = list(range(env_cfg.n_clients))
            elif rd == 0 and epo == 1:  # resume
                cache = copy.deepcopy(models)  # as the very 1st epoch changes everything
                make_ids = bak_make_ids
            reporting_train_loss_vec = train(models, make_ids, env_cfg, cm_map, fed_loader_train,
                                             reporting_train_loss_vec, verbose=False)
            # add to trace
            epoch_train_trace.append(
                np.array(reporting_train_loss_vec) / (np.array(client_shard_sizes) * env_cfg.train_pct))
            # print('>   @Devices> %d clients train loss vector this epoch:' % env_cfg.n_clients,
            #       np.array(reporting_train_loss_vec) / (np.array(client_shard_sizes) * env_cfg.train_pct))
            # local test reports
            reporting_test_loss_vec = local_test(models, make_ids, env_cfg,  cm_map, fed_loader_test,
                                                 reporting_test_loss_vec)
            # add to trace
            epoch_test_trace.append(
                np.array(reporting_test_loss_vec) / (np.array(client_shard_sizes) * env_cfg.test_pct))
            # print('>   @Devices> %d clients test loss vector this epoch:' % env_cfg.n_clients,
            #       np.array(reporting_test_loss_vec) / (np.array(client_shard_sizes) * env_cfg.test_pct))

        # Aggregation step
        # discriminative update of cloud cache and aggregate
        # pre-aggregation: update cache from picked clients
        update_cloud_cache(cache, models, picked_ids)
        print('\n> Aggregation step (Round #%d)' % rd)
        global_model = safa_aggregate(cache, picked_ids,client_shard_sizes, data_size)  # aggregation
        # post-aggregation: update cache from undrafted clients
        update_cloud_cache(cache, models, undrafted_ids)

        # versioning
        eu = len(picked_ids)  # effective updates
        eu_count += eu  # EUR
        version_var += 0.0 if eu == 0 else np.var(versions[make_ids])  # Version Variance
        update_versions(versions, make_ids, rd)
        print('>   @Cloud> Versions updated:', versions)

        # Reporting phase: distributed test of the global model
        post_aggre_loss_vec, acc = global_test(global_model, env_cfg, cm_map, fed_loader_test)
        print('>   @Devices> post-aggregation loss reports  = ',
              np.array(post_aggre_loss_vec) / ((np.array(client_shard_sizes)) * env_cfg.test_pct))
        # overall loss, i.e., objective (1) in McMahan's paper
        overall_loss = np.array(post_aggre_loss_vec).sum() / (data_size*env_cfg.test_pct)
        # update so-far best
        if overall_loss < best_loss:
            best_loss = overall_loss
            best_acc = acc
            best_model = global_model
            best_rd = rd
        if env_cfg.keep_best:  # if to keep best
            global_model = best_model
            overall_loss = best_loss
            # acc = best_acc
        print('>   @Cloud> post-aggregation loss avg = ', overall_loss)
        round_trace.append(overall_loss)
        acc_trace.append(acc)

        # update timers
        for c_id in range(env_cfg.n_clients):
            if c_id in make_ids:
                # client_local_round time T(k) = T_download(k) + T_train(k) + T_upload(k), where
                #   T_comm(k) = T_download(k) + T_upload(k) = 2* model_size / bw_k
                #   T_train = number of batches / client performance
                env_cfg.bw_c_set[c_id] = env_cfg.bw_c_set[c_id] + round(random.uniform(-0.002, 0.002), 3)
                while env_cfg.bw_c_set[c_id] <= 0:
                    env_cfg.bw_c_set[c_id] = env_cfg.bw_c_set[c_id] + round(random.uniform(0, 0.004), 3)
                T_comm = 2*env_cfg.model_size / env_cfg.bw_c_set[c_id]
                T_train = client_shard_sizes[c_id] / env_cfg.batch_size * env_cfg.n_epochs / clients_perf_vec[
                    c_id]
                # T_comm = 2*env_cfg.model_size / env_cfg.bw_set[0]
                # T_train = client_shard_sizes[c_id] / env_cfg.batch_size * env_cfg.n_epochs / clients_perf_vec[c_id]
                print('train time and comm. time locally:', T_train, T_comm)
                # timers
                client_round_timers[c_id] = min(response_time_limit, T_comm + T_train)  # including comm. and training
                client_round_comm_timers[c_id] = T_comm  # comm. is part of the run time
                client_timers[c_id] += client_round_timers[c_id]  # sum up
                client_comm_timers[c_id] += client_round_comm_timers[c_id]  # sum up
                if c_id in picked_ids:
                    picked_client_round_timers[c_id] = client_round_timers[c_id]  # we need to await the picked
                if c_id in deprecated_ids:  # deprecated clients, forced to sync. at distributing step
                    client_futile_timers[c_id] += progress_trace[rd][c_id] * client_round_timers[c_id]
                    client_round_timers[c_id] = response_time_limit  # no response
        dist_time = env_cfg.model_size * sync_count / env_cfg.bw_s_set  # T_disk = model_size * N_sync / BW
        # round_response_time = min(response_time_limit, max(picked_client_round_timers))  # w8 to meet quota
        # global_timer += dist_time + round_response_time
        # global_T_dist_timer += dist_time
        # Event updates
        event_handler.add_parallel('time', picked_client_round_timers, reduce='max')
        event_handler.add_sequential('time', dist_time)
        event_handler.add_sequential('T_dist', dist_time)
        time = event_handler.get_state('time')
        time_trace.append(time)

        print('> Round client run time:', client_round_timers)  # round estimated finish time

    # Stats
    global_timer = event_handler.get_state('time')
    global_T_dist_timer = event_handler.get_state('T_dist')
    # Traces
    print('> Train trace:')
    utils.show_epoch_trace(epoch_train_trace, env_cfg.n_clients, plotting=False, cols=1)  # training trace
    print('> Test trace:')
    utils.show_epoch_trace(epoch_test_trace, env_cfg.n_clients, plotting=False, cols=1)
    print('> Round trace:')
    utils.show_round_trace(round_trace, plotting=env_cfg.showplot, title_='SAFA')

    # display timers
    print('\n> Experiment stats')
    print('> Clients round time:', client_timers)
    print('> Clients futile run time:', client_futile_timers)
    futile_pcts = (np.array(client_futile_timers) / np.array(client_timers)).tolist()
    print('> Clients futile percent (avg.=%.3f):' % np.mean(futile_pcts), futile_pcts)
    eu_ratio = eu_count / env_cfg.n_rounds / env_cfg.n_clients
    print('> EUR:', eu_ratio)
    sync_ratio = sync_count / env_cfg.n_rounds / env_cfg.n_clients
    print('> SR:', sync_ratio)
    version_var = version_var/env_cfg.n_rounds
    print('> VV:', version_var)
    print('> Total time consumption:', global_timer)
    print('> Total distribution time (T_dist):', global_T_dist_timer)
    print('> Loss = %.6f/at Round %d:' % (best_loss, best_rd))

    # Logging
    detail_env = (client_shard_sizes, clients_perf_vec, clients_crash_prob_vec)
    utils.log_stats('../stats/exp_log.txt', env_cfg, detail_env, time_trace,epoch_train_trace, epoch_test_trace,
                    round_trace, acc_trace, make_trace, pick_trace, crash_trace, deprecated_trace,
                    client_timers, client_futile_timers, global_timer, global_T_dist_timer, eu_ratio, sync_ratio,
                    version_var, best_rd, best_loss, extra_args={'lag_tolerance': lag_t}, log_loss_traces=False)

    return best_model, best_rd, best_loss


# test area
# client_ids = [0,1,2, 3,4]
# versions =   [3,3,0,-1,2]
# rd = 4
# good_ids, deprecated_ids = version_filter(versions, client_ids, rd - 1, lag_tolerant=3)  # find deprecated
# latest_ids, straggler_ids = version_filter(versions, good_ids, rd - 1, lag_tolerant=0)  # find latest/straggled
# print(good_ids)
# print(deprecated_ids)
# print(latest_ids)
# print(straggler_ids)
# exit(0)
# ids =  [0,1,2,3,4]
# perf = [0,10,20,30,40]
# last_picks = [2,4]
# makes = [1,2,3,4]
# print(select_clients_CFCFM(makes,last_picks,perf,[0,1],quota=5))






