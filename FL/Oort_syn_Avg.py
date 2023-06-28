
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import sys
import os
import math
import random
import numpy as np
# import syft as sy
from learning_tasks import svmLoss
import utils
import FLLocalSupport as FLSup


def train(models, picked_ids, env_cfg, cm_map, fdl, last_loss_rep,
          last_clients_loss_sum,mode,verbose=True):
    """
    Execute one EPOCH of training process of any machine learning model on all clients
    :param models: a list of model prototypes corresponding to clients
    :param picked_ids: participating client indices for local training
    :param env_cfg: environment configurations
    :param cm_map: the client-model map, as a dict
    :param fdl: an instance of FederatedDataLoader
    :param last_loss_rep: loss reports in last run
    :param last_clients_loss_sum: loss**2 reports in last run
    :param mode : need sum of loss or sum of loss**2
    :param verbose: display batch progress or not.
    :return: epoch training loss of each client, batch-summed
    """
    dev = env_cfg.device
    if len(picked_ids) == 0:  # no training happens
        return last_loss_rep,last_clients_loss_sum
    # extract settings
    n_models = env_cfg.n_clients  # # of clients
    # initialize loss report, keep loss tracks for idlers, clear those for participants
    client_train_loss_vec = last_loss_rep
    client_train_loss_vec_squar=last_clients_loss_sum
    for id in picked_ids:
        client_train_loss_vec[id] = 0.0
        client_train_loss_vec_squar[id]=0.0
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
        inputs, labels = inputs, labels  # data to device
        model_id = cm_map[client.id]  # locate the right model index
        # neglect non-participants
        if model_id not in picked_ids:
            continue
        # mini-batch GD
        if mode==0:
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

        if mode==0:
            print('>   batch loss = ', loss.item())  # avg. batch loss
        client_train_loss_vec[model_id] += loss.detach().item()*len(inputs)  # sum up
        client_train_loss_vec_squar[model_id] += (loss.detach().item()*len(inputs))**2

    # Restore printing
    if not verbose:
        sys.stdout = sys.__stdout__
    # end an epoch-training - all clients have traversed their own local data once
    return client_train_loss_vec,client_train_loss_vec_squar


def local_test(models, env_cfg, picked_ids, n_models, cm_map, fdl, last_loss_rep):
    """
    Evaluate client models locally and return a list of loss/error
    :param models: a list of model prototypes corresponding to clients
    :param env_cfg: environment configurations
    :param picked_ids: selected client indices for local training
    :param n_models: # of models, i.e., clients
    :param cm_map: the client-model map, as a dict
    :param fdl: an instance of FederatedDataLoader
    :param last_loss_rep: loss reports in last run
    :return: epoch test loss of each client, batch-summed
    """
    if not picked_ids:  # no training happened
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

    # initialize evaluation mode
    for m in range(n_models):
        models[m].eval()
    # local evaluation, batch-wise
    acc = 0.0
    count = 0.0
    with torch.no_grad():
        for batch_id, (inputs, labels, client) in enumerate(fdl):
            # if len(labels)==0:
            #     continue
            # inputs, labels = inputs.to(dev), labels.to(dev)  # data to device
            model_id = cm_map[client.id]  # locate the right model index
            # neglect non-participants
            if model_id not in picked_ids:
                continue
            model = models[model_id]
            # inference
            y_hat = model(inputs)
            # loss
            loss = loss_func(y_hat, labels)
            client_test_loss_vec[model_id] += loss.item()
            # accuracy
            b_acc, b_cnt = utils.batch_sum_accuracy(y_hat, labels, env_cfg.loss)
            acc += b_acc
            count += b_cnt

    # print('> acc = %.6f' % (acc/count))
    return client_test_loss_vec


def global_test(model, env_cfg, cm_map, fdl):
    """
    Testing the aggregated global model by averaging its error on each local data
    :param model: the global model
    :param env_cfg: env configuration
    :param cm_map: the client-model map, as a dict
    :param fdl: FederatedDataLoader
    :return: global model's loss on each client (as a vector), accuracy
    """
    dev = env_cfg.device
    test_sum_loss_vec = [0 for _ in range(env_cfg.n_clients)]
    # Define loss based on task
    if env_cfg.loss == 'mse':  # regression task
        loss_func = nn.MSELoss(reduction='sum')
    elif env_cfg.loss == 'svmLoss':  # SVM task
        loss_func = svmLoss(reduction='sum')
    elif env_cfg.loss == 'nllLoss':  # CNN mnist task
        loss_func = nn.NLLLoss(reduction='sum')
        is_cnn = True

    print('> global test')
    # initialize evaluation mode
    model.eval()
    acc = 0.0
    count = 0
    # local evaluation, batch-wise
    for batch_id, (inputs, labels, client) in enumerate(fdl):
        # inputs, labels = inputs.to(dev), labels.to(dev)  # data to device
        model_id = cm_map[client.id]
        # inference
        y_hat = model(inputs)
        # loss
        loss = loss_func(y_hat, labels)
        test_sum_loss_vec[model_id] += loss.item()
        # compute accuracy
        b_acc, b_cnt = utils.batch_sum_accuracy(y_hat, labels, env_cfg.loss)
        acc += b_acc
        count += b_cnt

    print('>   acc = %.6f' % (acc / count))
    return test_sum_loss_vec, acc/count


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


def aggregate(global_model,models, picked_ids,local_shards_sizes, data_size):
    """
    The function implements aggregation step (FedAvg), i.e., w = sum_k(n_k/n * w_k)
    :param models: a list of local models
    :param picked_ids: selected client indices for local training
    :param local_shards_sizes: a list of local data sizes, aligned with the orders of local models, say clients.
    :param data_size: total data size
    :return: a global model
    """
    print('>   Aggregating (Oort)...')
    # shape the global model
    global_model = copy.deepcopy(global_model)
    global_model_params = global_model.state_dict()
    for pname, param in global_model_params.items():
        global_model_params[pname] = 0.0
    round_data_size = 0
    for id in picked_ids:
        round_data_size += local_shards_sizes[id]
    client_weights_vec = np.array(local_shards_sizes) / round_data_size  # client weights (i.e., n_k / n)
    # client_weights_vec = np.array(local_shards_sizes) / data_size  # client weights (i.e., n_k / n)
    for m in picked_ids:  # for each local model
        for pname, param in models[m].state_dict().items():
            global_model_params[pname] += param.data * client_weights_vec[m]  # sum up the corresponding param

    # load state dict back
    global_model.load_state_dict(global_model_params)
    return global_model


def get_cross_rounders(clients_est_round_T_train, max_round_interval):
    cross_rounder_ids = []
    for c_id in range(len(clients_est_round_T_train)):
        if clients_est_round_T_train[c_id] > max_round_interval:
            cross_rounder_ids.append(c_id)
    return cross_rounder_ids


def sort_ids_by_esttime_asc(id_list, est_time_list):
    """
    Sort a list of client ids according to their est. finish time, in an ascending ordered
    :param id_list: a list of client ids to sort
    :param est_time_list: full list of all clients' est. finish time
    :return: sorted id_list
    """
    # make use of a map
    cp_map = {}  # client-perf-map
    for id in id_list:
        cp_map[id] = est_time_list[id]  # build the map with corresponding perf
    # sort by perf
    sorted_map = sorted(cp_map.items(), key=lambda x: x[1], reverse=False)  # a sorted list of tuples
    sorted_id_list = [sorted_map[i][0] for i in range(len(id_list))]  # extract the ids into a list
    return sorted_id_list


def Select_oort(ava_ids,wait_num,n_clients,clients_loss_sum,client_shard_sizes,T_limit,clients_arrival_T):
    """
    :param clients_loss_sum is the sum of loss^2 for each client
    """

    #cliens' utility equals to |B_i|*sqrt((\sum loss^2) / |B_i|), clients_loss_sum
    client_utility=[0.0 for _ in range(n_clients)]
    data_utility = np.sqrt(np.array(clients_loss_sum)/np.array(client_shard_sizes))*np.array(client_shard_sizes)
    for i in range(n_clients):
        if clients_arrival_T[i]>T_limit:
            alpha=2
        else:
            alpha=0
        client_utility[i]=data_utility[i]*(T_limit/clients_arrival_T[i])**alpha
    sorted_ids = sort_ids_by_esttime_asc(ava_ids, client_utility)
    sorted_ids=sorted_ids[::-1]
    make_ids = sorted_ids[:wait_num]
    return make_ids


def run_Oort(env_cfg, glob_model, cm_map, data_size, fed_loader_train, fed_loader_test, client_shard_sizes,
              clients_perf_vec, clients_crash_prob_vec, crash_trace, progress_trace, response_time_limit):
    """
    Primal implementation of FedAvg for FL
    :param env_cfg:
    :param glob_model:
    :param cm_map:
    :param data_size:
    :param fed_loader_train:
    :param fed_loader_test:
    :param client_shard_sizes:
    :param clients_perf_vec:
    :param clients_crash_prob_vec:
    :param crash_trace:
    :param progress_trace:
    :param response_time_limit:
    :return:
    """
    # init
    global_model = glob_model  # the global model
    models = [None for _ in range(env_cfg.n_clients)]  # local models
    client_ids = list(range(env_cfg.n_clients))
    distribute_models(global_model, models, client_ids)  # init local models

    # traces
    reporting_train_loss_vec = [0 for _ in range(env_cfg.n_clients)]
    reporting_test_loss_vec = [0 for _ in range(env_cfg.n_clients)]
    epoch_train_trace = []
    epoch_test_trace = []
    make_trace = []
    pick_trace = []
    round_trace = []
    acc_trace = []
    time_trace=[]

    # Global event handler
    event_handler = FLSup.EventHandler(['time', 'T_dist'])
    # Local counters
    # 1. Local timers - record work time of each client
    client_timers = [0.01 for _ in range(env_cfg.n_clients)]  # totally
    client_comm_timers = [0.0 for _ in range(env_cfg.n_clients)]  # comm. totally
    # 2. Futile counters - progression (i,e, work time) in vain caused by local crashes
    clients_est_round_T_train = np.array(client_shard_sizes) / env_cfg.batch_size * env_cfg.n_epochs / np.array(
        clients_perf_vec)
    cross_rounders = get_cross_rounders(clients_est_round_T_train, response_time_limit)
    client_futile_timers = [0.01 for _ in range(env_cfg.n_clients)]  # totally
    clients_loss_sum = [0.0 for _ in range(env_cfg.n_clients)]
    eu_count = 0.0  # effective updates count
    sync_count = 0.0  # synchronization count
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
        # randomly pick a specified fraction of clients to launch training
        n_picks = math.ceil(env_cfg.n_clients * env_cfg.pick_pct)
        selected_ids = random.sample(range(env_cfg.n_clients), n_picks)  # initial selection
        selected_ids.sort()
        # simulate device or network failure
        crash_ids = crash_trace[rd]
        client_round_progress = progress_trace[rd]
        # get clients' utility
        bake_models = copy.deepcopy(models)
        if rd == 0:
            _, clients_loss_sum = train(bake_models, selected_ids, env_cfg, cm_map,
                                        fed_loader_train,
                                        reporting_train_loss_vec, clients_loss_sum, mode=1,
                                        verbose=False)
        # Oort client re-selection
        selected_ids = Select_oort(selected_ids, env_cfg.wait_num, env_cfg.n_clients, clients_loss_sum,
                                   client_shard_sizes,env_cfg.T_limit,clients_est_round_T_train)
        pick_trace.append(selected_ids)  # tracing
        print('> Clients selected(Oort): ', selected_ids)
        # some selected may crash
        submit_ids = [c_id for c_id in selected_ids if c_id not in crash_ids and c_id not in cross_rounders]
        # tracing
        make_trace.append(submit_ids)
        eu_count += len(submit_ids)  # count effective updates
        print('> Clients crashed: ', crash_ids)

        # distributing step: broadcast (for non-selected clients, just update their cache entry on the cloud
        # so as to have all models share initialization upon aggregation for loss reduction)
        print('>   @Cloud> distributing global model')
        distribute_models(global_model, models, client_ids)
        sync_count += len(selected_ids)  # count sync. overheads

        # Local training step
        for epo in range(env_cfg.n_epochs):  # local epochs (same # of epochs for each client)
            print('\n> @Devices> local epoch #%d' % epo)
            # invoke mini-batch training on selected clients, from the 2nd epoch
            if rd + epo == 0:  # 1st epoch all-in to get start points
                bak_make_ids = copy.deepcopy(submit_ids)
                submit_ids = list(range(env_cfg.n_clients))
            elif rd == 0 and epo == 1:  # resume
                submit_ids = bak_make_ids
            reporting_train_loss_vec,_ = train( models, submit_ids, env_cfg, cm_map, fed_loader_train,
                                             reporting_train_loss_vec,clients_loss_sum,mode=0, verbose=False)
            # add to trace
            epoch_train_trace.append(
                np.array(reporting_train_loss_vec) / (np.array(client_shard_sizes) * env_cfg.train_pct))
            # print('>   @Devices> %d clients train loss vector this epoch:' % env_cfg.n_clients,
            #       np.array(reporting_train_loss_vec) / (np.array(client_shard_sizes) * env_cfg.train_pct))
            # local test reports
            reporting_test_loss_vec = local_test(models, env_cfg, submit_ids, env_cfg.n_clients, cm_map,
                                                 fed_loader_test, reporting_test_loss_vec)
            # add to trace
            epoch_test_trace.append(
                np.array(reporting_test_loss_vec) / (np.array(client_shard_sizes) * env_cfg.test_pct))
            # print('>   %d @Devices> clients test loss vector this epoch:' % env_cfg.n_clients,
            #       np.array(reporting_test_loss_vec) / (np.array(client_shard_sizes) * env_cfg.test_pct))

        # Aggregation step
        print('\n> Aggregation step (Round #%d)' % rd)
        global_model = aggregate(global_model,models, submit_ids,client_shard_sizes, data_size)

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
            if c_id in selected_ids:  # only compute timers for the selected
                # client_local_round time T(k) = T_download(k) + T_train(k) + T_upload(k), where
                #   T_comm(k) = T_download(k) + T_upload(k) = 2* model_size / bw_k
                #   T_train = number of batches / client performance
                env_cfg.bw_c_set[c_id] = env_cfg.bw_c_set[c_id] + round(random.uniform(-0.002, 0.002), 3)
                while env_cfg.bw_c_set[c_id] <= 0:
                    env_cfg.bw_c_set[c_id] = env_cfg.bw_c_set[c_id] + round(random.uniform(0, 0.004), 3)
                T_comm = env_cfg.model_size / env_cfg.bw_c_set[c_id]
                T_train = client_shard_sizes[c_id] / env_cfg.batch_size * env_cfg.n_epochs / clients_perf_vec[
                    c_id]
                # T_comm = env_cfg.model_size / env_cfg.bw_set[0]
                # T_train = client_shard_sizes[c_id] / env_cfg.batch_size * env_cfg.n_epochs / clients_perf_vec[c_id]
                print('train time and comm. time locally:', T_train, T_comm)
                client_round_timers[c_id] = T_comm + T_train  # including comm. and training
                client_round_comm_timers[c_id] = T_comm  # comm. is part of the run time
                client_timers[c_id] += client_round_timers[c_id]  # sum up
                client_comm_timers[c_id] += client_round_comm_timers[c_id]  # sum up
                if c_id in crash_ids:  # failed clients
                    client_futile_timers[c_id] += client_round_timers[c_id] * client_round_progress[c_id]
                    # client_round_timers[c_id] = response_time_limit  # no response
        dist_time = env_cfg.model_size * sync_count / env_cfg.bw_s_set  # T_disk = model_size * N_sync / BW
        # round_response_time = min(response_time_limit, max(client_round_timers))
        # global_timer += dist_time + round_response_time
        # global_T_dist_timer += dist_time

        # Event updates
        event_handler.add_parallel('time', client_round_timers, reduce='max')
        event_handler.add_sequential('time', dist_time)
        event_handler.add_sequential('T_dist', dist_time)
        time = event_handler.get_state('time')
        time_trace.append(time)

        print('> Round client run time:', client_round_timers)  # round estimated finish time
        print('> Round client progress:', client_round_progress)  # round actual progress at last
        env_cfg.lr *= env_cfg.lr_decay  # learning rate decay

    # Stats
    global_timer = event_handler.get_state('time')
    global_T_dist_timer = event_handler.get_state('T_dist')
    # Traces
    print('> Train trace:')
    utils.show_epoch_trace(epoch_train_trace, env_cfg.n_clients, plotting=False, cols=1)  # training trace
    print('> Test trace:')
    utils.show_epoch_trace(epoch_test_trace, env_cfg.n_clients, plotting=False, cols=1)
    print('> Round trace:')
    utils.show_round_trace(round_trace, plotting=env_cfg.showplot, title_='Primal FedAvg')

    # display timers
    print('\n> Experiment stats')
    print('> Clients run time:', client_timers)
    print('> Clients futile run time:', client_futile_timers)
    futile_pcts = (np.array(client_futile_timers) / np.array(client_timers)).tolist()
    print('> Clients futile percent (avg.=%.3f):' % np.mean(futile_pcts), futile_pcts)
    eu_ratio = eu_count/env_cfg.n_rounds/env_cfg.n_clients
    print('> EUR:', eu_ratio)
    sync_ratio = sync_count/env_cfg.n_rounds/env_cfg.n_clients
    sync_ratio = sync_ratio/(env_cfg.pick_pct+0.1)*env_cfg.pick_pct if sync_ratio > env_cfg.pick_pct else sync_ratio
    print('> SR:', sync_ratio)
    print('> Total time consumption:', global_timer)
    print('> Total distribution time (T_dist):', global_T_dist_timer)
    print('> Loss = %.6f/at Round %d:' % (best_loss, best_rd))

    # Logging
    detail_env = (client_shard_sizes, clients_perf_vec, clients_crash_prob_vec)
    utils.log_stats('../stats/exp_log.txt', env_cfg, detail_env, time_trace,epoch_train_trace, epoch_test_trace,
                    round_trace, acc_trace, make_trace, pick_trace, crash_trace, None,
                    client_timers, client_futile_timers, global_timer, global_T_dist_timer, eu_ratio, sync_ratio, 0.0,
                    best_rd, best_loss, extra_args=None, log_loss_traces=False)

    return best_model, best_rd, best_loss
