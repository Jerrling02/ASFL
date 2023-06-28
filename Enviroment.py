
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import sys
import os
import random
from FL.learning_tasks import svmLoss
import FL.utils as utils
import FL.FLLocalSupport as FLSup
from FL.learning_tasks import MLmodelReg, MLmodelCNN
from FL.learning_tasks import MLmodelSVM



def sort_ids_by_atime_asc(id_list, atime_list):
    """
    Sort a list of client ids according to their performance, in an descending ordered
    :param id_list: a list of client ids to sort
    :param perf_list: full list of all clients' arrival time
    :return: sorted id_list
    """

    # make use of a map
    cp_map = {}  # client-perf-map
    for id in id_list:
        cp_map[id] = atime_list[id]  # build the map with corresponding perf
    # sort by perf

    sorted_map = sorted(cp_map.items(), key=lambda x: x[1])  # a sorted list of tuples
    sorted_id_list = [sorted_map[i][0] for i in range(len(id_list))]  # extract the ids into a list
    return sorted_id_list
def Select_adaptive(ava_ids,select_pt,n_clients,clients_loss_sum,client_shard_sizes):
    """
    :param clients_loss_sum is the sum of loss^2 for each client
    """

    #cliens' utility equals to |B_i|*sqrt((\sum loss^2) / |B_i|), clients_loss_sum
    client_utility = np.sqrt(np.array(clients_loss_sum)/np.array(client_shard_sizes))*np.array(client_shard_sizes)
    select_num = int(np.ceil(n_clients * select_pt))
    sorted_ids = sort_ids_by_atime_asc(ava_ids, client_utility)
    sorted_ids=sorted_ids[::-1]
    make_ids = sorted_ids[:select_num]
    return make_ids



def select_clients_CFCFM(make_ids, undrafted_ids,clients_arrival_T, quota):
    """
    Select clients to aggregate their models according to Compensatory First-Come-First-Merge principle.
    :param make_ids: ids of clients start their training this round
    :param undrafted_ids: ids of clients unpicked previous rounds
    :param clients_arrival_T: clients' arrival time at rd round
    :param quota: number of clients to draw this round
    :return:
    picks: ids of selected clients
    clients_arrival_T: clients' arrival time for next  round
    undrafted_ids: ids of undrafted clients
    """
    clients_ids = make_ids + undrafted_ids
    # set_lst1 = set(make_ids)
    # set_lst2 = set(undrafted_ids)
    # set_lst3 = set(clients_ids)
    #
    # # set会生成一个元素无序且不重复的可迭代对象，也就是我们常说的去重



    sorted_ids = sort_ids_by_atime_asc(clients_ids,clients_arrival_T)
    picks = sorted_ids[:quota]
    # num = 0
    # while quota+num<len(sorted_ids):
    #     if clients_arrival_T[sorted_ids[quota+num]] == 0:
    #         picks.append(sorted_ids[quota+num])
    #         num += 1
    #     else:
    #         break
    max_id = sorted_ids[quota-1]
    undrafted_ids = []
    # record well-progressed but undrafted ones
    for id in clients_ids:
        if id not in picks:
            undrafted_ids.append(id)
    #pick the client already upload





    return picks,undrafted_ids,max_id


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
            inputs, labels = inputs, labels  # data to device
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
            # if len(labels)==0:
            #     print(model_id,y_hat,labels)
            acc += b_acc
            count += b_cnt

    # print('> acc = %.6f' % (acc / count))
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
    # print('> global test')
    model.eval()
    # local evaluation, batch-wise
    acc = 0.0
    count = 0
    for batch_id, (inputs, labels, client) in enumerate(fdl):
        inputs, labels = inputs, labels  # data to device
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

    # print('>   acc = %.6f' % (acc/count))
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
def extract_weights(model):
    weights = []
    for name, weight in model.named_parameters():
        weights.append((name, weight.data))

    return weights

def extract_client_updates(global_model,models,picked_ids):

    baseline_weights = extract_weights(global_model)
    recieve_buffer=[]
    for m in picked_ids:
        recieve_buffer.append((m,extract_weights(models[m])))
    # Calculate updates from weights
    updates = []
    for m,weight in recieve_buffer:
        update = []
        for i, (name, weight) in enumerate(weight):
            bl_name, baseline = baseline_weights[i]

            # Ensure correct weight is being updated
            assert name == bl_name

            # Calculate update
            delta = weight - baseline
            update.append((name, delta))
        updates.append(update)

    return updates

def safa_aggregate(global_model,models, picked_ids,local_shards_sizes, varsigma,versions,rd):
    """
    The function implements aggregation step (Semi-Async. FedAvg), allowing cross-round com
    :param models: a list of local models
    :param picked_ids: selected client indices for local training
    :param local_shards_sizes: a list of local data sizes, aligned with the orders of local models, say clients.
    :param varsigma: decay coefficient
    :return: a global model
    """
    # print('>   Aggregating (SAAR)...')
    # updates = extract_client_updates(global_model,models,picked_ids)
    # baseline_weights = extract_weights(global_model)
    # #calculate client vec
    # client_weights_vec=[]
    # for m in picked_ids:
    #     client_weights_vec.append(local_shards_sizes[m]/data_size*10)
    # updated_state_dict = {}
    # for i, update in enumerate(updates):
    #     for j,(name, delta) in enumerate(update):
    #         bl_name, baseline = baseline_weights[j]
    #
    #         # Ensure correct weight is being updated
    #         assert name==bl_name
    #
    #         updated_state_dict[name] = baseline+ delta * client_weights_vec[i]
    #
    #
    # # load state dict back
    # global_model.load_state_dict(updated_state_dict)
    # return global_model
    # shape the global model
    global_model = copy.deepcopy(global_model)
    global_model_params = global_model.state_dict()
    global_model_params_bake=copy.deepcopy(global_model_params)
    for pname, param in global_model_params.items():
        global_model_params[pname] = 0.0
    round_data_size = 0
    for id in picked_ids:
        round_data_size += local_shards_sizes[id]
    client_weights_vec = np.array(local_shards_sizes) / round_data_size  # client weights (i.e., n_k / n)
    for m in picked_ids:  # for each local model
        factor=rd-versions[m]
        for pname, param in models[m].state_dict().items():
            global_model_params[pname] += client_weights_vec[m]*((varsigma**factor)*param.data+(1-(varsigma**factor))*global_model_params_bake[pname] )# sum up the corresponding param
    # load state dict back
    global_model.load_state_dict(global_model_params)
    return global_model


class Federate_learning():
    def __init__(self, args, cm_map, data_size, fed_loader_train, fed_loader_test, client_shard_sizes,
                clients_perf_vec, clients_crash_prob_vec, crash_trace, progress_trace):

        #init clients and dataloader
        self.cm_map = cm_map
        self.data_size= data_size
        self.fed_loader_train = fed_loader_train
        self.fed_loader_test = fed_loader_test
        self.client_shard_sizes = client_shard_sizes
        self.clients_perf_vec = clients_perf_vec
        self.clients_crash_prob_vec = clients_crash_prob_vec
        self.crash_trace = crash_trace
        self.progress_trace= progress_trace

        #init task
        self.device = args.device
        self.args = args
        self.n_clients = args.n_clients
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs
        self.bw_s_set = args.bw_s_set
        self.bw_c_set =args.bw_c_set
        self.model_size = args.model_size
        self.pick_C = args.pick_C
        self.train_pct = args.train_pct
        self.test_pct = args.test_pct
        self.keep_best = args.keep_best
        self.alpha = args.alpha
        self.beta = args.beta
        self.varsigma=args.varsigma
        self.n_rounds = args.n_rounds
        self.T_max = args.T_max
        self.wait_num_fixed=args.wait_num_fixed
        self.lag_t_fixed=args.lag_t_fixed
        self.load=args.load
        if self.wait_num_fixed:
            self.wait_num=args.wait_num
        if self.lag_t_fixed:
            self.lag_tol=args.lag_tol
        self.lag_t_max = args.lag_t_max
        self.wait_num_max=int(self.n_clients*self.pick_C)

        #init global model
        self.global_model = self.init_glob_model()  # the global model
        self.models = [None for _ in range(self.n_clients)]  # local models
        self.client_ids = list(range(self.n_clients))
        # reset cache
        self.cache = copy.deepcopy(self.models)
        self.bw_c_set = [round(random.uniform(0.125, 0.3), 3) for _ in range(self.n_clients)]



        self.reward = 0
        # 设置state为当前轮次的剩余计算通信资源，每个客户端的损失,当前轮次
        self.state = [-1] * 5
        self.state_space = len(self.state)
        self.observation_space = self.state_space
        self.action_bound = [0, 1]

        # self.action_space = self.num_edges * self.num_max+ self.num_edges * self.tau_max
        self.action_space = int(self.n_clients*self.pick_C) + self.lag_t_max


    def init_glob_model(self):
        """
        Initialize the global model as per the settings of FL and machine learning task

        :param env_cfg:
        :return: model
        """
        model = None
        dev = self.args.device
        # have to transiently set default tensor type to cuda.float, otherwise model.to(dev) fails on GPU
        if dev.type == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # instantiate models, one per client
        if self.args.task_type == 'Reg':
            model = MLmodelReg(in_features=self.args.in_dim, out_features=self.args.out_dim).to(dev)
        elif self.args.task_type == 'SVM':
            model = MLmodelSVM(in_features=self.args.in_dim).to(dev)
        elif self.args.task_type == 'CNN':
            model = MLmodelCNN(classes=10).to(dev)

        torch.set_default_tensor_type('torch.FloatTensor')
        return model

    def reset(self):
        # agents初始化
        # self.reward = 0
        torch.manual_seed(random.randint(0, 20))
        np.random.seed(random.randint(0, 20))

        self.reporting_train_loss_vec = [0.0 for _ in range(self.n_clients)]
        self.clients_loss_sum = [0.0 for _ in range(self.n_clients)]
        self.reporting_test_loss_vec = [0.0 for _ in range(self.n_clients)]
        self.versions = np.array([-1 for _ in range(self.n_clients)])
        self.epoch_train_trace = []
        self.epoch_test_trace = []
        self.pick_trace = []
        self.make_trace = []
        self.undrafted_trace = []
        self.deprecated_trace = []
        self.round_trace = []
        self.acc_trace = []
        self.client_timers = [0.01 for _ in range(self.n_clients)]  # totally
        self.clients_est_round_T_train = np.array(self.client_shard_sizes) / self.batch_size * self.n_epochs / np.array(
            self.clients_perf_vec)
        self.clients_est_round_T_comm = self.model_size / np.array(self.bw_c_set)

        # 第一轮时间包括初始分发模型时间，训练时间和上传时间
        self.clients_arrival_T = np.array(self.clients_est_round_T_train) + np.array(self.clients_est_round_T_comm)
        self.picked_ids = []
        self.undrafted_ids = []
        self.client_futile_timers = [0.0 for _ in range(self.n_clients)]  # totally
        self.eu_count = 0.0  # effective updates count
        self.sync_count = 0.0  # synchronization count
        self.version_var = 0.0
        self.max_id = None # the id of clients who has max arrival time
        # Best loss (global)
        self.best_rd = -1
        self.best_loss = float('inf')
        self.best_acc = -1.0
        self.best_model = None
        #reset models
        self.global_model = self.init_glob_model()  # the global model
        self.models = [None for _ in range(self.n_clients)]  # local models
        self.client_ids = list(range(self.n_clients))
        # reset cache
        self.cache = copy.deepcopy(self.models)
        # reset for first round of distribute
        distribute_models(self.global_model, self.models, self.client_ids)  # init local models

        # reset Global event handler
        self.event_handler = FLSup.EventHandler(['time', 'T_dist', 'obj_acc_time'])


        # 更新state
        for i in range(len(self.state)):
            self.state[i] = 0.0

        return self.state

    def step(self, actions,rd):

        done = False

        wait_num, lag_t = action_choice(actions)

        if self.load:
            if self.wait_num_fixed:
                wait_num=self.wait_num
            if self.lag_t_fixed:
                lag_t=self.lag_tol


        states,obj_acc_time,picked_client_round_timers,acc,total_time=self.train(lag_t, wait_num, rd)
        # 最前面的state是客户端损失
        for i in range(len(states)):
            self.state[i] = states[i]

        #update reward
        if total_time <= self.T_max:
            omega = self.alpha
        else:
            omega = self.beta
        # self.reward = (acc*(self.T_max/total_time)**omega)*10
        self.reward = acc*obj_acc_time*10


        return copy.copy(self.state), self.reward,picked_client_round_timers,acc,total_time

    # train() runs for one global round
    def train(self,lag_t,wait_num,rd):
        # print('\n> Round #%d' % rd)


        #reset number of syn clients
        m_syn=0
        # reset timers
        client_round_timers = [0.0 for _ in range(self.n_clients)]  # local time in current round
        client_round_comm_timers = [0.0 for _ in range(self.n_clients)]  # local comm. time in current round
        picked_client_round_timers = [0.0 for _ in range(self.n_clients)]  # the picked clients to wait
        # randomly pick a specified fraction of clients to launch training
        # quota = math.ceil(self.n_clients * self.pick_pct)  # the quota
        quota = wait_num

        # ---------------------distributing step--------------------
        # distribute the global model to the edge in a discriminative manner
        # print('>   @Cloud> distributing global model')
        good_ids, deprecated_ids = version_filter(self.versions, self.undrafted_ids, rd - 1,
                                                  lag_tolerant=lag_t)  # find deprecated
        # latest_ids, straggler_ids = version_filter(self.versions, good_ids, rd - 1, lag_tolerant=0)  # find latest/straggled
        # 强制同步后去掉undrafted clients：
        for id in self.undrafted_ids:
            if id in deprecated_ids:
                self.undrafted_ids.remove(id)
        # case 1: deprecated clients
        distribute_models(self.global_model, self.models,
                          deprecated_ids)  # deprecated clients are forced to sync. (sync.)
        update_cloud_cache_deprecated(self.cache, self.global_model,
                                      deprecated_ids)  # replace deprecated entries in cache
        self.deprecated_trace.append(deprecated_ids)
        # print('>   @Cloud> Deprecated clients (forced to sync.):', get_versions(deprecated_ids, self.versions))
        update_versions(self.versions, deprecated_ids, rd - 1)  # no longer deprecated
        #evaluate synchronous amount
        if rd == 0:
            m_syn = self.wait_num_max
        else:
            m_syn = len(deprecated_ids) + self.wait_num_max

        self.sync_count += m_syn  # count sync. overheads

        # update arrival time for undrafted clients
        dist_time = self.model_size * m_syn / self.bw_s_set
        bake_clients_arrival_T = self.clients_arrival_T
        for c_id in range(self.n_clients):
            if c_id in self.undrafted_ids:
                self.clients_arrival_T[c_id] = max(0, (self.clients_arrival_T[c_id] - bake_clients_arrival_T[
                    self.max_id] - dist_time))  # Arrival time minus waiting time of this round
            else:
                self.bw_c_set[c_id]=self.bw_c_set[c_id]+round(random.uniform(-0.002,0.002),3)
                while self.bw_c_set[c_id]<=0:
                    self.bw_c_set[c_id]=self.bw_c_set[c_id] + round(random.uniform(0, 0.004), 3)
                T_comm = self.model_size / self.bw_c_set[c_id]
                T_train = self.client_shard_sizes[c_id] / self.batch_size * self.n_epochs / self.clients_perf_vec[c_id]
                self.clients_arrival_T[c_id] = T_comm+T_train


        """select K clients at current round"""
        # simulate device or network failure
        crash_ids = self.crash_trace[rd]
        # 去除掉掉线的客户端
        # availabel_ids = [c_id for c_id in range(self.n_clients) if c_id not in crash_ids]
        availabel_ids = [c_id for c_id in range(self.n_clients)]
        # 去除掉仍在运行的客户端
        selected_ids = [c_id for c_id in availabel_ids if c_id not in self.undrafted_ids]
        # get clients' utility
        bake_models = copy.deepcopy(self.models)
        if rd==0:
            _,self.clients_loss_sum = train(bake_models, selected_ids, self.args, self.cm_map, self.fed_loader_train,
                                      self.reporting_train_loss_vec,self.clients_loss_sum, mode=1, verbose=False)
        make_ids = Select_adaptive(selected_ids, self.pick_C, self.n_clients, self.clients_loss_sum,
                                   self.client_shard_sizes)
        # distribute newest models to selected clients
        distribute_models(self.global_model, self.models, make_ids)  # up-to-version clients will sync. (sync.)

        # compensatory first-come-first-merge selection, last-round picks are considered low priority
        # print('> Clients undrafted: ', self.undrafted_ids)
        # print('> Clients make_ids: ', make_ids)
        self.picked_ids, self.undrafted_ids, self.max_id = select_clients_CFCFM(make_ids, self.undrafted_ids, self.clients_arrival_T, quota)

        # self.undrafted_ids = [c_id for c_id in make_ids if c_id not in picked_ids]
        # tracing
        # self.make_trace.append(make_ids)
        # self.pick_trace.append(self.picked_ids)
        # self.undrafted_trace.append(self.undrafted_ids)
        # print('> Clients crashed: ', crash_ids)
        # print('> Clients undrafted: ', self.undrafted_ids)
        # print('> Clients picked: ', self.picked_ids)  # first-come-first-merge



        # Local training step
        for epo in range(self.n_epochs):  # local epochs (same # of epochs for each client)
            # print('\n> @Devices> local epoch #%d' % epo)
            self.reporting_train_loss_vec,self.clients_loss_sum = train(self.models, make_ids, self.args, self.cm_map, self.fed_loader_train,
                                             self.reporting_train_loss_vec,self.clients_loss_sum, mode=0, verbose=False)
            # add to trace
            # self.epoch_train_trace.append(
            #     np.array(self.reporting_train_loss_vec) / (np.array(self.client_shard_sizes) * self.train_pct))
            # print('>   @Devices> %d clients train loss vector this epoch:' % env_cfg.n_clients,
            #       np.array(self.reporting_train_loss_vec) / (np.array(client_shard_sizes) * env_cfg.train_pct))
            # local test reports
            # self.reporting_test_loss_vec = local_test(self.models, make_ids, self.args, self.cm_map, self.fed_loader_test,
            #                                      self.reporting_test_loss_vec)
            # add to trace
            # self.epoch_test_trace.append(
            #     np.array(self.reporting_test_loss_vec) / (np.array(self.client_shard_sizes) * self.test_pct))
            # print('>   @Devices> %d clients test loss vector this epoch:' % env_cfg.n_clients,
            #       np.array(self.reporting_test_loss_vec) / (np.array(client_shard_sizes) * env_cfg.test_pct))

        # Aggregation step
        # discriminative update of cloud cache and aggregate
        # pre-aggregation: update cache from picked clients
        update_cloud_cache(self.cache, self.models, self.picked_ids)
        # print('\n> Aggregation step (Round #%d)' % rd)

        self.global_model = safa_aggregate(self.global_model,self.cache, self.picked_ids, self.client_shard_sizes, self.varsigma,self.versions,rd)  # aggregation
        # # post-aggregation: update cache from undrafted clients
        # update_cloud_cache(cache, models, self.undrafted_ids)

        # versioning
        eu = len(self.picked_ids)  # effective updates
        self.eu_count += eu  # EUR
        self.version_var += np.var(self.versions[availabel_ids])  # Version Variance
        update_versions_ids=[i for i in self.client_ids if i not in self.undrafted_ids]
        update_versions(self.versions, update_versions_ids, rd)
        # print('>   @Cloud> Versions updated:', self.versions)

        # Reporting phase: distributed test of the global model
        post_aggre_loss_vec, acc = global_test(self.global_model, self.args, self.cm_map, self.fed_loader_test)
        # print('>   @Devices> post-aggregation loss reports  = ',
        #       np.array(post_aggre_loss_vec) / ((np.array(self.client_shard_sizes)) * self.test_pct))
        # overall loss, i.e., objective (1) in McMahan's paper
        overall_loss = np.array(post_aggre_loss_vec).sum() / (self.data_size * self.test_pct)
        # update so-far best
        if overall_loss < self.best_loss:
            self.best_loss = overall_loss
            self.best_acc = acc
            self.best_model = self.global_model
            self.best_rd = rd
        if self.keep_best:  # if to keep best
            self.global_model = self.best_model
            overall_loss = self.best_loss
            acc = self.best_acc
        # print('>   @Cloud> post-aggregation loss avg = ', overall_loss)
        self.round_trace.append(overall_loss)
        self.acc_trace.append(acc)

        bake_clients_arrival_T = self.clients_arrival_T
        dist_time= self.model_size * m_syn / self.bw_s_set  # T_disk = model_size * N_sync / BW
        # update timers
        for c_id in range(self.n_clients):
            if c_id in make_ids:
                # timers
                client_round_timers[c_id] = bake_clients_arrival_T[c_id]  # including comm. and training
                # client_round_comm_timers[c_id] = T_comm  # comm. is part of the run time
                self.client_timers[c_id] += client_round_timers[c_id]  # sum up
                # self.client_comm_timers[c_id] += client_round_comm_timers[c_id]  # sum up
                if c_id in self.picked_ids:
                    picked_client_round_timers[c_id] = client_round_timers[c_id]  # we need to await the picked
                if c_id in deprecated_ids:  # deprecated clients, forced to sync. at distributing step
                    self.client_futile_timers[c_id] += self.progress_trace[rd][c_id] * client_round_timers[c_id]
                    client_round_timers[c_id] = bake_clients_arrival_T[self.max_id]   # no response

        # round_response_time = min(response_time_limit, max(picked_client_round_timers))  # w8 to meet quota
        # global_timer += dist_time + round_response_time
        # global_T_dist_timer += dist_time
        # Event updates
        T_k = self.event_handler.get_state('time')
        T_limit = (self.T_max - T_k) / (self.n_rounds - rd )
        self.event_handler.add_parallel('time', picked_client_round_timers, reduce='max')
        self.event_handler.add_sequential('time', dist_time)
        # self.event_handler.add_obeject('obj_acc_time', picked_client_round_timers, dist_time, T_limit)
        # self.event_handler.add_sequential('T_dist', dist_time)

        # print('> Round client run time:', client_round_timers)  # round estimated finish time
        stats=[]
        stats.append(overall_loss)
        total_time= self.event_handler.get_state('time')
        if total_time <= self.T_max:
            omega1 = self.alpha
        else:
            omega1 = self.beta
        # time_ratio=acc*(self.T_max/total_time)**omega1
        time_ratio=total_time/self.T_max
        stats.append(time_ratio)
        max_round_t=max(picked_client_round_timers)

        if max_round_t==0.0:
            obj_acc_time = 2
        else:
            T_round = max_round_t + dist_time
            if T_limit<=0.0:
                T_limit=self.T_max/ self.n_rounds
                omega = 0.03
            else:
                if T_round<=T_limit:
                    omega=self.alpha
                else:
                    omega = self.beta
            obj_acc_time = (T_limit/T_round)**(omega)
        stats.append(acc*obj_acc_time)
        # eu_ratio = self.eu_count / (rd+1) / self.n_clients #当前轮次的有效更新率=当前实际上传客户数/(当前轮数*总客户数)
        # stats.append(eu_ratio)
        sync_ratio = self.sync_count / (rd+1) / self.n_clients #当前同步率=当前强制同步客户数/(当前轮数*总客户数)
        stats.append(sync_ratio)
        version_var = self.version_var/(rd+1)#当前版本号方差=当前版本号方差累加和/当前轮数
        stats.append(version_var)
        # undrafted_ratio=len(self.undrafted_ids)/self.n_clients
        # stats.append(undrafted_ratio)


        return stats,obj_acc_time,picked_client_round_timers,acc,total_time



def action_choice(actions):

    return actions[0], actions[1]


