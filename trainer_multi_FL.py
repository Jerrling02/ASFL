#import gym
import datetime
import torch
from torchvision import datasets as torch_datasets
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from Enviroment import Federate_learning as FL
import FL.FLLocalSupport as FLSup
from FL.option import args_parser
from tqdm import tqdm
import FL.utils as utils
import random
# import syft as sy
# import FL.semiAysnc_reinforce as semiAysnc_reinforce
from FL.learning_tasks import MLmodelReg, MLmodelCNN
from FL.learning_tasks import MLmodelSVM
#from Discrete_SAC_Agent import SACAgent
#from multi_action_Agent import SACAgent

TRAINING_EVALUATION_RATIO = 4
RUNS = 20
EPISODES_PER_RUN = 1000
STEPS_PER_EPISODE = 200
# layer_num = 21


import torch


#from utilities.ReplayBuffer import ReplayBuffer



class Actor(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation, lag_t_max,wait_num_max,LOAD,PATH):
        super(Actor, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=128)
        self.layer_2 = torch.nn.Linear(in_features=128, out_features=128)
        self.layer_out1 = torch.nn.Linear(in_features=128, out_features = wait_num_max)
        self.layer_out2 = torch.nn.Linear(in_features=128, out_features = lag_t_max)
        self.output_layer = torch.nn.Linear(in_features=128, out_features=output_dimension)
        self.normal1=torch.nn.LayerNorm(normalized_shape=wait_num_max, eps=0, elementwise_affine=False)
        self.normal2 = torch.nn.LayerNorm(normalized_shape=lag_t_max, eps=0, elementwise_affine=False)
        self.output_activation = output_activation
        if LOAD:
            self.load_model(PATH)
        # self.num=num_edges


    def forward(self, inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))


        # y=self.output_activation(self.normal(self.layer_out1(layer_2_output)))
        x = []
        x.append(self.output_activation(self.normal1(self.layer_out1(layer_2_output))))
        x.append(self.output_activation(self.normal2(self.layer_out2(layer_2_output))))
        # for i in range(self.num):
        #     # x.append(self.output_activation(self.layer_out1(layer_2_output)))
        #     x.append(self.output_activation(self.layer_out2(layer_2_output)))

        output = x[0]
        for i in range(1, len(x)):
            output = torch.cat([output, x[i]], dim=1)
        # output = torch.cat([output, y], dim=1)
        # output = self.output_activation(self.layer_out2(layer_2_output))

        return output

    def load_model(self, PATH):
        self.load_state_dict(torch.load(PATH))

class Critic(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation=torch.nn.Identity()):
        super(Critic, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=128)
        self.layer_2 = torch.nn.Linear(in_features=128, out_features=64)
        self.output_layer = torch.nn.Linear(in_features=64, out_features=output_dimension)
        self.output_activation = output_activation

    def forward(self, inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = self.output_layer(layer_2_output)


        output = self.output_activation(layer_3_output)
        return output

class SACAgent:

    ALPHA_INITIAL = 1.
    REPLAY_BUFFER_BATCH_SIZE = 200
    DISCOUNT_RATE = 0.09
    LEARNING_RATE = 10 ** -6
    SOFT_UPDATE_INTERPOLATION_FACTOR = 0.01

    def __init__(self, environment,LOAD=False,Path=None):
        self.environment = environment
        self.num_clients = environment.n_clients
        self.lag_t_max = environment.lag_t_max
        self.wait_num_max= environment.wait_num_max
        self.state_dim = environment.state_space
        self.action_dim = environment.action_space
        #self.action_dim = 3*20
        #self.state_dim = self.environment.observation_space.shape[0]
        #self.action_dim = self.environment.action_space.n
        self.critic_local = Critic(input_dimension=self.state_dim,
                                    output_dimension=self.action_dim)
        self.critic_local2 = Critic(input_dimension=self.state_dim,
                                     output_dimension=self.action_dim)
        self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.LEARNING_RATE)
        self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=self.LEARNING_RATE)

        self.critic_target = Critic(input_dimension=self.state_dim,
                                     output_dimension=self.action_dim)
        self.critic_target2 = Critic(input_dimension=self.state_dim,
                                      output_dimension=self.action_dim)

        self.soft_update_target_networks(tau=1.)

        self.actor_local = Actor(
            input_dimension=self.state_dim,
            output_dimension=self.action_dim,
            output_activation=torch.nn.Softmax(dim=1),
            lag_t_max=self.lag_t_max,
            wait_num_max=self.wait_num_max,
            LOAD=LOAD,
            PATH=Path

        )
        self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.LEARNING_RATE)

        self.replay_buffer = ReplayBuffer(self.environment)

        self.target_entropy = 0.98 * -np.log(1 / self.action_dim)
        self.log_alpha = torch.tensor(np.log(self.ALPHA_INITIAL), requires_grad=True)
        self.alpha = self.log_alpha
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.LEARNING_RATE)

    def get_next_action(self, state, evaluation_episode=False):
        if evaluation_episode:
            discrete_action = self.get_action_deterministically(state)
        else:
            discrete_action = self.get_action_nondeterministically(state)
        return discrete_action

    def get_action_nondeterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action =[]
        # action=wait_num_p+lag_torlerant_p
        #get wait_num prob
        wait_num_probabilities = action_probabilities[:self.wait_num_max]
        #get lag_t prob
        lag_t_probabilities=action_probabilities[self.wait_num_max:self.lag_t_max+self.wait_num_max]

        discrete_action.append(np.random.choice(range(0, self.wait_num_max),
                                                    p=wait_num_probabilities)+1)
        discrete_action.append(np.random.choice(range(0, self.lag_t_max),
                                                p=lag_t_probabilities))
        # client_choice=list(np.random.choice(range(0, self.num_clients), size=self.client_select_size, p=client_actions,replace=False))
        # discrete_action=discrete_action+client_choice
        return discrete_action

    def get_action_deterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = []
        a=float("NaN")
        test_list_set = set(action_probabilities)
        if a in test_list_set:
            print("存在")
        # action=wait_num_p+lag_torlerant_p
        # get wait_num prob
        wait_num_probabilities = action_probabilities[:self.wait_num_max]
        # get lag_t prob
        lag_t_probabilities = action_probabilities[self.wait_num_max:self.lag_t_max + self.wait_num_max]

        discrete_action.append(np.argmax(wait_num_probabilities)+1)
        discrete_action.append(np.argmax(lag_t_probabilities))
        # for i in range(self.client_select_size):
        #     index=np.argmax(client_actions)
        #     client_actions[index]=0
        #     discrete_action.append(index)
        # client_choice=list(np.random.choice(range(0, self.num_clients), size=8, p=client_actions,replace=False))
        # discrete_action=discrete_action+client_choice\
        return discrete_action


    def train_on_transition(self, state, discrete_action, next_state, reward, done):
        # count = 0
        # for i in range(self.num_edges):
        #     # discrete_action[2*i] = discrete_action[2*i] + count
        #     # discrete_action[i] = discrete_action[i] + count
        #     # count += self.num_max
        #     # discrete_action[2*i + 1] = discrete_action[2*i + 1] + count
        #     discrete_action[i] = discrete_action[i] + count
        #     count += self.tau_max
        # idx=self.num_edges
        # for i in range(len(discrete_action)-self.num_edges):
        #     discrete_action[i+idx] = discrete_action[i+idx] + idx
        discrete_action[0] = discrete_action[0] - 1
        discrete_action[-1]=discrete_action[-1]+self.wait_num_max
        transition = (state, discrete_action, reward, next_state, done)
        self.train_networks(transition)

    def train_networks(self, transition):
        # Set all the gradients stored in the optimisers to zero.
        self.critic_optimiser.zero_grad()
        self.critic_optimiser2.zero_grad()
        self.actor_optimiser.zero_grad()
        self.alpha_optimiser.zero_grad()
        # Calculate the loss for this transition.
        self.replay_buffer.add_transition(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network
        # parameters.
        if self.replay_buffer.get_size() >= self.REPLAY_BUFFER_BATCH_SIZE:
            # get minibatch of 100 transitions from replay buffer
            minibatch = self.replay_buffer.sample_minibatch(self.REPLAY_BUFFER_BATCH_SIZE)
            minibatch_separated = list(map(list, zip(*minibatch)))

            # unravel transitions to get states, actions, rewards and next states
            states_tensor = torch.tensor(np.array(minibatch_separated[0]), dtype=torch.float32)
            actions_tensor = torch.tensor(np.array(minibatch_separated[1]),dtype=torch.float32)
            rewards_tensor = torch.tensor(np.array(minibatch_separated[2])).float()
            next_states_tensor = torch.tensor(np.array(minibatch_separated[3]))
            done_tensor = torch.tensor(np.array(minibatch_separated[4]))
            #actions_tensor_2 = torch.tensor(np.array(minibatch_separated[5]), dtype=torch.float32)

            critic_loss, critic2_loss = \
                self.critic_loss(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor)

            critic_loss.backward()
            critic2_loss.backward()
            self.critic_optimiser.step()
            self.critic_optimiser2.step()

            actor_loss, log_action_probabilities = self.actor_loss(states_tensor)

            actor_loss.backward()
            self.actor_optimiser.step()

            alpha_loss = self.temperature_loss(log_action_probabilities)

            alpha_loss.backward()
            self.alpha_optimiser.step()
            self.alpha = self.log_alpha.exp()

            self.soft_update_target_networks()

    def critic_loss(self, states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor):
        with torch.no_grad():
            action_probabilities, log_action_probabilities = self.get_action_info(next_states_tensor)
            next_q_values_target = self.critic_target.forward(next_states_tensor)
            next_q_values_target2 = self.critic_target2.forward(next_states_tensor)
            soft_state_values = (action_probabilities * (
                    torch.min(next_q_values_target, next_q_values_target2) - self.alpha * log_action_probabilities
            )).sum(dim=1)

            next_q_values = rewards_tensor + ~done_tensor * self.DISCOUNT_RATE*soft_state_values

        #actions = []
        #num = self.vehicle_num * 2
        temp = torch.split(actions_tensor, 1, dim = 1)


        #soft_q_values = self.critic_local(states_tensor).gather(1, actions_tensor.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        '''
        soft_q_values = self.critic_local(states_tensor)
        soft_q_values_1 = soft_q_values.gather(1, actions_tensor_1.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        soft_q_values_2 = soft_q_values.gather(1, actions_tensor_2.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        soft_q_values2 = self.critic_local2(states_tensor)
        soft_q_values2_1 = soft_q_values2.gather(1, actions_tensor_1.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        soft_q_values2_2 = soft_q_values2.gather(1, actions_tensor_2.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        '''
        soft_q_value = self.critic_local(states_tensor)
        soft_q_value2 = self.critic_local2(states_tensor)
        soft_q_values = []
        soft_q_values2 = []

        #a = temp[1].type(torch.int64).squeeze()

        for i in range(len(temp)):
            soft_q_values.append(soft_q_value.gather(1, temp[i].type(torch.int64).squeeze().unsqueeze(-1)).squeeze(-1))
            soft_q_values2.append(soft_q_value2.gather(1, temp[i].type(torch.int64).squeeze().unsqueeze(-1)).squeeze(-1))
        critic_square_error = torch.nn.MSELoss(reduction="none")(sum(soft_q_values), next_q_values)
        critic2_square_error = torch.nn.MSELoss(reduction="none")(sum(soft_q_values2), next_q_values)
        weight_update = [min(l1.item(), l2.item()) for l1, l2 in zip(critic_square_error, critic2_square_error)]
        self.replay_buffer.update_weights(weight_update)
        critic_loss = critic_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        return critic_loss, critic2_loss

    def actor_loss(self, states_tensor,):
        action_probabilities, log_action_probabilities = self.get_action_info(states_tensor)
        q_values_local = self.critic_local(states_tensor)
        q_values_local2 = self.critic_local2(states_tensor)
        inside_term = self.alpha * log_action_probabilities - torch.min(q_values_local, q_values_local2)
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        return policy_loss, log_action_probabilities

    def temperature_loss(self, log_action_probabilities):
        alpha_loss = -(self.log_alpha * (log_action_probabilities + self.target_entropy).detach()).mean()
        return alpha_loss

    def get_action_info(self, states_tensor):
        action_probabilities = self.actor_local.forward(states_tensor)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities

    def get_action_probabilities(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probabilities = self.actor_local.forward(state_tensor)
        return action_probabilities.squeeze(0).detach().numpy()

    def soft_update_target_networks(self, tau=SOFT_UPDATE_INTERPOLATION_FACTOR):
        self.soft_update(self.critic_target, self.critic_local, tau)
        self.soft_update(self.critic_target2, self.critic_local2, tau)

    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def predict_q_values(self, state):
        q_values = self.critic_local(state)
        q_values2 = self.critic_local2(state)
        return torch.min(q_values, q_values2)


class ReplayBuffer:

    def __init__(self, environment, capacity=5000):
        transition_type_str = self.get_transition_type_str(environment)
        self.buffer = np.zeros(capacity, dtype=transition_type_str)
        self.weights = np.zeros(capacity)
        self.head_idx = 0
        self.count = 0
        self.capacity = capacity
        self.max_weight = 10**-2
        self.delta = 10**-4
        self.indices = None

    def get_transition_type_str(self, environment):
        #state_dim = environment.observation_space.shape[0]
        state_dim = environment.state_space
        state_dim_str = '' if state_dim == () else str(state_dim)
        #state_type_str = environment.observation_space.sample().dtype.name
        state_type_str = "float32"
        #action_dim = "2"
        action_dim = 2
        #action_dim = environment.action_space.shape
        action_dim_str = '' if action_dim == () else str(action_dim)
        #action_type_str = environment.action_space.sample().__class__.__name__
        action_type_str = "int"

        # type str for transition = 'state type, action type, reward type, state type'
        transition_type_str = '{0}{1}, {2}{3}, float32, {0}{1}, bool'.format(state_dim_str, state_type_str,
                                                                             action_dim_str, action_type_str)

        return transition_type_str

    def add_transition(self, transition):
        self.buffer[self.head_idx] = transition
        self.weights[self.head_idx] = self.max_weight

        self.head_idx = (self.head_idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample_minibatch(self, size=100):
        set_weights = self.weights[:self.count] + self.delta
        probabilities = set_weights / sum(set_weights)
        self.indices = np.random.choice(range(self.count), size, p=probabilities, replace=False)
        return self.buffer[self.indices]

    def update_weights(self, prediction_errors):
        max_error = max(prediction_errors)
        self.max_weight = max(self.max_weight, max_error)
        self.weights[self.indices] = prediction_errors

    def get_size(self):
        return self.count
#----------------------semi-FL functions------------------------------------------
def save_KddCup99_tcpdump_tcp_tofile(fpath):
    """
    Fetch (from sklearn.datasets) the KddCup99 tcpdump dataset, extract tcp samples, and save to local as csv
    :param fpath: local file path to save the dataset
    :return: KddCup99 dataset, tcp-protocol samples only
    """
    xy = utils.fetch_KddCup99_10pct_tcpdump(return_X_y=False)
    np.savetxt(fpath, xy, delimiter=',', fmt='%.6f',
               header='duration, src_bytes, dst_bytes, land, urgent, hot, #failed_login, '
                      'logged_in, #compromised, root_shell, su_attempted, #root, #file_creations, #shells, '
                      '#access_files, is_guest_login, count, srv_cnt, serror_rate, srv_serror_rate, rerror_rate, '
                      'srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate, dst_host_cnt,'
                      'dst_host_srv_cnt, dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rt,'
                      'dst_host_srv_diff_host_rate, dst_host_serror_rate, dst_host_srv_serror_rate, '
                      'dst_host_rerror_rate, dst_host_srv_rerror_rate, label')


def init_FL_clients(nc):
    """
    # Create clients and a client-index map
    :param nc:  # of cuda clients
    :return: clients list, and a client-index map
    """
    clients = []
    cm_map = {}
    for i in range(nc):
        clients.append(FLSup.FLClient(id='client_' + str(i)))
        cm_map['client_' + str(i)] = i  # client i with model i
    return clients, cm_map

def init_models(env_cfg):
    """
    Initialize models as per the settings of FL and machine learning task
    :param env_cfg:
    :return: models
    """
    models = []
    dev = env_cfg.device
    # have to transiently set default tensor type to cuda.float, otherwise model.to(dev) fails on GPU
    if dev.type == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # instantiate models, one per client
    for i in range(env_cfg.n_clients):
        if env_cfg.task_type == 'Reg':
            models.append(MLmodelReg(in_features=env_cfg.in_dim, out_features=env_cfg.out_dim).to(dev))
        elif env_cfg.task_type == 'SVM':
            models.append(MLmodelSVM(in_features=env_cfg.in_dim).to(dev))
        elif env_cfg.task_type == 'CNN':
            models.append(MLmodelCNN(classes=10).to(dev))

    torch.set_default_tensor_type('torch.FloatTensor')
    return models


def init_glob_model(env_cfg):
    """
    Initialize the global model as per the settings of FL and machine learning task

    :param env_cfg:
    :return: model
    """
    model = None
    dev = env_cfg.device
    # have to transiently set default tensor type to cuda.float, otherwise model.to(dev) fails on GPU
    if dev.type == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # instantiate models, one per client
    if env_cfg.task_type == 'Reg':
        model = MLmodelReg(in_features=env_cfg.in_dim, out_features=env_cfg.out_dim).to(dev)
    elif env_cfg.task_type == 'SVM':
        model = MLmodelSVM(in_features=env_cfg.in_dim).to(dev)
    elif env_cfg.task_type == 'CNN':
        model = MLmodelCNN(classes=10).to(dev)

    torch.set_default_tensor_type('torch.FloatTensor')
    return model


def generate_clients_perf(env_cfg, from_file=False, s0=1e-2):
    """
    Generate a series of client performance values (in virtual time unit) following the specified distribution
    :param env_cfg: environment config file
    :param from_file: if True, load client performance distribution from file
    :param s0: lower bound of performance
    :return: a list of client's performance, measured in virtual unit
    """
    if from_file:
        fname = 'gen/clients_perf_'+str(env_cfg.n_clients)
        return np.loadtxt(fname)

    n_clients = env_cfg.n_clients
    perf_vec = None
    # Case 1: Equal performance
    if env_cfg.perf_dist[0] == 'E':  # ('E', None)
        perf_vec = [1.0 for _ in range(n_clients)]
        while np.min(perf_vec) < s0:  # in case of super straggler
            perf_vec = [1.0 for _ in range(n_clients)]

    # Case 2: eXponential distribution of performance
    elif env_cfg.perf_dist[0] == 'X':  # ('X', None), lambda = 1/1, mean = 1
        perf_vec = [random.expovariate(1.0) for _ in range(n_clients)]
        while np.min(perf_vec) < s0:  # in case of super straggler
            perf_vec = [random.expovariate(1.0) for _ in range(n_clients)]

    # Case 3: Normal distribution of performance
    elif env_cfg.perf_dist[0] == 'N':  # ('N', rlt_sigma), mu = 1, sigma = rlt_sigma * mu
        perf_vec = [0.0 for _ in range(n_clients)]
        for i in range(n_clients):
            perf_vec[i] = random.gauss(1.0, env_cfg.perf_dist[1] * 1.0)
            while perf_vec[i] <= s0:  # in case of super straggler
                perf_vec[i] = random.gauss(1.0, env_cfg.perf_dist[1] * 1.0)
    else:
        print('Error> Invalid client performance distribution option')
        exit(0)

    return perf_vec


def generate_clients_crash_prob(env_cfg):
    """
    Generate a series of probability that the corresponding client would crash (including device and network down)
    :param env_cfg: environment config file
    :return: a list of client's crash probability, measured in virtual unit
    """
    n_clients = env_cfg.n_clients
    prob_vec = None
    # Case 1: Equal prob
    if env_cfg.crash_dist[0] == 'E':  # ('E', prob)
        prob_vec = [env_cfg.crash_dist[1] for _ in range(n_clients)]

    # Case 2: uniform distribution of crashing prob
    elif env_cfg.crash_dist[0] == 'U':  # ('U', (low, high))
        low = env_cfg.crash_dist[1][0]
        high = env_cfg.crash_dist[1][1]
        # check
        if low < 0 or high < 0 or low > 1 or high > 1 or low >= high:
            print('Error> Invalid crash prob interval')
            exit(0)
        prob_vec = [random.uniform(low, high) for _ in range(n_clients)]
    else:
        print('Error> Invalid crash prob distribution option')
        exit(0)

    return prob_vec


def generate_crash_trace(env_cfg, clients_crash_prob_vec):
    """
    Generate a crash trace (length=# of rounds) for simulation,
    making every FA algorithm shares the same trace for fairness
    :param env_cfg: env config
    :param clients_crash_prob_vec: client crash prob. vector
    :return: crash trace as a list of lists, and a progress trace
    """
    crash_trace = []
    progress_trace = []
    for r in range(env_cfg.n_rounds):
        crash_ids = []  # crashed ones this round
        progress = [1.0 for _ in range(env_cfg.n_clients)]  # 1.0 denotes well progressed
        for c_id in range(env_cfg.n_clients):
            rand = random.random()
            if rand <= clients_crash_prob_vec[c_id]:  # crash
                crash_ids.append(c_id)
                progress[c_id] = rand / clients_crash_prob_vec[c_id]  # progress made before crash

        crash_trace.append(crash_ids)
        progress_trace.append(progress)

    return crash_trace, progress_trace
if __name__ == "__main__":

    np.random.seed(1)

    args = args_parser()
    env_cfg = args
    # params to tune
    cr_prob = args.cr_prob  # float(sys.argv[1])  # E(cr)
    lag_tol = args.lag_tol  # int(sys.argv[2])  # lag tolerance, for SAFA
    # wait_num = args.wait_num

    utils.show_settings(env_cfg, detail=False, detail_info=None)

    # load data
    if env_cfg.dataset == 'Boston':
        data = np.loadtxt(env_cfg.path, delimiter=',', skiprows=1)
        data = utils.normalize(data)
        data_merged = True
    elif env_cfg.dataset == 'tcpdump99':
        data = np.loadtxt(env_cfg.path, delimiter=',', skiprows=1)
        data = utils.normalize(data, expt=-1)  # normalize features but not labels (+1/-1 for SVM)
        data_merged = True
    elif env_cfg.dataset == 'mnist':
        # ref: https://github.com/pytorch/examples/blob/master/mnist/main.py
        mnist_train = torch_datasets.MNIST('data/mnist/', train=True, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        mnist_test = torch_datasets.MNIST('data/mnist/', train=False, download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        data_train_x = mnist_train.data.view(-1, 1, 28, 28).float()
        data_train_y = mnist_train.targets.long()
        data_test_x = mnist_test.data.view(-1, 1, 28, 28).float()
        data_test_y = mnist_test.targets.long()

        train_data_size = len(data_train_x)
        test_data_size = len(data_test_x)
        data_size = train_data_size + test_data_size
        data_merged = False
    else:
        print('E> Invalid dataset specified')
        exit(-1)
    # partition into train/test set, for Boston and Tcpdump data
    if data_merged:
        data_size = len(data)
        train_data_size = int(data_size * env_cfg.train_pct)
        test_data_size = data_size - train_data_size
        data = torch.tensor(data).float()
        data_train_x = data[0:train_data_size, 0:env_cfg.in_dim]  # training data, x
        data_train_y = data[0:train_data_size, env_cfg.out_dim * -1:].reshape(-1, env_cfg.out_dim)  # training data, y
        data_test_x = data[train_data_size:, 0:env_cfg.in_dim]  # test data following, x
        data_test_y = data[train_data_size:, env_cfg.out_dim * -1:].reshape(-1, env_cfg.out_dim)  # test data, x

    clients, c_name2idx = init_FL_clients(env_cfg.n_clients)  # create clients and a client-index map
    fed_data_train, fed_data_test, client_shard_sizes = utils.get_FL_datasets(data_train_x, data_train_y,
                                                                              data_test_x, data_test_y,
                                                                              env_cfg, clients)
    # pseudo distributed data loaders, by Syft
    # fed_loader_train = sy.FederatedDataLoader(fed_data_train, shuffle=env_cfg.shuffle, batch_size=env_cfg.batch_size)
    # fed_loader_test = sy.FederatedDataLoader(fed_data_test, shuffle=env_cfg.shuffle, batch_size=env_cfg.batch_size)
    fed_loader_train = FLSup.SimpleFedDataLoader(fed_data_train, c_name2idx,
                                                 batch_size=env_cfg.batch_size, shuffle=env_cfg.shuffle)
    fed_loader_test = FLSup.SimpleFedDataLoader(fed_data_test, c_name2idx,
                                                batch_size=env_cfg.batch_size, shuffle=env_cfg.shuffle)
    print('> %d clients data shards (data_dist = %s):' % (env_cfg.n_clients, env_cfg.data_dist[0]), client_shard_sizes)

    # prepare simulation
    # client performance = # of mini-batched able to process in one second
    clients_perf_vec = generate_clients_perf(env_cfg, from_file=True)  # generated, s0= 0.05,0.016,0.012 for Task 1,2,3

    print('> Clients perf vec:', clients_perf_vec)

    # client crash probability
    clients_crash_prob_vec = generate_clients_crash_prob(env_cfg)
    print('> Clients crash prob. vec:', clients_crash_prob_vec)
    # crash trace simulation
    crash_trace, progress_trace = generate_crash_trace(env_cfg, clients_crash_prob_vec)





    ##########################################################################################################
    #                                         LOCAL_ML                                                       #
    ##########################################################################################################
    # all_in_one(args, data_distribution)
    ##########################################################################################################
    #                                         FL                                                             #
    ##########################################################################################################
    # compare(dataloaders, locations_list)

    ##########################################################################################################
    #                                         RL                                                             #
    ##########################################################################################################
    env = FL(args, c_name2idx, data_size, fed_loader_train, fed_loader_test,
             client_shard_sizes, clients_perf_vec, clients_crash_prob_vec, crash_trace, progress_trace)

    agent_results = []
    plot_x = np.zeros(0)
    plot_y = np.zeros(0)
    plot_reward = np.zeros(0)
    plot_acc = np.zeros(0)
    plot_cost = np.zeros(0)
    plot_all_reward = np.zeros(0)
    plot_fake_reward = np.zeros(0)
    # print("model:%s, data_dist:%.1f, target_acc:%.4f, curr_cluster:%s, num_clients:%d"
    #       % (args.model, args.data_distribution, args.target_acc, args.edge_choice, args.num_clients))
    lag_t_max=args.lag_t_max
    wait_num_max=int(args.n_clients*args.pick_C)
    RL_path = args.RL_path
    reward_path = args.reward_path
    actor_path = args.actor_path
    load = args.load
    if load:
        EPISODES_PER_RUN=1
    for run in range(RUNS):
        agent = SACAgent(env,load,args.actor_path)
        run_results = []
        for episode_number in range(EPISODES_PER_RUN):
            evaluation_episode = episode_number % TRAINING_EVALUATION_RATIO == 0

            episode_reward = 0
            fake_reward = 0
            # real_episode_reward = 0
            # state_start = datetime.datetime.now()
            state = env.reset()
            # state_end = datetime.datetime.now()
            done = False
            for ep in tqdm(range(args.n_rounds)):
                if done:
                    break
                if ep == args.n_rounds-1:
                    done=True
                # action = agent.get_next_action(state,evaluation_episode)
                action = agent.get_action_nondeterministically(state)
                next_state, reward,picked_client_round_timers,acc,total_time = env.step(action,ep)
                reward=reward*10
                if not evaluation_episode:
                    agent.train_on_transition(state, action, next_state, reward, done)
                episode_reward += reward
                if ep == args.n_rounds-1:
                    if total_time<=env.T_max:
                        omega=env.alpha
                    else:
                        omega =env.beta
                    fake_reward = acc*(env.T_max/total_time)**omega
                    print("total_time:{}, acc:{}, objective:{}".format(total_time, acc, fake_reward))
                    # print("acc:", round(acc,3))
                plot_x = np.append(plot_x, args.n_rounds * run + ep)
                plot_acc = np.append(plot_acc, acc)
                # print("acc: ",acc)
                # plot_reward = np.append(plot_reward, reward)
                plot_cost = np.append(plot_cost, total_time)
                np.savez(RL_path, plot_x, plot_acc,plot_cost)
                state = next_state
            if evaluation_episode:
                print(f'Run: {run + 1}/{RUNS} | Episode: {episode_number + 1}/{EPISODES_PER_RUN}', end=' ')
                print("episoed:%d, reward:%.5f , ep:%d" %(episode_number, episode_reward,ep))
                # run_results.append(episode_reward)
            if episode_number==EPISODES_PER_RUN-1:
                if not load:
                    torch.save(agent.actor_local.state_dict(), "actor_model"+args.task_type+str(args.data_dist)+"_ep"+str(episode_number)+"_runs"+str(run))
            plot_y = np.append(plot_y, run)
            # plot_all_reward = np.append(plot_all_reward, real_episode_reward)
            plot_reward = np.append(plot_reward, episode_reward)
            plot_fake_reward = np.append(plot_fake_reward, fake_reward)
            np.savez(reward_path, plot_y,plot_reward, plot_fake_reward)
            torch.save(agent.actor_local.state_dict(), actor_path)
        agent_results.append(run_results)
    # env.close()

    n_results = EPISODES_PER_RUN // TRAINING_EVALUATION_RATIO
    results_mean = [np.mean([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    results_std = [np.std([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    mean_plus_std = [m + s for m, s in zip(results_mean, results_std)]
    mean_minus_std = [m - s for m, s in zip(results_mean, results_std)]

    x_vals = list(range(len(results_mean)))
    x_vals = [x_val * (TRAINING_EVALUATION_RATIO - 1) for x_val in x_vals]

    print("lr = 0.001")

    ax = plt.gca()
    ax.set_ylim([-30, 0])
    ax.set_ylabel('Episode Score')
    ax.set_xlabel('Training Episode')
    ax.plot(x_vals, results_mean, label='Average Result', color='#19CAAD')
    ax.plot(x_vals, mean_plus_std, color='#19CAAD', alpha=0.1)
    ax.fill_between(x_vals, y1=mean_minus_std, y2=mean_plus_std, alpha=0.1, color='#19CAAD')
    ax.plot(x_vals, mean_minus_std, color='#19CAAD', alpha=0.1)
    plt.legend(loc='best')
    plt.show()
