# FLLocalSupport.py
# Complementary local support codes to Pytorch FL learning
# @Author  : wwt
# @Date    : 2019-09-30

import torch


class FLClient:
    def __init__(self, id):
        self.id = id


class FLBaseDataset:
    def __init__(self, x, y, client=None,dev=None):
        """
        :param x: training set for a client
        :param y: test set for a client
        :param client: FLClient object
        """
        self.x = x.to(dev)
        self.y = y.to(dev)
        self.length = len(y)
        self.location = client

    def __len__(self):
        return len(self.x)

    def bind(self, client):
        """
        Bind this Base dataset to a local client
        :param client: client as a FLClient object
        :return: na
        """
        assert isinstance(client, FLClient)
        self.location = client


class FLFedDataset:
    def __init__(self, fbd_list):
        """
        :param fbd_list: a list of FLBaseDataset objects
        """
        self.fbd_list = fbd_list

        # count length
        self.total_datasize = 0
        for gbd in self.fbd_list:
            self.total_datasize += len(gbd)

    def __len__(self):
        return len(self.fbd_list)

    def __getitem__(self, item):
        return self.fbd_list[item]


class SimpleFedDataLoader:
    def __init__(self, fed_dataset, client2idx, batch_size, shuffle=False):
        self.fed_dataset = fed_dataset
        self.baseDataset_ptr = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_ptr = -1  # used for splitting into batches

        # shuffle
        if self.shuffle:
            for ds in self.c_idx2data:
                ds_size = len(ds)
                rand_idcs = torch.randperm(ds_size).tolist()
                ds.data = ds.data[rand_idcs]
                ds.targets = ds.targets[rand_idcs]

    def __iter__(self):
        self.batch_ptr = -1  # used for splitting into batches
        self.baseDataset_idx = 0  # loop by the order of BaseDataset
        self.baseDataset_ptr = self.fed_dataset[self.baseDataset_idx]
        self.client_idx = self.baseDataset_ptr.location
        return self

    def __next__(self):
        self.batch_ptr += 1  # this batch
        # update batch location
        if self.batch_ptr * self.batch_size >= self.baseDataset_ptr.length:  # if no more batch for the current client
            self.batch_ptr = 0  # reset
            self.baseDataset_idx += 1  # next BaseDataset
            if self.baseDataset_idx >= len(self.fed_dataset):  # no more client to iterate through
                self.stop()
            self.baseDataset_ptr = self.fed_dataset[self.baseDataset_idx]

        right_bound = self.baseDataset_ptr.length
        this_batch_x = self.baseDataset_ptr.x[self.batch_ptr * self.batch_size:
                                              min(right_bound, (self.batch_ptr + 1) * self.batch_size)]
        this_batch_y = self.baseDataset_ptr.y[self.batch_ptr * self.batch_size:
                                              min(right_bound, (self.batch_ptr + 1) * self.batch_size)]
        location = self.baseDataset_ptr.location

        return this_batch_x, this_batch_y, location

    def stop(self):
        raise StopIteration


@DeprecationWarning  # use SimpleFedDataLoader instead
class FLDataloader:
    def __init__(self, fed_set, shuffle=False, batch_size=1):
        self.fed_set = fed_set  # a FLFedDataset
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        self.set_id = 0
        self.batch_id = 0
        self.next_set = self.fed_set[self.set_id]  # one set per client, the 1st client's data frag here
        self.next_batch_x = self.next_set.x[self.batch_id*self.batch_size:
                                            min((self.batch_id+1)*self.batch_size, len(self.next_set))]
        self.next_batch_y = self.next_set.y[self.batch_id * self.batch_size:
                                            min((self.batch_id + 1) * self.batch_size, len(self.next_set))]
        self.next_batch = (self.next_batch_x, self.next_batch_y)
        return self

    def __next__(self):  #TODO
        self.batch_id += 1
        if self.set_id > len(self.fed_set):
            raise StopIteration
        else:
            batch = self.next_batch


# EventHandler for handling events and altering system states
class EventHandler:
    def __init__(self, state_names):
        """
        Initialize the states
        :param state_names: a name list of states
        """
        assert state_names is not None
        # System states stored as key-value pairs
        self.states = {sn: 0.0 for sn in state_names}

    def get_state(self, state_name):
        return self.states[state_name]

    def add_sequential(self, state_name, value):
        """
        Add a sequential event to the system and handle it
        by changing a specific state (only additive logic in our case)
        :param state_name:
        :param value:
        :return:
        """
        self.states[state_name] += value

    def add_parallel(self, state_name, values, reduce='max'):
        """
        Add parallel events to the system and handle it
        using a specific reduce method of 'none', 'max' or 'sum'
        :param state_name:
        :param values:
        :param reduce:
        :return:
        """
        if reduce == 'none':
            self.states[state_name] += values
        elif reduce == 'max':
            self.states[state_name] += max(values)
        elif reduce == 'sum':
            self.states[state_name] += sum(values)
        else:
            print('[Error] Wrong reduce method specified.')

    def add_obeject(self, state_name, values1, value2, T_limit):
        """
        Add obeject events to the system and handle it
        :param state_name:
        :param values1:
        :param value2:
        :param T_limit
        :return:
        """
        self.states[state_name] += T_limit/(max(values1) + value2)
# #test area
# c1 = FLClient('client_0')
# c2 = FLClient('client_1')
# fbd1 = FLBaseDataset(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), c1)
# fbd2 = FLBaseDataset(torch.tensor([5, 6, 7]), torch.tensor([5, 6, 2]), c2)
# ffd = FLFedDataset([fbd1, fbd2])
#
# for bid, fbd in enumerate(ffd):
#     print(bid, fbd)
#     print(fbd.x, fbd.y, fbd.location)
