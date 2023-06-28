# learning_tasks.py
# Implementations of machine learning tasks
# @Author  : wwt
# @Date    : 2019-7-30

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression

''' Define machine learning models '''
class MLmodelReg(nn.Module):
    """
    Linear regression model implemented as a single-layer NN
    """
    def __init__(self, in_features, out_features):
        super(MLmodelReg, self).__init__()
        self.linear = nn.Linear(in_features, out_features)  # in = independent vars, out = dependent vars

        # init weights
        self.init_weights()

    def init_weights(self, seed=None):
        # common random seed is vital for the effectiveness of averaging, see fig. 1 in
        # McMahan, B.'s paper
        if seed:
            torch.manual_seed(seed)
        else:  # all zero init, deprecated (likely to drop in a local-minimal trap)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)



    def forward(self, x):
        y_pred = self.linear(x)  # y_pred = w * x - b
        return y_pred


class MLmodelSVM(nn.Module):
    """
    Linear Support Vector Machine implemented as a single-layer NN plus regularization
    hyperplane: wx - b = 0
    + samples: wx - b > 1
    - samples: wx - b < -1
    Loss function =  ||w||/2 + C*sum{ max[0, 1 - y(wx - b)]^2 }, C is a hyper-param
    Guide: http://bytepawn.com/svm-with-pytorch.html
    """
    def __init__(self, in_features):
        super(MLmodelSVM, self).__init__()
        # self.linear = torch.nn.Linear(in_features, out_features)  # in = independent vars, out = dependent vars
        #
        # # init weights
        # torch.nn.init.zeros_(self.linear.weight)
        # torch.nn.init.zeros_(self.linear.bias)
        self.w = nn.Parameter(torch.zeros(in_features), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

    def get_w(self):
        return self.w

    def get_b(self):
        return self.b

    def forward(self, x):
        y_hat = torch.mv(x, self.w) - self.b  # y_pred = w * x - b
        return y_hat.reshape(-1, 1)


class svmLoss(nn.Module):
    """
    Loss function class for linear SVM
        reduction: reduction method
    """
    def __init__(self, reduction='mean'):
        super(svmLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_hat, y):
        """
        Loss calculation
        ||w||/2 + C*sum{ max[0, 1 - y*y_hat]^2 }, the regularization term is implemented in optim.SGD with weight_decay
        where y_hat = wx - b
        :param y_hat: output y as a tensor
        :param y: labels as a tensor
        :return: loss
        """
        tmp = 1 - (y * y_hat)
        # sample-wise max, (x + |x|)/2 = max(x, 0)
        abs_tmp = torch.sqrt(tmp**2).detach()
        tmp += abs_tmp  # differentiable abs
        tmp /= 2
        tmp = tmp ** 2
        if self.reduction == 'sum':
            return torch.sum(tmp)
        elif self.reduction == 'mean':
            return torch.mean(tmp)
        else:
            print('E> Wrong reduction method specified')



class MLmodelCNN(nn.Module):
    def __init__(self, classes=10):
        super(MLmodelCNN, self).__init__()
        # input = 28*28 PIL.ToTensor()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)  # 20 [(1*)5*5] conv. kernels
        self.conv2 = nn.Conv2d(20, 50, 5, 1)  # 50 [(20*)5*5] conv. kernels
        self.fc1 = nn.Linear(4 * 4 * 50, 500)  # linear 1
        self.fc2 = nn.Linear(500, classes)  # linear 2

        self.init_weights(seed=1)

    def init_weights(self, seed=None):
        # common random seed is vital for the effectiveness of averaging, see fig. 1 in
        # McMahan, B.'s paper
        if seed:
            torch.manual_seed(seed)
            nn.init.xavier_uniform_(self.conv1.weight)  # current best practice
            nn.init.xavier_uniform_(self.conv2.weight)
        else:  # deprecated (likely to drop in a local-minimal trap)
            nn.init.zeros_(self.conv1.weight)
            nn.init.zeros_(self.conv2.weight)
            nn.init.zeros_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


@ DeprecationWarning  # use svmLoss instead
def svm_loss(y_hat, y, reduction='sum'):
    """
    Loss function for linear SVM
    ||w||/2 + C*sum{ max[0, 1 - y*y_hat]^2 }, the regularization term is implemented in optim.SGD with weight_decay
    where y_hat = wx - b
    :param y_hat: output as tensor
    :param y: label as tensor
    :param reduction: reduction method
    :return: loss as tensor
    """
    batch_size = y.shape[0]
    tmp = 1 - (y * y_hat)
    # sample-wise max
    zeros_ = torch.zeros((batch_size, 1))
    tmp = torch.cat((zeros_, tmp.reshape(-1,1)), dim=1)  # zeros attached as a column
    sample_losses = torch.max(tmp, dim=1)[0]  # torch.max's return also contains indices
    sample_losses = sample_losses ** 2

    if reduction == 'sum':
        return torch.sum(sample_losses)
    elif reduction == 'mean':
        return torch.mean(sample_losses)
    else:
        print('E> Wrong reduction method specified')


# test area
# m = MLmodelSVM(3)
# x = torch.tensor([1.2, 1.0, 1.1])
# y = torch.tensor([1.])
# # w = torch.autograd.Variable(torch.tensor([1.0, 0.5, -0.5]), requires_grad=True)
# # b = torch.autograd.Variable(torch.tensor([0.0]), requires_grad=True)
# y_hat = m(x)
# print('y_hat=', y_hat)
# # loss = svm_loss(y_hat, y=torch.tensor([1], dtype=torch.float), reduction='mean')
# loss_func = svmLoss(reduction='mean')
# loss = loss_func(y_hat, y)
# print('loss=',loss)
# loss.backward()  # tensors flowing
# for param in m.parameters():
#     print(param, param.grad)
