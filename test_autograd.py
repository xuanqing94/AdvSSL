#!/usr/bin/env python
import sys
import math
import torch
import numpy as np

from utils.data_loader import *

from utils.data_loader import *
from utils.metrics import *
# fix random seed
np.random.seed(0)

class LabelProp(object):
    def __init__(self, data):
        self.data_name = data
        if data == 'cadata':
            self.features, self.labels = load_cadata()
            self.metric = RMSE
            self.task = 'regression'
        elif data == 'mnist':
            self.features, self.labels = load_mnist()
            self.metric = accuracy
            self.task = 'classification'
    
    def set_train_num(self, train_num):
        self.train_num = train_num
        return self

    def set_hparam(self, gamma):
        self.gamma = gamma
        return self

    def shuffle_data(self):
        # shuffle before split data
        n_data = self.features.shape[0]
        shuffle_idx = np.random.permutation(n_data)
        # do two inplace shuffle operations
        self.features = np.take(self.features, shuffle_idx, axis=0)
        self.labels = np.take(self.labels, shuffle_idx, axis=0)
        # convert to pytorch array
        self.features = torch.from_numpy(self.features).float().cuda()
        self.labels = torch.from_numpy(self.labels).float().cuda()

    def split_data(self):
        n_train = self.train_num
        # split data
        train_features = self.features[:n_train]
        test_features = self.features[n_train:]
        train_labels = self.labels[:n_train]
        test_labels = self.labels[n_train:]
        self.X_tr, self.X_te, self.y_tr, self.y_te = \
                train_features, test_features, train_labels, test_labels
    
    @classmethod
    def similarity_matrix(cls, X, gamma):
        tmp = X @ torch.transpose(X, 0, 1)
        n_data = X.size(0)
        diag = torch.diag(tmp)
        S = gamma * (2 * tmp - diag.view(1, n_data) - diag.view(n_data, 1))
        return torch.exp(S)
    
    @classmethod
    def diagnoal(cls, similarity_matrix):
        D = torch.diag(torch.sum(similarity_matrix, dim=1, keepdim=False))
        return D

    def l2_loss(self, delta_x):
        LP = LabelProp
        n_tr = self.train_num
        # perturb X
        X_ = self.features + delta_x
        S = LP.similarity_matrix(X_, self.gamma)
        D = LP.diagnoal(S)
        Suu = S[n_tr:, n_tr:]
        Duu = D[n_tr:, n_tr:]
        Sul = S[n_tr:, :n_tr]
        y_tr = self.y_tr
        y_te = self.y_te
        tmp = torch.mm(torch.inverse(Duu - Suu), Sul)
        y_pred = torch.mv(tmp, y_tr)
        diff = y_pred - y_te
        return -0.5 * torch.sum(diff * diff)
    
    def perturb_x_regression(self, d_max):
        """Find the optimal L2 constraint perturbation by
        projected gradient descent.
        """
        self.split_data()
        delta_x = torch.zeros_like(self.features, requires_grad=True).cuda()
        lr = 1.0e1
        while True:
            # calculate the L2 loss
            loss = self.l2_loss(delta_x)
            print(f'Loss: {loss.item()}')
            # backward to calculate the gradient
            g = torch.autograd.grad(loss, [delta_x])[0]
            # gradient descent
            delta_x.detach().add_(-lr * g)
            # project to l2-ball with radius d_max
            norm_delta_x = torch.norm(delta_x)
            if norm_delta_x > d_max:
                # project back
                delta_x.detach().mul_(d_max / norm_delta_x)
        return delta_x

def X_sensitivity_cadata(lp):
    lp.set_train_num(1000)
    X = lp.features
    mean_norm = torch.norm(X)
    d_max = 0.1 * mean_norm
    print(d_max)
    lp.set_hparam(.2)
    lp.perturb_x_regression(d_max)


if __name__ == "__main__":
    data ="cadata"
    # for cpusmall data, gamma=20 (data unnormalized!)
    # for cadata, gamma=1
    # for mnist gamma=0.6
    lp = LabelProp(data)
    lp.shuffle_data()
    X_sensitivity_cadata(lp)
