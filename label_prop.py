#!/usr/bin/env python
import sys
import numpy as np
import scipy as sp
from utils.data_loader import *
from utils.metrics import *
from utils.tr_solver import trust_region_solver 
# fix random seed
np.random.seed(0)


class LabelProp(object):
    def __init__(self, data):
        self.data_name = data
        if data == 'cadata':
            self.features, self.labels = load_cadata()
            self.metric = RMSE
        elif data == 'mnist':
            self.features, self.labels = load_mnist()
            self.metric = accuracy

    def load_svm_data(self, path, n_features=None, to_dense=False):
        features, labels = load_svmlight_file(path, n_features)
        if to_dense:
            features = np.asarray(features.todense())
            n_data = features.shape[0]
            # add a bias term
            features = np.hstack((features, np.ones(n_data).reshape(-1, 1)))
        return features, labels 
    
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
    def similarity_matrix(cls, X_tr, gamma):
        tmp = X_tr @ X_tr.T
        n_tr = X_tr.shape[0]
        diag = np.diag(tmp)
        S = gamma * (2 * tmp - diag.reshape(1, n_tr) -
                   diag.reshape(n_tr, 1))
        return np.exp(S)

    @classmethod
    def laplacian(cls, similarity_matrix):
        D = np.diag(np.sum(similarity_matrix, axis=1))
        return D - similarity_matrix
    
    def training(self, n_trial=1, perturb=None):
        LP = LabelProp # abbrv.
        mse_te = []
        for k_trial in range(n_trial):
            # shuffle & split data for a new experiment
            #self.shuffle_data()
            self.split_data()
            # label propagation
            n_tr, _ = self.X_tr.shape
            X = self.features
            y_tr = self.y_tr
            if perturb is not None:
                y_tr = self.y_tr + perturb
            S = LP.similarity_matrix(X, self.gamma)
            D = np.diag(np.sum(S, axis=1, keepdims=False))
            Suu = S[n_tr:, n_tr:]
            Duu = D[n_tr:, n_tr:]
            Sul = S[n_tr:, :n_tr]
            tmp = np.linalg.inv(Duu - Suu) @ Sul
            y_pred = np.sign(np.dot(tmp, y_tr))
            # evaluation
            mse = self.metric(y_pred, self.y_te)
            mse_te.append(mse)
        return np.mean(mse_te), np.std(mse_te)
    
    def perturb_y_type1(self, d_max):
        """When attacker does not known ground truth label
        d_max: maximum L-2 perturbation
        """
        LP = LabelProp
        self.split_data()
        n_tr, _ = self.X_tr.shape
        X = self.features
        S = LP.similarity_matrix(X, self.gamma)
        D = np.diag(np.sum(S, axis=1, keepdims=False))
        Suu = S[n_tr:, n_tr:]
        Duu = D[n_tr:, n_tr:]
        Sul = S[n_tr:, :n_tr]
        inv = np.linalg.inv(Duu - Suu)
        tmp = inv @ Sul
        M = tmp.T @ tmp
        eig_val, eig_vec = np.linalg.eig(M)
        # pick the largest eigen vector
        idx = np.argmax(eig_val)
        delta = d_max * eig_vec[:, idx] # eigen vectors are column-wise
        # return optimal perturbation
        return delta
    
    def perturb_y_type2(self, d_max):
        """When attacker knows the groud truth label
        The algorithm is to solve a trust region problem
        """
        LP = LabelProp
        self.split_data()
        X = self.features
        y_tr = self.y_tr
        n_tr = len(y_tr)
        y_te = self.y_te # knowns the ground truth label
        S = LP.similarity_matrix(X, self.gamma)
        D = np.diag(np.sum(S, axis=1, keepdims=False))
        Suu = S[n_tr:, n_tr:]
        Duu = D[n_tr:, n_tr:]
        Sul = S[n_tr:, :n_tr]
        tmp = np.linalg.inv(Duu - Suu) @ Sul
        M = tmp.T @ tmp
        e = tmp @ y_tr - y_te
        g = tmp.T @ e
        delta = trust_region_solver(M, g, d_max)
        return delta

    def perturb_y_type3(self, d_max):
        """Attacker does not know ground truth label
        d_max: maximum L-inf perturbation
        """
        LP = LabelProp
        self.split_data()
        n_tr, _ = self.X_tr.shape
        

def gamma_sensitivity_cadata(gamma_adv):
    print('N_train\tRMSE\tRMSE-eps+\tRMSE-eps-') 
    for num in [1000]:
        lp.set_train_num(num)
        d_max = .10 * np.sqrt(num) # d_max is l2-norm and it should grow with dimension?
        # find adversarial examples under gamma_adv
        lp.set_hparam(gamma_adv)
        delta = lp.perturb_y_type2(d_max)
        # reset gamma to gamma_test and evaluate the performance
        lp.set_hparam(gamma_test)
        mean_te, _ = lp.training(n_trial=1)
        mean_te_perturb, _ = lp.training(n_trial=1, perturb=delta)
        mean_te_perturb2, _ = lp.training(n_trial=1, perturb=-delta)
        print(f'{num}\t{mean_te}\t{mean_te_perturb}\t{mean_te_perturb2}')
        sys.stdout.flush()


def gamma_sensitivity_mnist(gamma_adv):
    print('N_train\tAcc\tAcc-eps+\tAcc-eps-')
    for num in [100, 300, 500, 1000, 1500, 2000, 3000]:
        lp.set_train_num(num)
        d_max = 0.3 * np.sqrt(num) # d_max is l2-norm and it should grow with dimension?
        # find adversarial examples under gamma_adv
        lp.set_hparam(gamma_adv)
        delta = lp.perturb_y_type1(d_max)
        # reset gamma to gamma_test and evaluate the performance
        lp.set_hparam(gamma_test)
        acc_te, _ = lp.training(n_trial=1)
        acc_te_perturb, _ = lp.training(n_trial=1, perturb=delta)
        acc_te_perturb2, _ = lp.training(n_trial=1, perturb=-delta)
        print(f'{num}\t{acc_te}\t{acc_te_perturb}\t{acc_te_perturb2}')
        sys.stdout.flush()


def find_good_gamma():
    data = "mnist"
    gamma = 1
    lp = LabelProp(data)
    lp.shuffle_data()
    lp.set_train_num(1500)
    for gamma in (0.4, 0.6, 0.8, 1.0, 1.2, 1.4):
        lp.set_hparam(gamma)
        mean_te, _ = lp.training(n_trial=1)
        print(f'{gamma}, {mean_te}')
    

if __name__ == "__main__":
    data ="mnist"
    # for cpusmall data, gamma=20 (data unnormalized!)
    # for cadata, gamma=1
    # for mnist gamma=0.6
    gamma_test = 0.6
    lp = LabelProp(data)
    lp.shuffle_data()
    for gamma_adv in np.arange(0.4, 0.7, 0.01):
        print(f"===================> Gamma_adv={gamma_adv} <=========================")
        gamma_sensitivity_mnist(gamma_adv)
    """
    filepath ="../data/cadata_sample"
    # for cpusmall data, gamma=20 (data unnormalized!)
    # for cadata, gamma=2.5
    gamma_adv = 1
    lp = LabelProp(filepath, to_dense=True).set_hparam(gamma)
    lp.shuffle_data()
    #lp.set_train_num(7000) 
    #mean_te, _ = lp.training(n_trial=1)
    #print(f'gamma: {gamma}\tRMSE: {mean_te}')

    print('N_train, RMSE, RMSE-eps+, RMSE-eps-') 
    for num in [100, 200, 300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]:
        lp.set_train_num(num)
        d_max = .10 * np.sqrt(num) # d_max is l2-norm and it should grow with dimension?
        #d_max = .15
        delta = lp.perturb_y_type1(d_max)
        mean_te, _ = lp.training(n_trial=1)
        mean_te_perturb, _ = lp.training(n_trial=1, perturb=delta)
        mean_te_perturb2, _ = lp.training(n_trial=1, perturb=-delta)
        print(f'{num}, {mean_te}, {mean_te_perturb}, {mean_te_perturb2}')
    """
