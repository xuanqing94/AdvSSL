#!/usr/bin/env python
import sys
import numpy as np
import scipy as sp
from utils.data_loader import *
from utils.metrics import *
from utils.tr_solver import trust_region_solver 
from utils.discrete_optim import greedy_method
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
    def similarity_matrix(cls, X, gamma):
        tmp = X @ X.T
        n_data = X.shape[0]
        diag = np.diag(tmp)
        S = gamma * (2 * tmp - diag.reshape(1, n_data) - diag.reshape(n_data, 1))
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
                if self.task == 'regression':
                    y_tr = self.y_tr + perturb
                elif self.task == 'classification':
                    y_tr = self.y_tr * perturb
                else:
                    raise ValueError(f'Invalid self.task: {self.task}')
            S = LP.similarity_matrix(X, self.gamma)
            D = np.diag(np.sum(S, axis=1, keepdims=False))
            Suu = S[n_tr:, n_tr:]
            Duu = D[n_tr:, n_tr:]
            Sul = S[n_tr:, :n_tr]
            tmp = np.linalg.inv(Duu - Suu) @ Sul
            if self.task == 'regression':
                y_pred = np.dot(tmp, y_tr)
            elif self.task == 'classification':
                y_pred = np.sign(np.dot(tmp, y_tr))
            else:
                raise ValueError(f'Invalid self.task: {self.task}')
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
    
    def perturb_y_classification(self, c_max):
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
        K = np.linalg.inv(Duu - Suu) @ Sul
        distortion = greedy_method(K, y_tr, y_te, c_max)
        return distortion

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
    
def evaluate(lp, gamma_adv, gamma_test, nl, max_perturb, flip_eps=False):
    """General function"""
    lp.set_train_num(nl)
    # find adversarial examples under gamma_adv
    lp.set_hparam(gamma_adv)
    if lp.task == "classification":
        delta = lp.perturb_y_classification(max_perturb)
    elif lp.task == "regression":
        delta = lp.perturb_y_regression(max_perturb)
    # reset gamma to gamma_test and evaluate
    lp.set_hparam(gamma_test)
    mean_te, _ = lp.training(n_trial=1)
    mean_te_perturb, _ = lp.training(n_trial=1, perturb=delta)
    if flip_eps:
        mean_te_perturb2, _ = lp.training(n_trial=1, perturb=-delta)
        return mean_te, mean_te_perturb, mean_te_perturb2
    else:
        return mean_te, mean_te_perturb


def gamma_sensitivity_mnist_sup(lp):
    """How sensitive is gamma to the success rate of attack? 
    Fix gamma at test time, change gamma_adv to different values
    """
    nl = 100
    c_max = 3 # c_max should grow with n_l ?
    gamma_test = 0.6
    gamma_adv = [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.6, 0.61, 0.63, 0.65, 0.67, 0.7]
    print('Gamma_adv\tAcc\tAcc-eps')
    for g_adv in gamma_adv:
        mean_te, mean_te_perturb = evaluate(lp, g_adv, gamma_test, nl, c_max, flip_eps=False)
        print(f'{g_adv}\t{mean_te}\t{mean_te_perturb}')
        sys.stdout.flush()

def perturb_sensitivity_mnist_sup(lp):
    """Change level of perturbation and evaluate the success rate"""
    nl = 100
    gamma_test = gamma_adv = 0.6
    c_max = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print('C_max\tAcc\tAcc-eps')
    for cm in c_max:
        mean_te, mean_te_perturb = evaluate(lp, gamma_adv, gamma_test, nl, cm, flip_eps=False)
        print(f'{cm}\t{mean_te}\t{mean_te_perturb}')
        sys.stdout.flush()

def nl_sensitivity_mnist_sup(lp):
    """Change number of training data and evaluate the success rate"""
    gamma_test = gamma_adv = 0.6
    c_max = 5
    nl_choices = [50, 100, 150, 200, 250]
    print('Nl\tAcc\tAcc-eps')
    for nl in nl_choices:
        mean_te, mean_te_perturb = evaluate(lp, gamma_adv, gamma_test, nl, c_max, flip_eps=False)
        print(f'{nl}\t{mean_te}\t{mean_te_perturb}')
        sys.stdout.flush()

if __name__ == "__main__":
    data ="mnist"
    # for cpusmall data, gamma=20 (data unnormalized!)
    # for cadata, gamma=1
    # for mnist gamma=0.6
    lp = LabelProp(data)
    lp.shuffle_data()
    gamma_sensitivity_mnist_sup(lp)
    # adv_list selection
    #cadata_advlist = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.3, 2.5, 2.7, 2.9, 3.5]
    #mnist_advlist = [0.3, 0.6, 0.7]
    #for gamma_adv in mnist_advlist:
    #    print(f"===================> Gamma_adv={gamma_adv} <=========================")
    #    gamma_sensitivity_mnist(gamma_adv)
