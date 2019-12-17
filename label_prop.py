#!/usr/bin/env python
import os
import sys
# When attacking the labels
import numpy as np
import scipy as sp
# for randomized svd
from sklearn.utils.extmath import randomized_svd

from utils.data_loader import *
from utils.metrics import *
from utils.tr_solver import trust_region_solver
from utils.spca_solver import sparse_pca
from utils.discrete_optim import greedy_method, threshold_method, \
        threshold_method_soft, probablistic_method, probablistic_method_soft, \
        exhaustive_search, greedy_with_init
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
        elif data == 'a9a':
            self.features, self.labels = load_a9a()
            self.metric = accuracy
            self.task = 'classification'
        elif data == 'covtype':
            self.features, self.labels = load_covtype()
            self.metric = accuracy
            self.task = 'classification'
        elif data == 'rcv1':
            self.features, self.labels = load_rcv1()
            self.metric = accuracy
            self.task = 'classification'
        elif data == 'e2006':
            self.features, self.labels = load_e2006()
            self.metric = RMSE
            self.task = 'regression'

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
        self.features = self.features[shuffle_idx]
        self.labels = self.labels[shuffle_idx]

    def split_data(self):
        n_train = self.train_num
        # split data
        train_features = self.features[:n_train]
        test_features = self.features[n_train:]
        train_labels = self.labels[:n_train]
        test_labels = self.labels[n_train:]
        self.X_tr, self.X_te, self.y_tr, self.y_te = \
                train_features, test_features, train_labels, test_labels

    def similarity_matrix(self, X, gamma):
        cache_file = f'./data/{self.data_name}.npy'
        if os.path.exists(cache_file):
            tmp = np.load(cache_file)
        else:
            if sp.sparse.issparse(X):
                X = X.tocoo()
                tmp = X @ X.T
                tmp = np.asarray(tmp.todense())
            else:
                tmp = X @ X.T
            np.save(cache_file, tmp)
        n_data = X.shape[0]
        diag = np.diag(tmp)
        S = gamma * (2 * tmp - diag.reshape(1, n_data) - diag.reshape(n_data, 1))
        return np.exp(S)

    def diagnoal(cls, similarity_matrix):
        D = np.diag(np.sum(similarity_matrix, axis=1))
        return D

    def l2_loss(self, delta_X):
        n_tr = self.train_num
        # perturb Xu, Xl with delta_X
        X_ = self.features + delta_X
        S = self.similarity_matrix(X_, self.gamma)
        D = self.diagnoal(S)
        Suu = S[n_tr:, n_tr:]
        Duu = D[n_tr:, n_tr:]
        Sul = S[n_tr:, :n_tr]
        tmp = np.linalg.inv(Duu - Suu) @ Sul
        y_tr = self.y_tr
        y_pred_ = np.dot(tmp, y_tr)
        diff = y_pred_ - self.y_te
        return -0.5 * np.sum(diff * diff)

    def training(self, n_trial=1, perturb=None):
        mse_te = []
        for k_trial in range(n_trial):
            y_pred = self.prediction(perturb)
            # evaluation
            mse = self.metric(y_pred, self.y_te)
            mse_te.append(mse)
        return np.mean(mse_te), np.std(mse_te)

    def prediction(self, perturb=None):
        # split data for a new experiment
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
        S = self.similarity_matrix(X, self.gamma)
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
        return y_pred

    def perturb_y_type1(self, d_max):
        """When attacker does not known ground truth label
        d_max: maximum L-2 perturbation
        """
        self.split_data()
        n_tr, _ = self.X_tr.shape
        X = self.features
        S = self.similarity_matrix(X, self.gamma)
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
        self.split_data()
        X = self.features
        y_tr = self.y_tr
        n_tr = len(y_tr)
        y_te = self.y_te # knowns the ground truth label
        S = self.similarity_matrix(X, self.gamma)
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
        self.split_data()
        n_tr, _ = self.X_tr.shape

    def perturb_y_regression(self, d_max, supervised=True):
        self.split_data()
        X = self.features
        y_tr = self.y_tr
        n_tr = len(y_tr)
        y_te = self.y_te
        S = self.similarity_matrix(X, self.gamma)
        D = np.diag(np.sum(S, axis=1, keepdims=False))
        Suu = S[n_tr:, n_tr:]
        Duu = D[n_tr:, n_tr:]
        Sul = S[n_tr:, :n_tr]
        K = np.linalg.inv(Duu - Suu) @ Sul
        if supervised:
            # attacker knows y_te
            M = K.T @ K
            e = K @ y_tr - y_te
            g = K.T @ e
            delta = trust_region_solver(M, g, d_max)
        else:
            # attacker does not know y_te
            M = K.T @ K
            eig_val, eig_vec = np.linalg.eig(M)
            # pick the largest eigen vector
            idx = np.argmax(eig_val)
            delta = d_max * eig_vec[:, idx] # eigen vectors are column-wise
        return delta

    def perturb_y_regression_random(self, d_max, supervised=True):
        self.split_data()
        n_tr = len(self.y_tr)
        delta = np.random.randn(n_tr)
        return d_max * delta / np.linalg.norm(delta)

    def perturb_y_classification_random(self, c_max, supervised=True):
        self.split_data()
        n_tr = len(self.y_tr)
        delta = np.ones(n_tr)
        # randomly select c_max
        idx = np.random.choice(n_tr, c_max, replace=False)
        delta[idx] = -1
        return delta

    def perturb_y_classification(self, c_max, supervised=True):
        self.split_data()
        X = self.features
        y_tr = self.y_tr
        n_tr = len(y_tr)
        if supervised:
            # knowns the ground truth label
            y_te = self.y_te
        else:
            # does not know the ground truth label
            y_te = self.prediction()
        S = self.similarity_matrix(X, self.gamma)
        D = np.diag(np.sum(S, axis=1, keepdims=False))
        Suu = S[n_tr:, n_tr:]
        Duu = D[n_tr:, n_tr:]
        Sul = S[n_tr:, :n_tr]
        K = np.linalg.inv(Duu - Suu) @ Sul
        # greedy / threshold / probablistic / exhaustive_search
        distortion = probablistic_method_soft(K, y_tr, y_te, c_max)
        return distortion

    def perturb_y_regression_sparse(self, lam1, lam2, d_max, supervised=True):
        assert supervised == False, 'Cannot deal with supervised case'
        self.split_data()
        X = self.features
        y_tr = self.y_tr
        n_tr = len(y_tr)
        S = self.similarity_matrix(X, self.gamma)
        D = np.diag(np.sum(S, axis=1, keepdims=False))
        Suu = S[n_tr:, n_tr:]
        Duu = D[n_tr:, n_tr:]
        Sul = S[n_tr:, :n_tr]
        K = np.linalg.inv(Duu - Suu) @ Sul
        distortion = sparse_pca(K, lam1, lam2)
        return distortion * d_max

    def perturb_x_regression(self, d_max):
        self.split_data()
        delta_X = np.zeros_like(self.features)
        grad_delta_fn = grad(self.l2_loss)
        grad_delta = grad_delta_fn(delta_X)
        print(grad_delta)

    def hub_score(self):
        # caluclate the graph adj matrix
        X = self.features
        S = self.similarity_matrix(X, self.gamma)
        #D = np.diag(np.sum(S, axis=1, keepdims=False))
        U, Sig, Vt = randomized_svd(S, n_components=1)
        score = U[:self.train_num, :]
        return score

def find_good_gamma():
    data = "cadata"
    lp = LabelProp(data)
    lp.shuffle_data()
    lp.set_train_num(500)
    lp.split_data()
    y_te = lp.y_te
    #print('y_te norm: ', np.linalg.norm(y_te))
    #exit(0)
    for gamma in [0.1, 0.5, 1, 1.5, 2]:
        lp.set_hparam(gamma)
        mean_te, _ = lp.training(n_trial=1)
        print(f'{gamma}, {mean_te}')

def evaluate(lp, gamma_adv, gamma_test, nl, max_perturb, flip_eps=False, supervised=True):
    """General function"""
    lp.set_train_num(nl)
    # find adversarial examples under gamma_adv
    lp.set_hparam(gamma_adv)
    if lp.task == "classification":
        delta = lp.perturb_y_classification_random(max_perturb, supervised=supervised)
    elif lp.task == "regression":
        #delta = lp.perturb_y_regression(max_perturb, supervised=supervised)
        #delta = lp.perturb_y_regression_sparse(max_perturb[0], max_perturb[1], max_perturb[2], supervised=False)
        delta = lp.perturb_y_regression_random(max_perturb, supervised=supervised)
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
    nl = 600
    c_max = 30 # c_max should grow with n_l ?
    gamma_test = 0.6
    gamma_adv = [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.6, 0.61,0.63, 0.64, 0.65, 0.66, 0.67, 0.7, 0.71, 0.72, 0.73]
    #gamma_adv = [0.6]
    print('#Gamma_adv\tAcc\tAcc-eps')
    for g_adv in gamma_adv:
        mean_te, mean_te_perturb = evaluate(lp, g_adv, gamma_test, nl, c_max, flip_eps=False)
        print(f'{g_adv}\t{mean_te}\t{mean_te_perturb}')
        sys.stdout.flush()

def perturb_sensitivity_mnist(lp, supervised=True):
    """Change level of perturbation and evaluate the success rate"""
    nl = 100
    gamma_test = gamma_adv = 0.6
    c_max = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print('#C_max\tAcc\tAcc-eps')
    for cm in c_max:
        mean_te, mean_te_perturb = evaluate(lp, gamma_adv, gamma_test, nl, cm, flip_eps=False, supervised=supervised)
        print(f'{cm}\t{mean_te}\t{mean_te_perturb}')
        sys.stdout.flush()

def perturb_sensitivity_rcv1(lp, supervised=True):
    """Change level of perturbation and evaluate the success rate"""
    nl = 1000
    gamma_test = gamma_adv = 0.1
    c_max = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 ,24, 25, 26, 27, 28, 29, 30]
    #c_max = [20]
    print('#C_max\tAcc\tAcc-eps')
    for cm in c_max:
        mean_te, mean_te_perturb = evaluate(lp, gamma_adv, gamma_test, nl, cm, flip_eps=False, supervised=supervised)
        print(f'{cm}\t{mean_te}\t{mean_te_perturb}')
        sys.stdout.flush()

def perturb_sensitivity_cadata(lp, supervised=True):
    """Change level of perturbation and evaluate the success rate"""
    print('#D_max\tRMSE\tRMSE-eps')
    gamma_test = gamma_adv = 1.0
    nl = 1000
    unit = np.sqrt(nl)
    #d_max = [0, 0.01, 0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    d_max = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18]
    for dm in d_max:
        dm = dm * unit
        if supervised:
            mean_te, mean_te_perturb = evaluate(lp, gamma_adv, gamma_test, nl, dm, flip_eps=False, supervised=True)
        else:
            mean_te, mean_te_perturb, mean_te_perturb2 = evaluate(lp, gamma_adv, gamma_test, nl, dm, flip_eps=True, supervised=False)
            mean_te_perturb = max(mean_te_perturb, mean_te_perturb2)
        print(f'{dm}\t{mean_te}\t{mean_te_perturb}')
        sys.stdout.flush()

def perturb_sensitivity_e2006(lp, supervised=True):
    """Change level of perturbation and evaluate the success rate"""
    print('#D_max\tRMSE\tRMSE-eps')
    gamma_test = gamma_adv = 1.0
    nl = 300
    unit = np.sqrt(nl)
    #d_max = [0, 0.01, 0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    d_max = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18]
    for dm in d_max:
        dm = dm * unit
        if supervised:
            mean_te, mean_te_perturb = evaluate(lp, gamma_adv, gamma_test, nl, dm, flip_eps=False, supervised=True)
        else:
            mean_te, mean_te_perturb, mean_te_perturb2 = evaluate(lp, gamma_adv, gamma_test, nl, dm, flip_eps=True, supervised=False)
            mean_te_perturb = max(mean_te_perturb, mean_te_perturb2)
        print(f'{dm}\t{mean_te}\t{mean_te_perturb}')
        sys.stdout.flush()

def gamma_sensitivity_cadata(gamma_adv):
    print('#N_train\tRMSE\tRMSE-eps+\tRMSE-eps-')
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

def X_sensitivity_cadata(lp):
    lp.set_train_num(1000)
    d_max = 0.10 * np.sqrt(1000)
    lp.set_hparam(1.0)
    lp.perturb_x_regression(d_max)
    exit(0)

def nl_sensitivity_mnist_sup(lp):
    """Change number of training data and evaluate the success rate"""
    gamma_test = gamma_adv = 0.6
    c_max = 5
    nl_choices = [50, 100, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]
    print('Nl\tAcc\tAcc-eps')
    for nl in nl_choices:
        mean_te, mean_te_perturb = evaluate(lp, gamma_adv, gamma_test, nl, c_max, flip_eps=False)
        print(f'{nl}\t{mean_te}\t{mean_te_perturb}')
        sys.stdout.flush()

def test_elestic_net(lp):
    gamma_test = gamma_adv = 1.0
    lam2 = 0.0
    nl = 300
    d_max = 0.18 * np.sqrt(nl)
    print('#lam1\tRMSE\tRMSE-eps')
    for lam1 in [0.37]:
        mean_te, mean_te_perturb, mean_te_perturb2 = evaluate(lp, gamma_adv, gamma_test, nl, (lam1, lam2, d_max), flip_eps=True, supervised=False)
        mean_te_perturb = max(mean_te_perturb, mean_te_perturb2)
        print(f"{lam1}\t{mean_te}\t{mean_te_perturb}")

def test_hub_score():
    data = "mnist"
    gamma = 0.6
    n_tr = 500
    lp = LabelProp(data)
    lp.set_hparam(gamma)
    lp.set_train_num(n_tr)
    lp.shuffle_data()
    lp.split_data()
    score = lp.hub_score()
    #d_max = 0.1 * np.sqrt(n_tr)
    c_max = 500
    d_y = lp.perturb_y_classification(c_max)
    #d_y = lp.perturb_y_regression(d_max)
    #d_y = lp.perturb_y_regression_sparse(0.1, 0, d_max, supervised=False)
    #d_y = np.abs(d_y)
    np.savez(f'./exp-data/hub_score/{data}_{n_tr}', score=score, d_y_abs=d_y)


if __name__ == "__main__":
    #test_hub_score()
    #exit(0)
    data = "rcv1"
    # for cpusmall data, gamma=20 (data unnormalized!)
    # for cadata, gamma=1
    # for mnist gamma=0.6
    # for rcv1 gamma=0.1
    # for e2006 gamma=1
    lp = LabelProp(data)
    lp.shuffle_data()
    #X_sensitivity_cadata(lp)
    #gamma_sensitivity_mnist_sup(lp)
    perturb_sensitivity_rcv1(lp)
    #perturb_sensitivity_rcv1(lp, supervised=True)

    #test_elestic_net(lp)
    #nl_sensitivity_mnist_sup(lp)
    # adv_list selection
    #cadata_advlist = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.3, 2.5, 2.7, 2.9, 3.5]
    #mnist_advlist = [0.3, 0.6, 0.7]
    #for gamma_adv in mnist_advlist:
    #    print(f"===================> Gamma_adv={gamma_adv} <=========================")
    #    gamma_sensitivity_mnist(gamma_adv)
