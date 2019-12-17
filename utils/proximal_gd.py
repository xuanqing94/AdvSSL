import numpy as np


def func_val(K, g, d_y, lam1, lam2):
    tmp = K @ d_y + g
    return -0.5 * np.sum(tmp * tmp) + lam1 * np.sum(np.abs(tmp)) + lam2 / 2 * np.sum(tmp * tmp)

def prox_gd(K, g, lam1, lam2):
    lr = 1.0e-4
    n_l, n_l = K.shape
    d_y = np.random.randn(n_l) * 0.01
    threshold = lr * lam1
    M = K.T @ K
    for i in range(100):
        #DEBUG
        print('func val: ', func_val(K, g, d_y, lam1, lam2))
        tmp = (1 - lr * lam2) * d_y + lr * M @ d_y + lr * K.T @ g
        tmp[np.abs(tmp) < threshold] = 0.0
        mask = tmp > threshold
        tmp[mask] = tmp[mask] - threshold
        mask = tmp < -threshold
        tmp[mask] = tmp[mask] + threshold
        d_y = tmp
    return d_y
