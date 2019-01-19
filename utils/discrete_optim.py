import numpy as np


# Methods to solve combinatoric problem:
# min_{d_y} -0.5 * |K(y_l*d_y) - y_u|**2
#      s.t. d_y has less than c '-1' elements

def func_val(K, y_l, d_y, y_u):
    # calculate the objective function value
    tmp = K @ (y_l * d_y) - y_u
    return -0.5 * np.sum(tmp * tmp)

def greedy_method(K, y_l, y_u, c):
    """
    make sure c < len(y_l)
    """
    nu, nl = K.shape
    # see paper for algorithm details
    d_y = np.ones(nl, dtype=int)
    for i in range(c):
        # calculate the original function value
        func_original = func_val(K, y_l, d_y, y_u)
        progress = 0
        best_idx = -1
        # find the i-th best element to flip
        for j, is_flipped in enumerate(d_y):
            if is_flipped == -1:
                # if this element is flipped, skip it
                continue
            # try flip this element and check the function decrement
            d_y[j] = -1
            func_try = func_val(K, y_l, d_y, y_u)
            if func_try - func_original < progress:
                # made a better progress
                progress = func_try - func_original
                best_idx = j
            # reset this element
            d_y[j] = 1
        # greedy
        if best_idx >= 0:
            d_y[best_idx] = -1
        else:
            # not improvable
            break
    return d_y


def threshold_method(K, y_l, y_u, c):
    """Solve a trust region problem
    M = K^T * K
    g = K^T * y_u
    d_max = sqrt(nl)
    """
    M = K.T @ K
    g = k.T @ y_u
    d_max = np.sqrt(len(y_l))
    solution = trust_region_solver(M, g, d_max, max_iter=200, stepsize=1.0e-3)
    # select c smallest elements
    idx = np.argsort(solution)
    d_y = np.ones_like(y_l)
    for i in idx:
        if solution[i] < 1:
            d_y[i] = -1
    return d_y


def probablistic_method():
    pass
