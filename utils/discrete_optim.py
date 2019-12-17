from itertools import combinations
from .tr_solver import trust_region_solver
import numpy as np
import torch

torch.manual_seed(0)

# Methods to solve combinatoric problem:
# min_{d_y} -0.5 * |K(y_l*d_y) - y_u|**2
#      s.t. d_y has less than c '-1' elements

def func_val(K, y_l, d_y, y_u):
    # calculate the objective function value
    tmp = np.sign(K @ (y_l * d_y)) - y_u
    return -0.5 * np.sum(tmp * tmp)

def exhaustive_search(K, y_l, y_u, c):
    """Performs a brute-force search of all choices"""
    d_y = np.ones_like(y_l)
    if c == 0: # fix a corner case
        return d_y
    original = func_val(K, y_l, d_y, y_u)
    progress = 0
    # generate combinations
    for selection in combinations(range(len(y_l)), c):
        selection = list(selection)
        flip = np.ones_like(y_l)
        flip[selection] = -1
        val = func_val(K, y_l, flip, y_u)
        if val - original < progress:
            progress = val - original
            d_y = flip
    return d_y

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

cache = None

def greedy_with_init(K, y_l, y_u, c):
    """Greedy method but with 2 good initializations"""
    if c <= 1:
        return greedy_method(K, y_l, y_u, c)
    global cache
    if cache is None:
        d_y = exhaustive_search(K, y_l, y_u, 2)
        cache = d_y
    else:
        # Debug
        print("#Using cached values")
        d_y = cache
    c_remains = c - 2
    nu, nl = K.shape
    # see paper for algorithm details
    for i in range(c_remains):
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
    g = K.T @ y_u
    d_max = np.sqrt(len(y_l))
    #d_max = len(y_l)

    solution = trust_region_solver(M, g, d_max, max_iter=100, stepsize=1.0e-3)
    # select c smallest elements
    idx = np.argsort(solution)
    d_y = np.ones_like(y_l)
    count = 0
    for i in idx:
        if solution[i] < 1:
            d_y[i] = -1
            count += 1
            if count == c:
                break
    return d_y

def threshold_method_soft(K, y_l, y_u, c):
    # move numpy array to torch tensor
    K = torch.from_numpy(K).float()
    y_l = torch.from_numpy(y_l).float()
    y_u = torch.from_numpy(y_u).float()
    d_y = torch.ones_like(y_l, requires_grad=True)
    # loss = -0.5 * || tanh(K (yl*d_y)) - yu || ^2 + beta * 0.125 * || 1-d_y || ^2
    beta = 0.1
    lr = 0.01
    T = 0.5
    for _ in range(300):
        diff = torch.tanh(T * K @ (y_l * d_y)) - y_u
        loss = -0.5 * torch.sum(diff * diff) + 0.125 * beta * torch.sum((1 - d_y) ** 2)
        g = torch.autograd.grad(loss, [d_y])[0]
        d_y.detach().add_(-lr * g)
    solution = d_y.detach().numpy()
    print(solution.shape)
    idx = np.argsort(solution)
    d_y = np.ones_like(solution)
    count = 0
    for i in idx:
        if solution[i] < 1:
            d_y[i] = -1
            count += 1
            if count == c:
                break
    return d_y

def probablistic_method(K, y_l, y_u, c):
    alpha = 0.5 * np.ones_like(y_l)
    tau = 0.5      # temperature
    lam = 0.1
    lr = 1.0e-5
    for i in range(100):
        epsilon = np.random.gumbel(len(y_l)) - np.random.gumbel(len(y_l))
        #print('epsilon: ', epsilon)
        tmp = np.exp((np.log(alpha / (1.0 - alpha)) + epsilon) / tau)
        #print('tmp: ', tmp)
        z = 2.0 / (1.0 + tmp) - 1.0     # normalize z from [0, 1] to [-1, 1]
        v = y_l * z
        grad_v = -K.T @ (K @ v - y_u)
        grad_z = grad_v * y_l
        grad_alpha = grad_z * (-2 * tmp / (1.0 + tmp) / (1.0 + tmp)) * (1.0 / alpha + 1.0 / (1.0 - alpha)) / tau
        grad_alpha += lam * alpha       # add a regularization term
        alpha -= lr * grad_alpha
        # project alpha to [0, 1]
        alpha = np.clip(alpha, 1.0e-3, 1-1.0e-3)
    # evaluate function value
    idx = np.argsort(alpha)[::-1]
    d_y = np.ones_like(y_l)
    count = 0
    for i in idx:
        if alpha[i] > 0.5:
            d_y[i] = -1
            count += 1
            if count == c:
                break
    #val = func_val(K, y_l, d_y, y_u)
    #print('#flip: ', np.sum(d_y < 0), 'function value: ', val)
    return d_y

def probablistic_method_soft(K, y_l, y_u, c):
    # move numpy array to torch tensor
    K = torch.from_numpy(K).float()
    n_l = len(y_l)
    y_l = torch.from_numpy(y_l).float()
    y_u = torch.from_numpy(y_u).float()
    d_y = torch.ones_like(y_l, requires_grad=True)
    # loss = -0.5 * || tanh(K (yl*d_y)) - yu || ^2 + beta * 0.125 * || 1-d_y || ^2
    alpha = 0.5 * torch.ones_like(y_l, requires_grad=True)
    lam = 0.01
    tau = 0.2
    T = 10
    lr = 1.0e-6
    for i in range(1500):
        U1 = torch.rand((n_l,))
        U2 = torch.rand((n_l,))
        epsilon = torch.log(torch.log(U1) / torch.log(U2))
        tmp = torch.exp((torch.log(alpha / (1.0 - alpha)) + epsilon) / tau)
        #print('tmp: ', tmp)
        z = 2.0 / (1.0 + tmp) - 1.0     # normalize z from [0, 1] to [-1, 1]
        v = y_l * z
        diff = torch.tanh(T * K @ v) - y_u
        loss = -0.5 * torch.sum(diff * diff) + 0.5 * lam * torch.sum(alpha * alpha)
        #print(loss.item())
        g = torch.autograd.grad(loss, [alpha])[0]
        alpha.detach().add_(-lr * g)
        alpha.detach().clamp_(1.0e-3, 1.0-1.0e-3)

    alpha = alpha.detach().numpy()
    # evaluate function value
    idx = np.argsort(alpha)[::-1]
    d_y = np.ones_like(y_l)
    count = 0
    for i in idx:
        if alpha[i] > 0.5 and count < c:
            d_y[i] = -1
            count += 1
            if count == c:
                break
    #val = func_val(K, y_l, d_y, y_u)
    #print('#flip: ', np.sum(d_y < 0), 'function value: ', val)
    return d_y
