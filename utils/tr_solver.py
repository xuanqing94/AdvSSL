import numpy as np

def trust_region_solver(M, g, d_max, max_iter=100, stepsize=1.0e-2):
    """Solves trust region problem with gradient descent
    maximize 1/2 * x^T M x + g^T x
    s.t. |x|_2 <= d_max
    initialize x = g / |g| * d_max
    """
    x = g / np.linalg.norm(g) * d_max
    for _ in range(max_iter):
        # gradient ascent
        x = x + stepsize * (M @ x + g)
        # projection to sphere
        x = x / np.linalg.norm(x) * d_max
        ## debug
        #loss = 0.5 * x.T @ M @ x + g.T @ x
        #print(f'Loss: {loss}')
    return x

