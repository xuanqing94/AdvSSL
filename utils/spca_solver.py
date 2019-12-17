import numpy as np
from sklearn.decomposition import SparsePCA


def sparse_pca(K, alpha, ridge_alpha):
    transformer = SparsePCA(n_components=1, alpha=alpha, ridge_alpha=ridge_alpha, normalize_components=False, random_state=0)
    transformer.fit(K)
    val = transformer.components_[0]
    print('#nnz: ', np.sum(np.abs(val) > 1.0e-10))
    #print(np.sum(val * val))
    #val = np.random.randn(K.shape[1])
    return val / np.linalg.norm(val)
