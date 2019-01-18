import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_cadata(normalize=True):
    # load_data
    features, labels = load_svmlight_file('./data/cadata_sample')
    features = np.asarray(features.todense())
    n_data = features.shape[0]
    # add a bias term
    features = np.hstack((features, np.ones(n_data).reshape(-1, 1)))
    # expand 1 dimension to labels
    labels = labels[:, None]
    # scaler 1: scale features to zero-mean unit-var
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)
    # scaler 2: scale labels to [0, 1] by min-max
    scaler = MinMaxScaler()
    scaler.fit(labels)
    labels = scaler.transform(labels).squeeze()
    return features, labels 

def load_mnist():
    # load_data
    features, labels = load_svmlight_file('./data/mnist_1_7.scale')
    features = np.asarray(features.todense())
    n_data = features.shape[0]
    # add a bias term
    features = np.hstack((features, np.ones(n_data).reshape(-1, 1)))
    # make 7 as negative class
    labels[labels == 7] = -1
    return features, labels 

