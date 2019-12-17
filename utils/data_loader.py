import numpy as np
from scipy.sparse import vstack
from sklearn.datasets import load_svmlight_file, load_svmlight_files
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
    """Mnist is already scaled"""
    # load_data
    features, labels = load_svmlight_file('./data/mnist_1_7.scale')
    features = np.asarray(features.todense())
    n_data = features.shape[0]
    # add a bias term
    features = np.hstack((features, np.ones(n_data).reshape(-1, 1)))
    # make 7 as negative class
    labels[labels == 7] = -1
    return features, labels 

def load_a9a():
    files = ['./data/a9a', './data/a9a.t']
    features = [None, None]
    labels = [None, None]
    for i in range(2):
        # load_data
        feature, label = load_svmlight_file(files[i], n_features=123)
        feature = np.asarray(feature.todense())
        n_data = feature.shape[0]
        features[i] = feature
        labels[i] = label
    # concatenate
    print(features[0].shape, features[1].shape)
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    # scale features to zero-mean unit-var
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)
    pos = np.sum(labels == 1)
    neg = np.sum(labels == -1)
    return features, labels

def load_rcv1():
    # load_data
    feature, label = load_svmlight_file('./data/rcv1_train.binary')
    feature = np.asarray(feature.todense())
    n_data = feature.shape[0]
    # scale features to zero-mean unit-var
    """
    scaler = StandardScaler()
    scaler.fit(feature)
    feature = scaler.transform(feature)
    pos = np.sum(label == 1)
    neg = np.sum(label == -1)
    print(f"{pos}, {neg}")
    """
    return feature[:10000] * 10, label[:10000]

def load_e2006():
    # laod data
    feature_tr, label_tr, feature_te, label_te = load_svmlight_files(['./data/E2006.train', \
            './data/E2006.test'], n_features=150360)
    feature = vstack([feature_tr, feature_te])
    # expand 1 dimension to labels
    label = np.concatenate([label_tr, label_te], axis=0)
    # remove outliers from labels
    std_y = np.std(label)
    mean_y = np.mean(label)
    mask = np.logical_and(label > mean_y - 3.0 * std_y, label < mean_y + 3.0 * std_y)
    print(f'keep {np.sum(mask)} / {len(mask)} rows')
    # select rows
    feature = feature[mask]
    label = label[mask]

    # scale labels by standard scaler
    label = label[:, None]
    scaler = MinMaxScaler()
    scaler.fit(label)
    label = scaler.transform(label).squeeze()
    return feature * 10, label
