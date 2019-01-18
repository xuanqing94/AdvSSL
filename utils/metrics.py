import numpy as np

def RMSE(y_pred, y_truth):
    mse = np.sum((y_pred - y_truth) ** 2) / len(y_truth)
    sqrt_mse = np.sqrt(mse)
    return sqrt_mse


def accuracy(y_pred, y_truth):
    correct = np.sum(y_pred == y_truth)
    return correct / len(y_truth)
