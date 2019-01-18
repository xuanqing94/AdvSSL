import autograd.numpy as np
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_svmlight_file

# fix random seed
np.random.seed(0)

class ManifoldMethod(object):
    def __init__(self, path, n_features=None, to_dense=False):
        features, labels = self.load_svm_data(path, n_features, to_dense)
        self.features, self.labels = features, labels
        self.labels /= 100.0

    def load_svm_data(self, path, n_features=None, to_dense=False):
        features, labels = load_svmlight_file(path, n_features)
        if to_dense:
            features = features.todense()
            n_data = features.shape[0]
            # add a bias term
            features = np.hstack((features, np.ones(n_data).reshape(-1, 1)))
        return features, labels 
    
    def set_ratio(self, train_ratio, label_ratio):
        self.train_ratio = train_ratio
        self.label_ratio = label_ratio
    
    def set_hparam(self, gamma, lam, beta):
        self.gamma = gamma
        self.lam = lam
        self.beta = beta

    def shuffle_data(self):
        # shuffle before split data
        n_data = self.features.shape[0]
        shuffle_idx = np.random.permutation(n_data)
        # do two inplace shuffle operations
        self.features = np.take(self.features, shuffle_idx, axis=0)
        self.labels = np.take(self.labels, shuffle_idx, axis=0)

    def split_data(self, train_ratio, label_ratio):
        assert label_ratio >= 0 and label_ratio <= 1
        assert train_ratio >= 0 and train_ratio <= 1
        # calculate number of data in splits
        n_data, _ = self.features.shape
        n_train = int(train_ratio * n_data)
        n_label_data = int(label_ratio * n_train)
        n_unlabel_data = n_train - n_label_data
        n_test_data = n_data - n_train
        # split data
        label_features = self.features[:n_label_data]
        unlabel_features = self.features[n_label_data:n_train]
        test_features = self.features[n_train:]
        label_labels = self.labels[:n_label_data]
        unlabel_labels = self.labels[n_label_data:n_train]
        test_labels = self.labels[n_train:]
        self.X_l, self.X_u, self.X_t, self.y_l, self.y_u, self.y_t = \
                label_features, unlabel_features, test_features, label_labels, unlabel_labels, test_labels

    @classmethod
    def similarity_matrix(cls, X_tr, gamma):
        tmp = X_tr @ X_tr.T
        n_tr = X_tr.shape[0]
        diag = np.diag(tmp)
        S = gamma * (2 * tmp - diag.reshape(1, n_tr) -
                   diag.reshape(n_tr, 1))
        return np.exp(S)

    @classmethod
    def laplacian(cls, similarity_matrix):
        D = np.diag(np.sum(similarity_matrix, axis=1))
        return D - similarity_matrix

    def training(self, n_trial=1):
        MM = ManifoldMethod  # abbrv.
        mse_lv, mse_uv, mse_tv = [], [], []
        for k_trial in range(n_trial):
            self.shuffle_data()
            self.split_data(self.train_ratio, self.label_ratio)
            d = self.X_l.shape[1]
            X_l, X_u = self.X_l, self.X_u
            y_l = self.y_l
            X_tr = np.vstack((X_l, X_u))
            S = MM.similarity_matrix(X_tr, self.gamma)
            L = MM.laplacian(S)
            tmp = np.linalg.inv(X_l.T @ X_l + self.lam * np.eye(d) +
                              self.beta * X_tr.T @ L @ X_tr) @ X_l.T
            w = np.dot(tmp, y_l)
            mse_l, mse_u, mse_t = self.evaluate(w)
            mse_lv.append(mse_l)
            mse_uv.append(mse_u)
            mse_tv.append(mse_t)
        return (np.mean(mse_lv), np.std(mse_lv)), \
                (np.mean(mse_uv), np.std(mse_uv)), \
                (np.mean(mse_tv), np.std(mse_tv))
    
    def training_optimal(self, n_trial=1):
        mse_lv, mse_uv, mse_tv = [], [], []
        for k_trial in range(n_trial):
            self.shuffle_data()
            # in the optimal case, we make label_ratio == 1
            # i.e. all training data are labelled
            self.split_data(self.train_ratio, 1)
            d = self.X_l.shape[1]
            X_l = self.X_l
            y_l = self.y_l
            # do OLS regression
            tmp = np.linalg.inv(X_l.T @ X_l + self.lam * np.eye(d)) @ X_l.T
            w = np.dot(tmp, y_l)
            mse_l, mse_u, mse_t = self.evaluate(w)
            mse_lv.append(mse_l)
            mse_uv.append(mse_u)
            mse_tv.append(mse_t)
        return (np.mean(mse_lv), np.std(mse_lv)), \
                (np.mean(mse_uv), np.std(mse_uv)), \
                (np.mean(mse_tv), np.std(mse_tv))


    def evaluate(self, w):
        """Calculate MSE in label, unlabel and test sets"""
        mse_l = np.sum((self.X_l @ w.T - self.y_l) ** 2) / self.X_l.shape[0]
        mse_u = np.sum((self.X_u @ w.T - self.y_u) ** 2) / self.X_u.shape[0]
        mse_t = np.sum((self.X_t @ w.T - self.y_t) ** 2) / self.X_t.shape[0]
        return mse_l, mse_u, mse_t
      

if __name__ == "__main__":
    filepath = "./data/cpusmall/cpusmall_scale"
    train_ratio = 0.1
    label_ratio = 0.8
    gamma = 1.0e-4
    lam = 1.0e-0
    beta = 0
    mm = ManifoldMethod(filepath, train_ratio, label_ratio, gamma, lam, beta, to_dense=True)
    for r in (0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.8):
        mm.reset_ratio(r, 1.0)
        (mean_l, std_l), (mean_u, std_u), (mean_t, std_t) = mm.training_optimal(n_trial=100)
        print(f'Mean_label: {mean_l:.3f},\tStd_label: {std_l:.3f}\nMean_unlabel: {mean_u:.3f},\tStd_unlabel: {std_u:.3f}\nMean_test: {mean_t:.3f},\tStd_test: {std_t:.3f}\n')
    #(mean_l, std_l), (mean_u, std_u), (mean_t, std_t) = mm.training(n_trial=1)
    #print(f'Mean_label: {mean_l:.3f},\tStd_label: {std_l:.3f}\nMean_unlabel: {mean_u:.3f},\tStd_unlabel: {std_u:.3f}\nMean_test: {mean_t:.3f},\tStd_test: {std_t:.3f}\n')
