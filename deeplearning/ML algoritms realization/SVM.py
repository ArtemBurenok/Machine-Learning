import numpy as np


class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_epoch=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_epoch = num_epoch
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_sample, n_features = X.shape

        self.w = np.zeros(n_sample)
        self.b = 0

        for _ in range(self.num_epoch):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        pass