import numpy as np


class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_samples, n_features), dtype=np.float64)
        self._var = np.zeros((n_samples, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[c == y]
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            self._priors[c, :] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(X) for x in X]
        return y_pred

    def _predict(self, X):
        posteriors = []

        for ind, x in enumerate(self._classes):
            prior = np.log(self._priors[ind])
            class_conditional =

    def _pdf(self, class_ind, x):
        mean = self._mean[class_ind]
        var = self._var[class_ind]
        numerator = np.exp(-(x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

