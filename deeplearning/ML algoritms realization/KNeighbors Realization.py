import numpy as np


class KNeighbors:
    def __init__(self, neighbors):
        self.neighbors = neighbors
        self.target = None
        self.samples = None

    def fit(self, samples, target):
        assert(len(samples) == len(target))
        self.samples = np.array(samples)
        self.target = np.array(target)

    def predict(self, value):
        distances = np.ones(self.samples.shape[1])

        for i in range(self.samples.shape[1]):
            distance = np.abc(self.samples[i] - value)
            distances[i] = distance

        valueDict = {distances: self.target}
        keys = list(valueDict.keys()).sort()
        minArray = [keys[i] for i in range(2)]

        meanArr = []
        for key in minArray:
            meanArr.append(valueDict.get(key))
        return np.array(meanArr).mean()


neighbor = KNeighbors(neighbors=3)
