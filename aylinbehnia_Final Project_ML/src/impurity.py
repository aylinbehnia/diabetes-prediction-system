import numpy as np

class Impurity:

    @staticmethod
    def gini(y):
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p**2)

    @staticmethod
    def entropy(y):
        p = np.bincount(y) / len(y)
        return -np.sum(p * np.log2(p + 1e-9))
