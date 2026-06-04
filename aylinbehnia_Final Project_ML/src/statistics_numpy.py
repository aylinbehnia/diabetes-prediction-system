import numpy as np

class NumpyStats:

    @staticmethod
    def basic_stats(array):
        return {
            "mean": np.mean(array),
            "variance": np.var(array),
            "median": np.median(array)
        }

    @staticmethod
    def covariance(x, y):
        return np.cov(x, y)[0,1]

    @staticmethod
    def broadcasting_example(arr):
        return arr * 2

    @staticmethod
    def filter_above_mean(arr):
        return arr[arr > np.mean(arr)]