import numpy as np

"""
Class to provide resampling techniques.
"""


class Resampling:

    def __init__(self,k_folds, x, y):

        self.x = x
        self.y = y
        self.k = k_folds
        self.n = len(self.x)

    def cross_validation(self):
        """
        Method that will resample the data in k-fold groups

        :return:
        """
        np.random.shuffle(self.x)

        # Split the data into k equal groups
        group_size = self.n // self.k  # divide with integral result (discard remainder)
        groups = [self.x[i:i + group_size] for i in range(0, self.n, group_size)]

        return groups

    def scikit_cross_validation(self):

        pass

    def bootstrapping(self):
        """
        Method to reshuffle/split the data with bootstrapping.
        :return:
        """