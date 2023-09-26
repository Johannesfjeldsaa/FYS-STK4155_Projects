import numpy as np
from LinRegression import LinRegression

"""
Class to provide resampling techniques.
"""

# gjøre slik at den kan ta inn 1D og 2D

class Resampling:

    def __init__(self, k_folds, polydegree, x, y, z,):

        self.x = x
        self.y = y
        self.z = z
        self.k = k_folds
        self.n = len(self.x)

    def create_kfold_groups(self, k_folds):
        """
        Method that will resample the data in k-fold groups.

        :return: k number of equally large groups with entries in x reshuffled
        """
        np.random.shuffle(self.x)  # Shuffles with replacement so x gets shuffled
        n = len(self.x)  # Number of entries

        # Split the data into k equal groups
        group_size = n // k_folds  # divide with integral result (discard remainder)
        self.k_groups = [self.x[i:i + group_size] for i in range(0, n, group_size)]

        return self.k_groups

    def create_x_data_cross_validation(self, k_folds):

        test_data_iteration_k = []   # List of the test groups as they get iterated through
        train_data_iteration_k = []   #  List of train groups as they get iterated through

        if self.k_groups is not None:
            for i in range(k_folds):
                # Use the i-th part as the test set and the rest as the train set
                test_data_iteration_k.append(self.groups[i])
                train_data_iteration_k.append(np.concatenate(self.groups[:i] + self.groups[i+1:],axis=0))

        return test_data_iteration_k, train_data_iteration_k





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