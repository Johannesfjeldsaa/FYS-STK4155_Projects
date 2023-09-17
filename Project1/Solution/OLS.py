### Import packages ###

import numpy as np
from sklearn.model_selection import train_test_split

### 1D OLS ###

class OLS():
    def __init__(self, x, y, z=None):
        if z is not None:
            X = np.column_stack((x.ravel(), y.ravel()))
            self.X = np.column_stack((np.ones(X.shape[0]), X))
            self.y = z.ravel()
        else:
            self.X = np.column_stack(np.column_stack((np.ones(X.shape[0]), x.ravel())))
            self.y = y.ravel()

        self.splitted = False
        self.scaled = False
        self.scaling_methode = None

    def check_vectors_same_length(self, a, b):
        """
        Check if two variables are vectors (1D arrays) and have the same length.

        Parameters:
        a (array-like): The first variable to check.
        b (array-like): The second variable to check.

        Raises:
        ValueError: If the inputs are not valid vectors of the same length.

        Returns:
        None
        """
        # Convert the inputs to NumPy arrays (if they are not already)
        a = np.asarray(a)
        b = np.asarray(b)

        # Check if both variables are 1D arrays and have the same length
        if a.ndim == 1 and b.ndim == 1 and len(a) == len(b):
            return
        else:
            raise ValueError("Input variables must be 1D arrays of the same length.")


    def split_data(self, test_size):
        '''

        :param test_size:
        :return:
        '''
        if self.scaled is True:
            raise ValueError('Split before you scale!')

        self.splitted = True

        (self.X_train,
         self.X_test,
         self.y_train,
         self.y_train) = train_test_split(self.X, self.y,
                                          test_size=test_size)  # Splits data based on test_size


    def standard_scaling(self):
        if self.splitted is not True:
            raise ValueError('Split before you scale!')

        self.scaled = True
        self.scaling_methode = 'StandardScaling'

        self.y_train_mean = np.mean(self.y_train)
        self.y_train_scaled = self.y_train - self.y_train_mean

        self.X_train_mean = np.mean(self.X_train, axis=0)
        self.X_train_scaled = self.X_train - self.X_train_mean
        self.X_test_scaled = self.X_test - self.X_train_mean  # Use mean from training data



    def train_by_OLS(self, train_on_scaled=None):
        if self.splitted is not True:
            raise ArithmeticError('Split data before performing model training')

        if train_on_scaled is None or True:
            if self.scaled is True:
                if self.scaling_methode == 'Standardscaling':
                    beta = (np.linalg.inv(self.X_train_scaled.T @ self.X_train_scaled) @
                            self.X_train_scaled.T @ self.y_train_scaled)
            else:
                raise ValueError('Parse train_on_scaled=False in order to train on unscaled data. '
                                 'Else perform scaling and repeat command')

        elif train_on_scaled is False:
            beta = (np.linalg.inv(self.X_train.T @ self.X_train) @
                    self.X_train.T @ self.y_train)

        else:
            raise ValueError(f'parameter train_on_scaled takes bolean True or False not {train_on_scaled}')

        self.beta = beta

        return beta

    def predict_traing(self):
        self.y_pred_train = self.X_train @ self.beta

        return self.y_pred_train

    def predict_test(self):
        if self.scaled is True:
            if self.scaling_methode == 'Standardscaling':
                self.y_pred_test = (self.X_test @ self.beta) + self.y_train_mean
        else:
            self.y_pred_test = (self.X_test @ self.beta)

        return self.y_pred_test

    def MSE(self, true_y, predicted_y):
        '''
        Calculates the the mean squared error (MSE) of the model.
        In essence we calculate the average error for the predicted y_i's compared to the true y_i's.

        :param true_y:
        :param predicted_y:
        :return:
        '''

        self.check_vectors_same_length(true_y, predicted_y)

        n = len(true_y)
        SSR = np.sum((true_y - predicted_y) ** 2) # Residual som of squares, measures the unexplained variability ("errors")
        MSE = (1 / n) * SSR
        return MSE

    def R_squared(self, true_y, predicted_y):
        '''
        Calculates the coefficient of determination R^2. R^2 quantifies the proportion of total variability in the
        dataset that is explained by the model.

        :param true_y:
        :param predicted_y:
        :return:
        '''

        self.check_vectors_same_length(true_y, predicted_y)

        SSR = np.sum((true_y - predicted_y) ** 2)  # Residual som of squares, measures the unexplained variability ("errors")
        TSS = np.sum((true_y - np.mean(true_y)) ** 2) # Total sum of squares, measures the total variability

        R2 = 1 - SSR/TSS

        return R2