### Import packages ###

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


class LinRegression:
    supported_methods = {'regression_method': ['OLS', 'Ridge'],
                         'scaling_method': ['StandardScaling']}

    def __init__(self, poly_degree, x, y, z=None, regression_method=None, scaling_method='None'):

        self.poly_degree = poly_degree
        poly = PolynomialFeatures(degree=self.poly_degree)
        if z is None:
            self.X = poly.fit_transform(x.reshape(-1, 1))
            self.y = y
        else:
            # Horizontally concat columns to creat design matrix
            self.X = poly.fit_transform(np.c_[x, y])
            self.y = z

        # Define matrices and vectors
        self.X_train = None
        self.X_train_scaled = None
        self.X_test = None
        self.X_test_scaled = None
        self.X_scaler = None
        self.y_train = None
        self.y_train_scaled = None
        self.y_pred_train = None
        self.y_test = None
        self.y_pred_test = None
        self.y_scaler = None

        ### Define attributes of the linear regression
        self.splitted = False
        # Regression
        if regression_method is None:
            self.regression_method = 'OLS'
        else:
            if regression_method in self.supported_methods['regression_method']:
                self.regression_method = regression_method
            else:
                supported_methods = self.supported_methods['regression_method']
                raise ValueError(f'regression_method was {regression_method}, expected {supported_methods}')
        # Scaling
        self.scaled = False
        if scaling_method is None:
            self.scaling_method = None
        else:
            if scaling_method in self.supported_methods['scaling_method']:
                self.scaling_method = scaling_method
            else:
                supported_methods = self.supported_methods['scaling_method']
                raise ValueError(f'scaling_method was {scaling_method}, expected {supported_methods}')


    def check_vectors_same_length(self, a, b):
        """
        Check if two vectors (1D arrays) and have the same length.

        Parameters:
        a (array-like): The first variable to check.
        b (array-like): The second variable to check.

        Raises:
        ValueError: If the inputs are not valid vectors of the same length.

        Returns:
        None
        """
        # Check if both variables are 1D arrays and have the same length
        if len(a) != len(b):
            raise ValueError(f'input must be same length, not {len(a)} and {len(b)}')
        else:
            return True

    def split_data(self, test_size):
        """

        :param test_size:
        :return:
        """
        if self.scaled is True:
            raise ValueError('Split before you scale!')


        (self.X_train,
         self.X_test,
         self.y_train,
         self.y_test) = train_test_split(self.X, self.y,
                                         test_size=test_size)  # Splits data based on test_size

        self.splitted = True

        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale(self, scaling_method=None):
        if self.splitted is not True:
            raise ValueError('Split before you scale!')

        if scaling_method and self.scaling_method is None:
            print('No scaling method provided, using standard scaling')
            self.scaling_method = 'StandardScaling'
        else:
            if scaling_method in self.supported_methods['scaling_method']:
                self.scaling_method = scaling_method
            else:
                supported_methods = self.supported_methods['scaling_method']
                raise ValueError(f'scaling_method was {scaling_method}, expected {supported_methods}')

        if self.scaling_method == 'StandardScaling':
            self.y_scaler = np.mean(self.y_train)
            self.y_train_scaled = self.y_train - self.y_scaler

            self.X_scaler = np.mean(self.X_train, axis=0)
            self.X_train_scaled = self.X_train - self.X_scaler
            self.X_test_scaled = self.X_test - self.X_scaler  # Use mean from training data


        self.scaled = True

    def train_model(self, regression_method=None, train_on_scaled=None, la=None):
        if self.splitted is not True:
            raise ArithmeticError('Split data before performing model training.')

        if regression_method and self.regression_method is None:
            print('No method for training was provided, using OLS')
            self.regression_method = 'OLS'
        else:
            if regression_method in self.supported_methods['regression_method']:
                self.regression_method = regression_method
            else:
                supported_methods = self.supported_methods['regression_method']
                raise ValueError(f'regression_method was {regression_method}, expected {supported_methods}')


        if self.regression_method == 'Ridge':
            try:
                float(la)
            except ValueError:
                print(f'la must be float, not {type(la)}')

        train_on_scaled = train_on_scaled if train_on_scaled is not None else False

        if train_on_scaled:
            if self.scaled is True:
                X_train = self.X_train_scaled
                y_train = self.y_train_scaled
            else:
                raise ValueError(f'Scale data before using train_on_scaled=True')
        elif train_on_scaled is False:
            X_train = self.X_train
            y_train = self.y_train
        else:
            raise ValueError(f'train_on_scaled takes arguments True or False, not {train_on_scaled}')

        if self.regression_method == 'OLS':
            self.beta = np.linalg.pinv(X_train.T @ X_train) @ (X_train.T @ y_train)
        elif self.regression_method == 'ridge':
            I = np.eye(self.poly_degree, self.poly_degree)
            self.beta = np.linalg.pinv(X_train.T @ X_train + la*I) @ X_train.T @ y_train
            
        return self.beta

    def predict_training(self):
        if self.scaled is True:
            self.y_pred_train = self.X_train_scaled @ self.beta + self.y_scaler
        else:
            self.y_pred_train = self.X_train @ self.beta

        return self.y_pred_train

    def predict_test(self):
        if self.scaled is True:
            self.y_pred_test = (self.X_test_scaled @ self.beta) + self.y_scaler
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
        TSS = np.sum((true_y - np.mean(true_y)) ** 2)  # Total sum of squares, measures the total variability

        R2 = 1 - SSR/TSS

        return R2