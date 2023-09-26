### Import packages ###

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import random

class LinRegression:
    supported_methods = {'regression_method': ['OLS', 'Ridge', 'Lasso'],
                         'scaling_method': ['StandardScaling']}


    def __init__(self, poly_degree, x, y, z=None, regression_method=None, scaling_method=None):

        self.poly_degree = poly_degree

        if z is None:
            self.X = PolynomialFeatures(degree=self.poly_degree).fit_transform(x.reshape(-1, 1))
            self.y = y
        else:
            # Horizontally concat columns to creat design matrix
            self.X = self.create_design_matrix(x, y, self.poly_degree)
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
        self.beta = None
        self.k_folds = None
        self.k_groups = None

        ### Define attributes of the linear regression
        self.splitted = False
        self.cross_validation = False
        self.bootstrapping = False

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

    def create_design_matrix(self, x, y, n ):  # From example week 35
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int((n+1)*(n+2)/2)		# Number of elements in beta
        X = np.ones((N,l))

        for i in range(1,n+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                X[:,q+k] = (x**(i-k))*(y**k)

        return X


    def design_matrix_equal_identity(self):
        cols = np.shape(self.X)[1]
        self.X = np.eye(cols, cols)

        if (self.X.T @ self.X).all() == (np.eye(cols, cols)).all():
            print('hurra')



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


    def create_kfold_groups(self, k_folds):
        """
        Method that will resample the data in k-fold groups.

        :return: k number of equally large groups with entries in x reshuffled
        """

        # For 1D cases:
        """
        np.random.shuffle(self.x)  # Shuffles with replacement so x gets shuffled
        n = len(self.x)  # Number of entries

        # Split the data into k equal groups
        group_size = n // k_folds  # divide with integral result (discard remainder)
        self.k_groups = [self.x[i:i + group_size] for i in range(0, n, group_size)]

        return self.k_groups
        """

        #  For 2D cases
        self.k_folds = k_folds
        n = len(self.x)

        # Shuffle the data, and keep design matrix and z values aligned
        list_tuple_to_shuffle = list(zip(self.X, self.y))
        random.shuffle(list_tuple_to_shuffle)
        matrix_shuffled, z_shuffled = zip(*list_tuple_to_shuffle)

        # matrix_shuffled and z_shuffled come out as tuples, and so must be converted to lists.
        matrix_shuffled, z_shuffled = list(matrix_shuffled), list(z_shuffled)

        # Split the matrix into k groups of rows from the shuffled matrix
        group_size = n // self.k_folds  # divide with integral result (discard remainder)

        # makes k groups of the shuffled rows of the design matrix

        self.k_groups = [matrix_shuffled[i:i + group_size] for i in range(0, n, group_size)]

        return self.k_groups


    def create_list_train_test_cross_validation(self):

        test_rows_iterations = []   # List of the test matrice row groups as they get iterated through
        train_rows_iterations = []   #  List of train matrices row groups as they get iterated through

        if self.k_groups is not None:
            for i in range(self.k_folds):
                # Use the i-th part as the test set and the rest as the train set
                test_rows_iterations.append(self.k_groups[i])
                train_rows_iterations.append(np.concatenate(self.k_groups[:i] + self.k_groups[i+1:],axis=0))
        else:
            raise ValueError('You must divide into groups before performing cross validation')

        return test_rows_iterations, train_rows_iterations

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
            # assuming std = 1
            self.y_scaler = np.mean(self.y_train)
            self.y_train_scaled = self.y_train - self.y_scaler

            self.X_scaler = np.mean(self.X_train, axis=0)
            self.X_train_scaled = self.X_train - self.X_scaler
            self.X_test_scaled = self.X_test - self.X_scaler  # Use mean from training data


        self.scaled = True

    def train_model(self, regression_method=None, train_on_scaled=None, la=None): # kfold_train=None, kfold_test=None):
        if self.splitted is not True and self.cross_validation is not True:
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

        if self.cross_validation is not True:
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

        if self.cross_validation is True:
            X_train = self.X
            y_train = self.y

        if self.regression_method == 'OLS':
            self.beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
        elif self.regression_method == 'ridge':
            cols = np.shape(X_train)[1]
            I = np.eye(cols, cols)
            self.beta = np.linalg.pinv(X_train.T @ X_train + la*I) @ X_train.T @ y_train
            
        return self.beta

    def predict(self):
        self.y_pred = self.X @ self.beta

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

        mean_true_y = np.mean(true_y)
        SSR = np.sum((true_y - predicted_y) ** 2)  # Residual som of squares, measures the unexplained variability ("errors")
        TSS = np.sum((true_y - mean_true_y) ** 2)  # Total sum of squares, measures the total variability

        R2 = 1 - SSR/TSS

        return R2
