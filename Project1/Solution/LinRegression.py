### Import packages ###

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

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
        self.x = x
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
        self.y_groups = None

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


    def cross_validation_train_model(self, k_folds, regression_method=None):
        """
        Method to perform cross validation resampling and then train the model.
        This method have not considered scaling, and doesnt take any scaling inputs for now.
        Should be able to perform training on OLS, Ridge and Lasso.

        :param k: k folds to decide how large the resampling groups should be.

        :return:
        mean_B_parameters: optimal beta parameters for the model
        mean_MSE_test: The mean MSE obtained from the iterated test sets
        mean_MSE_train: The mean MSE obtained from the iterated training sets
        mean_R2_test: The mean R2 obtained from the iterated test sets
        mean_R2_train: The mean R2 obtained from the iterated training sets
        """

        self.cross_validation = True   # set the marker for cross validation being executed

        # create the k groups:

        self.k_folds = k_folds
        n = len(self.x)

        # Shuffle the data, and keep design matrix and z values aligned
        list_tuple_to_shuffle = list(zip(self.X, self.y))
        np.random.shuffle(list_tuple_to_shuffle)
        matrix_shuffled, z_shuffled = zip(*list_tuple_to_shuffle)

        # matrix_shuffled and z_shuffled come out as tuples, and so must be converted to lists.
        matrix_shuffled, z_shuffled = list(matrix_shuffled), list(z_shuffled)

        # Split the matrix into k groups of rows from the shuffled matrix
        group_size = n // self.k_folds  # divide with integral result (discard remainder)

        # makes k groups of the shuffled rows of the design matrix
        self.k_groups = [matrix_shuffled[i:i + group_size] for i in range(0, n, group_size)]
        self.y_groups = [z_shuffled[i:i + group_size] for i in range(0, n, group_size)]

        # Lists to store temporarily found scores and parameters
        opt_beta = []
        MSE_test = []
        MSE_train = []
        R2_test = []
        R2_train = []

        # Go through iterations of changing out test and training sets
        for i in range(self.k_folds):
            # Use the i-th part as the test set and the rest as the train set
            test_matrix = np.array(self.k_groups[i])
            train_matrix = np.array(np.concatenate(self.k_groups[:i]
                                                   + self.k_groups[i + 1:], axis=0))
            y_test_cv = (self.y_groups[i])
            y_train_cv = np.concatenate(self.y_groups[:i] + self.y_groups[i + 1:], axis=0)

            # perform for OLS first, but in Ridge and Lasso when working
            if self.regression_method == 'OLS':
                self.train_model(regression_method=self.regression_method,
                                 cv_X=train_matrix, cv_y=y_train_cv)
                opt_beta.append(self.beta)   # store optimal betas for each cross validation set

                # find for training set: X_training @ beta
                y_train_pred = train_matrix @ self.beta
                MSE_train.append(self.MSE(y_train_cv, y_train_pred))
                R2_train.append(self.R_squared(y_train_cv, y_train_pred))

                # find for test set: test_matrix @ beta
                y_test_pred = test_matrix @ self.beta
                MSE_test.append(self.MSE(y_test_cv, y_test_pred))
                R2_test.append(self.R_squared(y_test_cv, y_test_pred))

        B_matrix = np.array(opt_beta)
        opt_beta_model = []

        for i in range(len(B_matrix[0,:])):
            column = B_matrix[:,i]
            opt_beta_model.append(np.mean(column))

        return opt_beta_model, np.mean(MSE_test), np.mean(MSE_train), \
               np.mean(R2_test), np.mean(R2_train)

    def scikit_cross_validation_train_model(self, k):

        # make k groups
        kfold = KFold(n_splits=k, shuffle=True)

        if self.regression_method == 'OLS':
            OLS = LinearRegression(fit_intercept=False)

            # loop over trials in order to estimate the expectation value of the MSE
            estimated_mse_folds = cross_val_score(OLS, self.X, self.y,
                                                  scoring='neg_mean_squared_error', cv=kfold)
            print(estimated_mse_folds)

        elif self.regression_method == 'Ridge':
            pass
        elif self.regression_method == 'Lasso':
            pass
        else:
            raise ValueError('A valid regression model is not given')

        return np.mean(-estimated_mse_folds)


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

    def train_model(self, regression_method=None, train_on_scaled=None, la=None,
                    cv_X=None, cv_y=None):
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
            if cv_X is not None:
                X_train = cv_X
            else:
                raise ValueError('Must pass in training matrices for finding beta in cross validation')
            if cv_y is not None:
                y_train = cv_y
            else:
                raise ValueError('Must pass in belonging y_train for finding beta in cross validation')

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
