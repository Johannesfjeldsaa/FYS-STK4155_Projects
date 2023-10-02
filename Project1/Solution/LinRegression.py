### Import packages ###

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample

from sklearn import linear_model


class LinRegression:
    supported_methods = {'regression_method': ['OLS', 'Ridge', 'Lasso'],
                         'scaling_method': ['StandardScaling']}

    def __init__(self, poly_degree, x, y, z=None):

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
        self.lmb = None

        self.k_folds = None
        self.k_groups = None
        self.y_groups = None

        ### Define attributes of the linear regression
        self.splitted = False
        self.resampling = False

        self.regression_method = None
        self.scaling_method = None

        self.scaled = False


    def create_design_matrix(self, x, y, n):
        """
        Creates a design matrix from two input variables
        Copied from example week 35
        
        Parameters:
        x (array-like): The first variable
        y (array-like): The second variable
        n (int): The polynomial degree

        Returns:
        The design matrix
        """
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int((n+1)*(n+2)/2)		# Number of elements in beta
        X = np.ones((N,l))

        for i in range(1,n+1):
            q = int(i * (i + 1) / 2)
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
        Splits the data into test and training set using the scikit-learn 
        method train_test_split
        
        Parameters:
        test_size (float): The fraction of the data to use as test data
        
        Returns:
        X_train: Training part of design matrix
        X_test: Test part of design matrix
        y_train: Training part of output values
        y_test: Test part og output values
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


    def cross_validation_train_model(self, k_folds, regression_method=None, lmb=None):
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

        self.resampling = True   # set the marker for cross validation being executed

        if lmb is not None:
            self.lmb = lmb

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

            # perform training after which training method has been passed

            self.train_model(regression_method=regression_method, la=self.lmb,
                             X_train=train_matrix, y_train=y_train_cv)

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

    def scikit_cross_validation_train_model(self, k, regression_method=None, lmb=None):

        # make k groups
        kfold = KFold(n_splits=k, shuffle=True)

        if regression_method == 'OLS':
            OLS = LinearRegression(fit_intercept=False)

            # loop over trials in order to estimate the expectation value of the MSE
            estimated_mse_folds = cross_val_score(OLS, self.X, self.y,
                                                  scoring='neg_mean_squared_error', cv=kfold)
            estimated_r2_folds = cross_val_score(OLS, self.X, self.y, scoring='r2', cv=kfold)

        elif regression_method == 'Ridge':
            ridge = Ridge(lmb, fit_intercept=False)

            # loop over trials in order to estimate the expectation value of the MSE
            estimated_mse_folds = cross_val_score(ridge, self.X, self.y,
                                                  scoring='neg_mean_squared_error', cv=kfold)
            estimated_r2_folds = cross_val_score(ridge, self.X, self.y, scoring='r2', cv=kfold)

        elif regression_method == 'Lasso':
            lasso = Lasso(fit_intercept=False)

            # loop over trials in order to estimate the expectation value of the MSE
            estimated_mse_folds = cross_val_score(lasso, self.X, self.y,
                                                  scoring='neg_mean_squared_error', cv=kfold)
            estimated_r2_folds = cross_val_score(lasso, self.X, self.y, scoring='r2', cv=kfold)
        else:
            raise ValueError('A valid regression model is not given')

        return np.mean(-estimated_mse_folds), np.mean(estimated_r2_folds)

    def bootstrapping_train_model(self,n_bootstraps):
        self.resampling = True

        if self.splitted is not True:
            raise ValueError('Split data before perfoming bootstrapping')

        # The following (m x n_bootstraps) matrix holds the column vectors y_pred
        # for each bootstrap iteration.

        y_pred = np.zeros((self.y_test.shape[0], n_bootstraps))
        for i in range(n_bootstraps):
            x_, y_ = resample(self.X_train, self.y_train)

            # Evaluate the new model on the same test data each time.

            self.train_model(regression_method='OLS', X_train=x_, y_train=y_)

            y_pred[:, i] = self.predict_test().ravel()

        # Note: Expectations and variances taken w.r.t. different training
        # data sets, hence the axis=1. Subsequent means are taken across the test data
        # set in order to obtain a total value, but before this we have error/bias/variance
        # calculated per data point in the test set.
        # Note 2: The use of keepdims=True is important in the calculation of bias as this
        # maintains the column vector form. Dropping this yields very unexpected results.

        error = np.mean(np.mean((self.y_test.reshape(len(self.y_test),1) - y_pred) ** 2, axis=1, keepdims=True))
        bias = np.mean((self.y_test.reshape(len(self.y_test),1) - np.mean(y_pred, axis=1, keepdims=True)) ** 2)
        variance = np.mean(np.var(y_pred, axis=1, keepdims=True))

        return y_pred, error, bias, variance

    def scale(self, scaling_method=None):
        """
        Scales the data according to the chosen scaling method
        
        Parameters:
        scaling_method (string): The chosen scaling method.
        
        """
    
        if self.splitted is not True:
            raise ValueError('Split before you scale!')

        if (scaling_method is None) and (self.scaling_method is None):
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

        self.scaled = True # Marks that the data are now scaled


    def train_model(self, regression_method=None, train_on_scaled=False, la=None,
                    X_train=None, y_train=None):
        """
        Function for training the model.

        X_train and y_train given when performing resampling methods
        (Bootstrap and cross validation).

        Parameters
        ----------
        regression_method : String, optional
            The regression method to use. The default is None.
        train_on_scaled : String, optional
            Weather to train on scaled data or not. The default is False.
        la : Float, optional
            The lambda value to use for ridge or lasso regression. 
            The default is None.
        cv_X : TYPE, optional
            DESCRIPTION. The default is None.
        cv_y : TYPE, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        ArithmeticError
            Gives warning if the data is not split before training the model.
        ValueError
            Gives warning if train_on_scaled is True and data is not scaled

        Returns
        -------
        array-like
            The fitted beta parameters.

        """
    
        if self.splitted is not True and self.resampling is not True:
            raise ArithmeticError('Split data before performing model training.')


        if (regression_method is None) and (self.regression_method is None):
            print('No method for training was provided, using OLS')
            self.regression_method = 'OLS'
        else:
            if regression_method in self.supported_methods['regression_method']:
                self.regression_method = regression_method
            else:
                supported_methods = self.supported_methods['regression_method']
                raise ValueError(f'regression_method was {regression_method}, expected {supported_methods}')

        #train_on_scaled = train_on_scaled if train_on_scaled is not None else False

        if (X_train is None) and (y_train is None):

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

        #if self.resampling is True:
        #    if cv_X is not None:
        #        X_train = cv_X
        #    else:
        #        raise ValueError('Must pass in training matrices for finding beta in cross validation')
        #    if cv_y is not None:
        #        y_train = cv_y
        #    else:
        #        raise ValueError('Must pass in belonging y_train for finding beta in cross validation')

        if self.regression_method == 'OLS':
            self.beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
        elif self.regression_method == 'Ridge':
            cols = np.shape(X_train)[1]
            I = np.eye(cols, cols)
            self.beta = np.linalg.pinv(X_train.T @ X_train + la*I) @ X_train.T @ y_train
        elif self.regression_method == "Lasso":
            RegLasso = linear_model.Lasso(la, fit_intercept=False, max_iter=int(10e6))
            RegLasso.fit(X_train, y_train)
            self.beta = RegLasso.coef_
            
        return self.beta

    def predict(self):
        """
        Function for predicting the output using the trained model

        """
        self.y_pred = self.X @ self.beta

    def predict_training(self):
        """
        Function for predicting the training output using the trained model

        """
        if self.scaled is True:
            self.y_pred_train = self.X_train_scaled @ self.beta + self.y_scaler
        else:
            self.y_pred_train = self.X_train @ self.beta

        return self.y_pred_train

    def predict_test(self):
        """
        Function for predicting the test output using the trained model

        """
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
