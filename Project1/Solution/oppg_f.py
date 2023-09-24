import numpy as np
from LinRegression import LinRegression
"""
Part f): Cross-validation as resampling techniques, adding more complexity¶
The aim here is to write your own code for another widely popular resampling technique,
 the so-called cross-validation method.

Implement the k-fold cross-validation algorithm (write your own code) and evaluate again
 the MSE function resulting from the test folds. You can compare your own code with that from
 Scikit-Learn if needed.

Compare the MSE you get from your cross-validation code with the one you got from your
bootstrap code. Comment your results. Try 5-10 folds.
You can also compare your own cross-validation code with the one provided by Scikit-Learn.

In addition to using the ordinary least squares method, you should include
both Ridge and Lasso regression.

"""

# number of datapoints
n=100

#Function to test on
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2) + np.random.normal(0, 0.1, x.shape)

polydegree = 5

np.random.seed(1917)

# number of datapoints
n=100

#Function to test on
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2) + np.random.normal(0, 0.1, x.shape)

# Divide into k more or less equally sized subsets, kfolds
k = 10

# Shuffle the data
x_shuffle = np.random.shuffle(x) #Shuffles with replacement so x gets shuffled

# Split the data into k equal groups
group_size = n // k # divide with integral result (discard remainder)
groups = [x[i:i+group_size] for i in range(0, n, group_size)]

# loop through the groups, using one of them as a test set each time for performing
# different regression and MSE testing

MSE_score_test_data = []
MSE_score_train_data = []

for i in range(k):
    # Use the i-th part as the test set and the rest as the train set
    test_data = groups[i]
    train_data = np.concatenate(groups[:i] + groups[i+1:],axis=0)
    y_train_data = np.exp(-train_data ** 2) + 1.5 * np.exp(-(train_data - 2) ** 2) + np.random.normal(0, 0.1, train_data.shape)
    y_test_data = np.exp(-test_data ** 2) + 1.5 * np.exp(-(test_data - 2) ** 2) + np.random.normal(0, 0.1, test_data.shape)

    # Create linreggression class for each new x and y
    Linreg_train = LinRegression(polydegree, train_data, y_train_data)
    Linreg_test = LinRegression(polydegree, test_data, y_test_data)
    Linreg_train.cross_validation = True
    Linreg_test.cross_validation = True

    # Find betas OLS, Ridge, Lasso
    Linreg_train.train_model(regression_method='OLS', train_on_scaled=False)
    Linreg_test.train_model(regression_method='OLS', train_on_scaled=False)

    # Predict model data
    Linreg_train.predict()
    Linreg_test.predict()

    # Perform MSE and saving them for each K group
    MSE_score_train_data.append(Linreg_train.MSE(y_train_data,Linreg_train.y_pred))
    MSE_score_test_data.append(Linreg_test.MSE(y_test_data, Linreg_test.y_pred))

# range penalty parameter for ridge and lasso
lmb_Ridge_Lasso = np.linspace(0,10,20)
