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

#Create a linear regression class
polydegree = 5
Linreg = LinRegression(polydegree, x, y)

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

Linreg.cross_validation = True

for i in range(k):
    # Use the i-th part as the test set and the rest as the train set
    test_data = groups[i]
    train_data = np.concatenate(groups[:i] + groups[i+1:],axis=0)
    y_train_data = np.exp(-train_data ** 2) + 1.5 * np.exp(-(train_data - 2) ** 2) + np.random.normal(0, 0.1, train_data.shape)
    y_test_data = np.exp(-test_data ** 2) + 1.5 * np.exp(-(test_data - 2) ** 2) + np.random.normal(0, 0.1, test_data.shape)

    # create design_matrices out of the new data
    X_kfold_train = Linreg.create_design_matrix(train_data,y_train_data,n)
    X_kfold_test = Linreg.create_design_matrix(test_data, y_test_data, n)


    # Now perform OLS, Ridge, Lasso
    Linreg.train_model(regression_method='OLS', train_on_scaled=False, X_cv_train=X_kfold_train, y_cv_train=y_train_data) ## fordi data splittes med cross validation blir det feil i linreg classe her!
    print(Linreg.beta)


# range penalty parameter for ridge and lasso
lmb_Ridge_Lasso = np.linspace(0,10,20)
