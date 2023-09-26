import numpy as np
from LinRegression import LinRegression
from Franke_function import FrankeFunction
import matplotlib.pyplot as plt
import random

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
"""

"""
Now test for the frankefunction
"""

"""
x = np.linspace(-3, 3, n).reshape(-1, 1)
x_shuffle = np.random.shuffle(x) #Shuffles with replacement so x gets shuffled
y = np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2) + np.random.normal(0, 0.1, x.shape)

# Split the data into k equal groups
group_size = n // k # divide with integral result (discard remainder)
groups = [x[i:i+group_size] for i in range(0, n, group_size)]

# loop through the groups, using one of them as a test set each time for performing
# different regression and MSE testing

"""

"""
now plot for model complexity and average of each k fold
"""

"""

n= 100
k = 10
x = np.linspace(-3, 3, n).reshape(-1, 1)

mean_MSE_training_from_cross_validation = []
mean_MSE_test_from_cross_validation = []

for polydegree in range(polydegree+1):

    if polydegree != 0:

        # Split the data into k equal groups
        group_size = n // k  # divide with integral result (discard remainder)
        groups = [x[i:i + group_size] for i in range(0, n, group_size)]

        MSE_score_test_data = []
        MSE_score_train_data = []

        for i in range(k):
            # Use the i-th part as the test set and the rest as the train set
            test_data = groups[i]
            train_data = np.concatenate(groups[:i] + groups[i+1:],axis=0)
            y_train_data = np.exp(-train_data ** 2) + 1.5 * np.exp(-(train_data - 2) ** 2)\
                           #+ np.random.normal(0, 0.1, train_data.shape)
            y_test_data = np.exp(-test_data ** 2) + 1.5 * np.exp(-(test_data - 2) ** 2)\
                          #+ np.random.normal(0, 0.1, test_data.shape)

            z_train = FrankeFunction(train_data, y_train_data)
            z_test = FrankeFunction(test_data, y_test_data)

            # Create linreggression class for each new x and y
            Linreg_franke_train = LinRegression(polydegree, train_data, y_train_data, z_train)
            Linreg_franke_train.cross_validation = True

            Linreg_franke_test = LinRegression(polydegree, test_data, y_test_data, z_test)
            Linreg_franke_test.cross_validation = True

            # Find betas OLS, Ridge, Lasso
            Linreg_franke_train.train_model(regression_method='OLS', train_on_scaled=False)
            Linreg_franke_test.train_model(regression_method='OLS', train_on_scaled=False)

            # Predict model data
            Linreg_franke_train.predict()
            Linreg_franke_test.predict()

            # Perform MSE and saving them for each K group
            MSE_score_train_data.append(Linreg_franke_train.MSE(y_train_data,Linreg_franke_train.y_pred))
            MSE_score_test_data.append(Linreg_franke_test.MSE(y_test_data, Linreg_franke_test.y_pred))

        mean_MSE_training_from_cross_validation.append(np.mean(MSE_score_train_data))
        mean_MSE_test_from_cross_validation.append(np.mean(MSE_score_test_data))


plt.figure()
plt.xlabel('model complexity')
plt.ylabel('Mean MSE from cross validation')
plt.plot(range(1,polydegree+1), mean_MSE_training_from_cross_validation, label='MSE train')
plt.plot(range(1,polydegree+1), mean_MSE_test_from_cross_validation, label='MSE test')
plt.legend()
plt.show()

"""

"""
Now redo with the design matrix being randomly shuffled by rows, taking out k-1 row as training 
and  k row is design matrix for testing

"""

polydegree = 3
x = np.linspace(0,10,100)
y = np.linspace(0,10,100)
z = FrankeFunction(x,y)
n = len(x)

# lag design matrix, which needs to be reshuffled ( together with z ) an done cross validation on rows
cross_validation = LinRegression(polydegree,x,y,z)

k = 10

# shuffle data and create groups

k_groups = cross_validation.create_kfold_groups(5)

# create lists of matrices to perform MSE, r^2 find optimal betas on

test_matrices_iterations, train_matrices_iterations = cross_validation.create_list_train_test_cross_validation()

print(test_matrices_iterations[0])
print(train_matrices_iterations[0])

# loop through the groups, using one of them as a test set each time for performing
# different regression and MSE testing

opt_training_beta = []
opt_test_beta = []
MSE_score_test = []
MSE_score_test = []



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





cross_validation_test = LinRegression(polydegree,x,y,z)


