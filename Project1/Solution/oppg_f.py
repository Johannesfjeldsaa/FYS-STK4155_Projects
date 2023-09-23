import numpy as np

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

for i in range(k):
    # Use the i-th part as the test set and the rest as the train set
    test = groups[i]
    train = np.concatenate(groups[:i] + groups[i+1:],axis=0)

    # Now perform OLS, Ridge, Lasso



# range penalty parameter for ridge and lasso
lmb_Ridge_Lasso = np.linspace(0,10,20)
