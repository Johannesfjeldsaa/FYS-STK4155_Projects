import numpy as np
from LinRegression import LinRegression
from Franke_function import FrankeFunction
import matplotlib.pyplot as plt
import random


polydegree = 10
x = np.linspace(0,1,100)
y = np.linspace(0,1,100)
z = FrankeFunction(x,y)
k = 5
nlambdas = 10
lambdas = np.logspace(-5, 1, nlambdas)
np.random.seed(1917)

# store the results for each model complexity
optimal_beta_cross_validation = []
MSE_test = []
MSE_train = []
R2_test = []
R2_train = []

for polydegree in range(polydegree+1):

    if polydegree != 0: # skipping the intercept

        cross_validation = LinRegression(polydegree, x, y, z)  # this needs to be remade each time
        mean_B_parameters, mean_MSE_test, mean_MSE_train, mean_R2_test, mean_R2_train \
            = cross_validation.cross_validation_train_model(k, regression_method='OLS')

        optimal_beta_cross_validation.append(mean_B_parameters)
        MSE_test.append(mean_MSE_test)
        MSE_train.append(mean_MSE_train)
        R2_test.append(mean_R2_test)
        R2_train.append(mean_R2_train)

"""
using scikit_learn library
"""

mean_scikit_MSE_cv = []  # test mse
mean_scikit_r2_cv = [] # test r2

for polydegree in range(polydegree+1):

    if polydegree != 0:

        scikit_cv = LinRegression(polydegree,x,y,z)
        mean_scikit_MSE_cv.append(scikit_cv.scikit_cross_validation_train_model(k,regression_method='OLS')[0])
        mean_scikit_r2_cv.append(scikit_cv.scikit_cross_validation_train_model(k,regression_method='OLS')[1])

""" Performing cross validation with ridge """
optimal_beta_cross_validation_Ridge = []
MSE_test_Ridge = []
MSE_train_Ridge = []
R2_test_Ridge = []
R2_train_Ridge = []

for polydegree in range(polydegree+1):

    if polydegree != 0: # skipping the intercept
        for lambd in lambdas:
            cross_validation = LinRegression(polydegree, x, y, z)  # this needs to be remade each time
            mean_B_parameters, mean_MSE_test, mean_MSE_train, mean_R2_test, mean_R2_train \
                = cross_validation.cross_validation_train_model(k, regression_method='Ridge', lmb=lambd)

            optimal_beta_cross_validation_Ridge.append(mean_B_parameters)
            MSE_test_Ridge.append(mean_MSE_test)
            MSE_train_Ridge.append(mean_MSE_train)
            R2_test_Ridge.append(mean_R2_test)
            R2_train_Ridge.append(mean_R2_train)

mean_scikit_MSE_cv_Ridge = []  # test mse
mean_scikit_r2_cv_Ridge = [] # test r2

for polydegree in range(polydegree+1):

    if polydegree != 0:

        for lambd in lambdas:
            scikit_cv = LinRegression(polydegree,x,y,z)
            mean_scikit_MSE_cv_Ridge.append(scikit_cv.scikit_cross_validation_train_model(k,regression_method='Ridge', lmb=lambd)[0])
            mean_scikit_r2_cv_Ridge.append(scikit_cv.scikit_cross_validation_train_model(k,regression_method='Ridge', lmb=lambd)[1])

""" Performing cross validation with Lasso """

optimal_beta_cross_validation_Lasso = []
MSE_test_Lasso = []
MSE_train_Lasso = []
R2_test_Lasso = []
R2_train_Lasso = []

for polydegree in range(polydegree+1):

    if polydegree != 0: # skipping the intercept
        for lambd in lambdas:
            cross_validation = LinRegression(polydegree, x, y, z)  # this needs to be remade each time
            mean_B_parameters, mean_MSE_test, mean_MSE_train, mean_R2_test, mean_R2_train \
                = cross_validation.cross_validation_train_model(k, regression_method='Lasso', lmb=lambd)

            optimal_beta_cross_validation_Lasso.append(mean_B_parameters)
            MSE_test_Lasso.append(mean_MSE_test)
            MSE_train_Lasso.append(mean_MSE_train)
            R2_test_Lasso.append(mean_R2_test)
            R2_train_Lasso.append(mean_R2_train)

mean_scikit_MSE_cv_Lasso = []  # test mse
mean_scikit_r2_cv_Lasso = [] # test r2

for polydegree in range(polydegree+1):

    if polydegree != 0:

        scikit_cv = LinRegression(polydegree,x,y,z)
        mean_scikit_MSE_cv_Lasso.append(scikit_cv.scikit_cross_validation_train_model(k,regression_method='Lasso')[0])
        mean_scikit_r2_cv_Lasso.append(scikit_cv.scikit_cross_validation_train_model(k,regression_method='Lasso')[1])


"""
Compare own code to scikit code (can add in bootstrap later here)
"""

plt.figure()
plt.xlabel('model complexity')
plt.ylabel('MSE score')
plt.plot(range(1,polydegree + 1), MSE_test, 'r', label='Own code')
plt.plot(range(1,polydegree + 1), mean_scikit_MSE_cv, 'b', label='Scikit')
plt.legend()
plt.show()

plt.figure()
plt.xlabel('model complexity')
plt.ylabel('R2 score')
plt.plot(range(1,polydegree + 1), R2_test, 'r', label='Own code')
plt.plot(range(1,polydegree + 1), mean_scikit_r2_cv, 'b', label='Scikit')
plt.legend()
plt.show()

fig, ax1 = plt.subplots(figsize=(9, 6))
ax2 = ax1.twinx()
plt.xlabel('model complexity')
ax1.set_ylabel('Mean MSE from cross validation')
ax1.plot(range(1,polydegree+1), MSE_train, 'b', label='MSE train')
ax1.plot(range(1,polydegree+1), MSE_test, 'g', label='MSE test')
ax1.legend(loc='lower right')
ax2.set_ylabel('Mean R2 value cross validation')
ax2.plot(range(1,polydegree+1), R2_train, 'purple', label='R2 train')
ax2.plot(range(1,polydegree+1), R2_test, 'r', label='R2 test')
ax2.legend(loc='center right')
plt.show()


" plotting for ridge and lasso, comparing to scikit aswell"

plt.figure()
plt.xlabel('lambda value')
plt.ylabel('R2 score')
plt.plot(int(np.log(lambdas,10)), R2_test_Ridge, 'r', label='Own code Ridge')
plt.plot(int(np.log(lambdas,10)), mean_scikit_r2_cv_Ridge, 'b', label='Scikit Ridge')
plt.plot(int(np.log(lambdas,10)), 'g', label='Own code Lasso')
plt.plot(int(np.log(lambdas,10)), mean_scikit_r2_cv_Lasso, 'p', label='Scikit Lasso')
plt.legend()
plt.show()

plt.figure()
plt.xlabel('model complexity')
plt.ylabel('R2 score')
plt.plot(range(1,polydegree + 1), R2_test, 'r', label='Own code')
plt.plot(range(1,polydegree + 1), mean_scikit_r2_cv, 'b', label='Scikit')
plt.legend()
plt.show()
