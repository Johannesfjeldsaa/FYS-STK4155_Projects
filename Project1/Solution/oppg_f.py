import numpy as np
from LinRegression import LinRegression
from Franke_function import FrankeFunction
import matplotlib.pyplot as plt
import random


polydegree = 14
x = np.linspace(0,1,100)
y = np.linspace(0,1,100)
z = FrankeFunction(x,y)
k = 5
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
            = cross_validation.cross_validation_train_model(k)

        optimal_beta_cross_validation.append(mean_B_parameters)
        MSE_test.append(mean_MSE_test)
        MSE_train.append(mean_MSE_train)
        R2_test.append(mean_R2_test)
        R2_train.append(mean_R2_train)

"""
using scikit_learn library
"""

mean_scikit_MSE_cv = []  # antar dette er test mse?

for polydegree in range(polydegree+1):

    if polydegree != 0:

        scikit_cv = LinRegression(polydegree,x,y,z)
        mean_scikit_MSE_cv.append(scikit_cv.scikit_cross_validation_train_model(k))

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

print(optimal_beta_cross_validation)