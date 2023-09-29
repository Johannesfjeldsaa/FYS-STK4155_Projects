import numpy as np
from LinRegression import LinRegression
from Franke_function import FrankeFunction
import matplotlib.pyplot as plt
import random


polydegree = 13
x = np.linspace(0,1,100)
y = np.linspace(0,1,100)
z = FrankeFunction(x,y)
k = 10
seed = 1917
np.random.seed(seed)

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
            = cross_validation.cross_validation_train_model(k, seed)

        optimal_beta_cross_validation.append(mean_B_parameters)
        MSE_test.append(mean_MSE_test)
        MSE_train.append(mean_MSE_train)
        R2_test.append(mean_R2_test)
        R2_train.append(mean_R2_train)



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