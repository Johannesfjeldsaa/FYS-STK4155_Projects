import numpy as np
from LinRegression import LinRegression
from Franke_function import FrankeFunction
import matplotlib.pyplot as plt
import random

np.random.seed(100)

polydegree = 13
x = np.linspace(0,1,100)
y = np.linspace(0,1,100)
z = FrankeFunction(x,y)
n = len(x)
k = 10

mean_B_parameters_each_model = []
mean_MSE_test = []
mean_MSE_train = []
mean_R2_test = []
mean_R2_train = []

for polydegree in range(polydegree+1):

    if polydegree != 0:

        cross_validation = LinRegression(polydegree, x, y, z)
        k_groups = cross_validation.create_kfold_groups(k)
        test_matrices_iterations, train_matrices_iterations, y_cv_test, y_cv_train = cross_validation.create_list_cross_validation_analysis()

        # storing for each k iteration

        opt_beta = []
        MSE_test = []
        MSE_train = []
        R2_test = []
        R2_train = []

        for test_matrix, train_matrix, y_test, y_train in \
                zip(test_matrices_iterations, train_matrices_iterations, y_cv_test, y_cv_train):

            cross_validation.train_model(regression_method='OLS', cv_X=train_matrix, cv_y=y_train)
            opt_beta.append(cross_validation.beta)   # store optimal betas for each cross validation set

            # find for training set: X_training @ beta
            y_train_pred = train_matrix @ cross_validation.beta  #gives self.y_pred
            MSE_train.append(cross_validation.MSE(y_train, y_train_pred))  # store MSE score for each train set
            R2_train.append(cross_validation.R_squared(y_train, y_train_pred))

            # find for test set: test_matrix @ beta
            y_test_pred = test_matrix @ cross_validation.beta
            MSE_test.append(cross_validation.MSE(y_test, y_test_pred))  # store MSE score for each test set
            R2_test.append(cross_validation.R_squared(y_test, y_test_pred))

        mean_MSE_test.append(np.mean(MSE_test))
        mean_MSE_train.append(np.mean(MSE_train))
        mean_R2_test.append(np.mean(R2_test))
        mean_R2_train.append(np.mean(R2_train))

        B_matrix = np.array(opt_beta)

        opt_beta_model = []
        for i in range(len(B_matrix[0,:])):
            column = B_matrix[:,i]
            opt_beta_model.append(np.mean(column))

        mean_B_parameters_each_model.append(opt_beta_model)  # can be used to find the optimal b parameters for each model comlpexities ("optimal model")

fig, ax1 = plt.subplots(figsize=(9, 6))
ax2 = ax1.twinx()
plt.xlabel('model complexity')
ax1.set_ylabel('Mean MSE from cross validation')
ax1.plot(range(1,polydegree+1), mean_MSE_train, 'b', label='MSE train')
ax1.plot(range(1,polydegree+1), mean_MSE_test, 'g', label='MSE test')
ax1.legend(loc='lower right')
ax2.set_ylabel('Mean R2 value cross validation')
ax2.plot(range(1,polydegree+1), mean_R2_train, 'purple', label='R2 train')
ax2.plot(range(1,polydegree+1), mean_R2_test, 'r', label='R2 test')
ax2.legend(loc='center right')
plt.show()

