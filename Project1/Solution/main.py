from Franke_function import FrankeFunction
from LinRegression import LinRegression
from Plotting import Plotting
from setup import save_fig, data_path

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')
import seaborn as sns

if __name__ == '__main__':

    # Set seed and generate random data used for a)-c)
    np.random.seed(2500)

    N = 1000
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    z = FrankeFunction(x, y)


    # Solution to exercise a)

    print('\n EXERCISE a) \n')

    MSE_test_scores = []
    R2_test_scores = []
    beta_parameters = []
    polydegree = 5

    for polyorder in range(1,polydegree+1):

        OLS_regression = LinRegression(polyorder, x, y, z) #create class

        OLS_regression.split_data(1/5) # perform split of data

        OLS_regression.scale(scaling_method='StandardScaling')

        OLS_regression.train_model(train_on_scaled=True, regression_method='OLS')
        beta_parameters.append(OLS_regression.beta)

        OLS_regression.predict_training()

        OLS_regression.predict_test()

        MSE_training = OLS_regression.MSE(OLS_regression.y_train, OLS_regression.y_pred_train)
        MSE_test = OLS_regression.MSE(OLS_regression.y_test, OLS_regression.y_pred_test)

        MSE_test_scores.append(MSE_test)

        R2_training = OLS_regression.R_squared(OLS_regression.y_train, OLS_regression.y_pred_train)
        R2_test = OLS_regression.R_squared(OLS_regression.y_test, OLS_regression.y_pred_test)

        R2_test_scores.append(R2_test)



    # Plotting results of task 1a OLS regression
    plots_task_1a = Plotting(5, MSE_test_scores, R2_test_scores, beta_parameters)
    plots_task_1a.plot_MSE_scores()
    plots_task_1a.plot_R2_scores()
    plots_task_1a.plot_betaparams_polynomial_order()


    # Solution to exercise b)

    print('\n EXERCISE b) \n')

    lambdas = np.logspace(-4, 0, 8)

    MSE_test_scores_ridge = {}
    R2_test_scores_ridge = {}
    beta_parameters_ridge = {}

    for la in lambdas:
        print(la)
        polydegree = 3

        MSE_test_scores_ridge_per_la = []
        R2_test_scores_ridge_per_la = []
        beta_parameters_ridge_per_la = []

        for polyorder in range(1, polydegree + 1):
            ridge_regression = LinRegression(polyorder, x, y, z)  # create class

            ridge_regression.split_data(1 / 5)  # perform split of data

            ridge_regression.scale(scaling_method='StandardScaling')

            ridge_regression.train_model(train_on_scaled=True, regression_method='Ridge')
            beta_parameters_ridge_per_la.append(ridge_regression.beta)

            ridge_regression.predict_training()

            ridge_regression.predict_test()

            MSE_training = ridge_regression.MSE(ridge_regression.y_train, ridge_regression.y_pred_train)
            MSE_test = ridge_regression.MSE(ridge_regression.y_test, ridge_regression.y_pred_test)

            MSE_test_scores_ridge_per_la.append(MSE_test)

            R2_training = ridge_regression.R_squared(ridge_regression.y_train, ridge_regression.y_pred_train)
            R2_test = ridge_regression.R_squared(ridge_regression.y_test, ridge_regression.y_pred_test)

            R2_test_scores_ridge_per_la.append(R2_test)

        beta_parameters_ridge[la] = beta_parameters_ridge_per_la
        MSE_test_scores_ridge[la] = MSE_test_scores_ridge_per_la
        R2_test_scores_ridge[la] = R2_test_scores_ridge_per_la