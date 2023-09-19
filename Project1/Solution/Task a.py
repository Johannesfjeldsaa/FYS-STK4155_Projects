import numpy as np

import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')


from Franke_function import FrankeFunction
from LinRegression import LinRegression
from Plotting import Plotting
from setup import save_fig, data_path

if __name__ == '__main__':

    np.random.seed(2500)

    N = 1000
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    z = FrankeFunction(x, y)

    MSE_test_scores = []
    R2_test_scores = []
    beta_parameters = []
    polydegree = 5

    for polyorder in range(1,polydegree+1):

        OLS_regression = LinRegression(polyorder, x, y, z) #create class
        OLS_regression.split_data(1/5) # perform split of data

        print(f'Split performed: {OLS_regression.splitted}')

        OLS_regression.scale(scaling_method='StandardScaling')
        print(f'Scaling performed: {OLS_regression.scaled}\n'
               f'Scaling methode: {OLS_regression.scaling_method}')

        OLS_regression.train_model(train_on_scaled=True, regression_method='OLS')
        print(f'The optimal parametres are: {OLS_regression.beta}')
        beta_parameters.append(OLS_regression.beta)

        OLS_regression.predict_training()

        OLS_regression.predict_test()

        MSE_training = OLS_regression.MSE(OLS_regression.y_train, OLS_regression.y_pred_train)
        MSE_test = OLS_regression.MSE(OLS_regression.y_test, OLS_regression.y_pred_test)

        MSE_test_scores.append(MSE_test)

        # The mean squared error
        print(f'Mean squared error training: {MSE_training:.4f}')
        print(f'Mean squared error test: {MSE_test:.4f}')

        R2_training = OLS_regression.R_squared(OLS_regression.y_train, OLS_regression.y_pred_train)
        R2_test = OLS_regression.R_squared(OLS_regression.y_test, OLS_regression.y_pred_test)

        R2_test_scores.append(R2_test)
        print(f'R^2 training: {R2_training:.4f}')
        print(f'R^2 test: {R2_test:.4f}')

    # Plotting results of task 1a OLS regression
    plots_task_1a = Plotting(5, MSE_test_scores, R2_test_scores, beta_parameters)
    plots_task_1a.plot_MSE_scores()
    plots_task_1a.plot_R2_scores()
    plots_task_1a.plot_betaparams_polynomial_order()
