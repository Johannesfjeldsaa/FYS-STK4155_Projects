#%%
from Franke_function import FrankeFunction
from LinRegression import LinRegression
from Plotting import Plotting
from setup import save_fig, data_path

import numpy as np
import pandas as pd
import math
import itertools
import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')



if __name__ == '__main__':

    def run_experiment_a_c(regression_method, scaling_method, polynomal_orders, x, y, z=None, lambda_values=None):
        """

        :param regression_method:
        :param polynomal_orders: list
        :param x:
        :param y:
        :param z:
        :param lambda_values: list
        :return:
        """

        MSE_test_df = pd.DataFrame(index=polynomal_orders)
        MSE_train_df = pd.DataFrame(index=polynomal_orders)

        R2_test_df = pd.DataFrame(index=polynomal_orders)
        R2_train_df = pd.DataFrame(index=polynomal_orders)

        # Number of elements in beta with higest order polynomial
        l = int((max(polynomal_orders)+1)*(max(polynomal_orders)+2)/2)

        beta_parameters_df = pd.DataFrame(index=polynomal_orders, columns=range(l))

        summary_df = pd.DataFrame(index=polynomal_orders)

        for polyorder in polynomal_orders:
            LinReg = LinRegression(polyorder, x, y, z)  # create class
            LinReg.split_data(1 / 5)  # perform split of data

            LinReg.scale(scaling_method=scaling_method)  # Scaling data

            if lambda_values is None:  # OLS
                LinReg.train_model(train_on_scaled=True, regression_method=regression_method)
                LinReg.predict_training()
                LinReg.predict_test()

                MSE_test_df.loc[polyorder, 'OLS'] = LinReg.MSE(LinReg.y_test, LinReg.y_pred_test)
                MSE_train_df.loc[polyorder, 'OLS'] = LinReg.MSE(LinReg.y_train, LinReg.y_pred_train)
                R2_test_df.loc[polyorder, 'OLS'] = LinReg.R_squared(LinReg.y_test, LinReg.y_pred_test)
                R2_train_df.loc[polyorder, 'OLS'] = LinReg.R_squared(LinReg.y_train, LinReg.y_pred_train)
                beta_parameters_df.loc[polyorder] = list(itertools.chain(LinReg.beta, [np.nan for _ in range(l - len(LinReg.beta))]))

            else:  ## Ridge and Lasso

                beta_parameters_list = []
                for la in lambda_values:
                    LinReg.train_model(train_on_scaled=True,
                                       regression_method=regression_method,
                                       la=la)
                    LinReg.predict_training()
                    LinReg.predict_test()

                    MSE_test_df.loc[polyorder, la] = LinReg.MSE(LinReg.y_test, LinReg.y_pred_test)
                    MSE_train_df.loc[polyorder, la] = LinReg.MSE(LinReg.y_train, LinReg.y_pred_train)
                    R2_test_df.loc[polyorder, la] = LinReg.R_squared(LinReg.y_test, LinReg.y_pred_test)
                    R2_train_df.loc[polyorder, la] = LinReg.R_squared(LinReg.y_train, LinReg.y_pred_train)

                    beta_parameters_list.append(LinReg.beta)

                # lag summary
                optimal_la_MSE = MSE_test_df.loc[polyorder].idxmin()
                summary_df.loc[polyorder, 'Optimal lambda MSE'] = optimal_la_MSE
                summary_df.loc[polyorder, 'Min test MSE'] = MSE_test_df.loc[polyorder, optimal_la_MSE]
                optimal_la_R2 = R2_test_df.loc[polyorder].idxmax()
                summary_df.loc[polyorder, 'Optimal lambda R2'] = optimal_la_R2
                summary_df.loc[polyorder, 'Max test R2'] = R2_test_df.loc[polyorder, optimal_la_R2]

                # velg ut tilhørende betaparameter
                indx = MSE_test_df.columns.get_loc(optimal_la_MSE)
                optimal_betas = beta_parameters_list[indx]
                beta_parameters_df.loc[polyorder] = list(itertools.chain(optimal_betas, [np.nan for _ in range(l - len(optimal_betas))]))

                # legg til i beta_parameters_df
        return MSE_train_df, MSE_test_df, R2_train_df, R2_test_df, beta_parameters_df, summary_df


#%%
    # Set seed and generate random data used for a)-c)
    np.random.seed(2500)

    N = 1000
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    z = FrankeFunction(x, y)
#%%
    # Solution to exercise a)

    print('\n ###### Task A) \n')
    polynomal_orders = [1, 2, 3, 4, 5]
    (MSE_train_df,
     MSE_test_df,
     R2_train_df,
     R2_test_df,
     beta_parameters_df,
     summary_df) = run_experiment_a_c(regression_method = 'OLS',
                                      scaling_method='StandardScaling',
                                             polynomal_orders=polynomal_orders,
                                             x=x,
                                             y=y,
                                             z=z)

    print(MSE_test_df)
    print(R2_test_df)
    print(beta_parameters_df)
    # Plotting results of task 1a OLS regression
    plots_task_1a = Plotting(max(polynomal_orders),MSE_test_df['OLS'], R2_test_df['OLS'], None)
    plots_task_1a.plot_MSE_scores()
    plots_task_1a.plot_R2_scores()
    #plots_task_1a.plot_betaparams_polynomial_order()

# %%
    print('\n #### Task b) #### \n')
    nlambdas = 100
    lambdas = np.logspace(-5, 1, nlambdas)

    polynomal_orders = [1, 2, 3, 4, 5]

    (MSE_train_df,
     MSE_test_df,
     R2_train_df,
     R2_test_df,
     beta_parameters_df,
     summary_df) = run_experiment_a_c(regression_method='Ridge',
                                      scaling_method='StandardScaling',
                                      polynomal_orders=polynomal_orders,
                                      lambda_values=lambdas,
                                      x=x,
                                      y=y,
                                      z=z)
    print(MSE_test_df)
    print(R2_test_df)
    print(beta_parameters_df)
    print(summary_df)

#%%
    print('\n #### Task c) #### \n')

    nlambdas = 100
    lambdas = np.logspace(-5, 1, nlambdas)

    polynomal_orders = [1, 2, 3, 4, 5]

    (MSE_train_df,
     MSE_test_df,
     R2_train_df,
     R2_test_df,
     beta_parameters_df,
     summary_df) = run_experiment_a_c(regression_method='Lasso',
                                      scaling_method='StandardScaling',
                                      polynomal_orders=polynomal_orders,
                                      lambda_values=lambdas,
                                      x=x,
                                      y=y,
                                      z=z)
    print(MSE_test_df)
    print(R2_test_df)
    print(beta_parameters_df)
    print(summary_df)
