#%%
from Franke_function import FrankeFunction
from LinRegression import LinRegression
from Plotting import Plotting
from setup import save_fig, data_path

import numpy as np
import pandas as pd
#import math
import itertools
import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')

from imageio import imread
import matplotlib.pyplot as plt
 


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
    
    # Without noise:
    #z = FrankeFunction(x, y)
    
    # With noise:
    z = FrankeFunction(x, y) + np.random.normal(0, 0.1, x.shape)
#%%
    # Solution to exercise a)

    print('\n ###### Task A) \n')
    polynomal_orders = [1, 2, 3, 4, 5]

    (MSE_train_df_OLS,
     MSE_test_df_OLS,
     R2_train_df_OLS,
     R2_test_df_OLS,
     beta_parameters_df_OLS,
     summary_df_OLS) = run_experiment_a_c(regression_method = 'OLS',
                                          scaling_method='StandardScaling',
                                          polynomal_orders=polynomal_orders,
                                          x=x,
                                          y=y,
                                          z=z)

    print(MSE_test_df_OLS)
    print(R2_test_df_OLS)
    print(beta_parameters_df_OLS)
    # Plotting results of task 1a OLS regression
    plots_task_1a = Plotting(polynomal_orders, MSE_test_df_OLS, 
                             R2_test_df_OLS, beta_parameters_df_OLS)
    plots_task_1a.plot_MSE_scores()
    plots_task_1a.plot_R2_scores()
    plots_task_1a.plot_betaparams_polynomial_order()

# %%
    print('\n #### Task b) #### \n')
    nlambdas = 100
    lambdas = np.logspace(-5, 1, nlambdas)

    polynomal_orders = [1, 2, 3, 4, 5]

    (MSE_train_df_ridge,
     MSE_test_df_ridge,
     R2_train_df_ridge,
     R2_test_df_ridge,
     beta_parameters_df_ridge,
     summary_df_ridge) = run_experiment_a_c(regression_method='Ridge',
                                            scaling_method='StandardScaling',
                                            polynomal_orders=polynomal_orders,
                                            lambda_values=lambdas,
                                            x=x,
                                            y=y,
                                            z=z)
    print(MSE_test_df_ridge)
    print(R2_test_df_ridge)
    print(beta_parameters_df_ridge)
    print(summary_df_ridge)
    
    plots_task_1b = Plotting(polynomal_orders, MSE_test_df_ridge, 
                             R2_test_df_ridge, beta_parameters_df_ridge)
    plots_task_1b.plot_MSE_for_all_lambdas(5)
    
    num_lambdas_to_plot = 6
    idx_to_plot = np.round(np.linspace(0, len(lambdas) - 1,
                                       num_lambdas_to_plot)).astype(int)
    
    plots_task_1b.plot_MSE_some_lambdas(lambdas_to_plot=lambdas[idx_to_plot])
    
    plots_task_1b.plot_betaparams_polynomial_order()
    
    
#%%
    print('\n #### Task c) #### \n')

    nlambdas = 100
    lambdas = np.logspace(-5, 1, nlambdas)

    polynomal_orders = [1, 2, 3, 4, 5]

    (MSE_train_df_lasso,
     MSE_test_df_lasso,
     R2_train_df_lasso,
     R2_test_df_lasso,
     beta_parameters_df_lasso,
     summary_df_lasso) = run_experiment_a_c(regression_method='Lasso',
                                            scaling_method='StandardScaling',
                                            polynomal_orders=polynomal_orders,
                                            lambda_values=lambdas,
                                            x=x,
                                            y=y,
                                            z=z)
    print(MSE_test_df_lasso)
    print(R2_test_df_lasso)
    print(beta_parameters_df_lasso)
    print(summary_df_lasso)
    
    plots_task_1c = Plotting(polynomal_orders, MSE_test_df_lasso, 
                             R2_test_df_lasso, beta_parameters_df_lasso)
    plots_task_1c.plot_MSE_for_all_lambdas(5)
    
    num_lambdas_to_plot = 6
    idx_to_plot = np.round(np.linspace(0, len(lambdas) - 1,
                                       num_lambdas_to_plot)).astype(int)
    
    plots_task_1c.plot_MSE_some_lambdas(lambdas_to_plot=lambdas[idx_to_plot])
    
    plots_task_1c.plot_betaparams_polynomial_order()
    
    
#%% Making summary plots for task a) through c)
# Var ikke sikker på hvordan inkludere dette i Plotting klassen ettersom det 
# trengs input fra alle 3 metodene (OLS, Ridge, Lasso)

    plt.figure()
    
    plt.plot(MSE_test_df_OLS.index, MSE_test_df_OLS.OLS, label="OLS")
    plt.plot(summary_df_ridge.index, summary_df_ridge["Min test MSE"],
             "--", label="Ridge")
    plt.plot(summary_df_lasso.index, summary_df_lasso["Min test MSE"],
             "-.", label="Lasso")
    
    plt.xticks(summary_df_ridge.index)
    
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Polynomial degree")
    plt.legend()
    
    plt.show()
    
    plt.figure()
    
    plt.plot(R2_test_df_OLS.index, R2_test_df_OLS.OLS, label="OLS")
    plt.plot(summary_df_ridge.index, summary_df_ridge["Max test R2"],
             "--", label="Ridge")
    plt.plot(summary_df_lasso.index, summary_df_lasso["Max test R2"],
             "-.", label="Lasso")
    
    plt.xticks(summary_df_ridge.index)
    
    plt.ylabel(r"R$^2$")
    plt.xlabel("Polynomial degree")
    plt.legend()
    
    plt.show()

#%% 

    print('\n #### Task e) #### \n')

    np.random.seed(2500)
    
    N = 500
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))
    
    # With noise:
    z = FrankeFunction(x, y) + np.random.normal(0, 0.2, x.shape)
    
    polynomal_orders = [i for i in range(1, 80)]
    (MSE_train_df_OLS,
     MSE_test_df_OLS,
     R2_train_df_OLS,
     R2_test_df_OLS,
     beta_parameters_df_OLS,
     summary_df_OLS) = run_experiment_a_c(regression_method = 'OLS',
                                          scaling_method='StandardScaling',
                                          polynomal_orders=polynomal_orders,
                                          x=x,
                                          y=y,
                                          z=z)
    
    plots_task_1e = Plotting(polynomal_orders, MSE_test_df_OLS, 
                             R2_test_df_OLS, beta_parameters_df_OLS, 
                             MSE_train_df_OLS)
    
    plots_task_1e.plot_MSE_test_and_training()

#%% 

    print('\n #### Task g) #### \n')
    
    # Load the terrain
    terrain = imread(r'DataFiles\SRTM_data_Norway_1.tif')
    
    N = 1000
    m = 5 # polynomial order
    
    #x = np.random.randint(0, N, N)
    #y = np.random.randint(0, N, N)
    
    x = np.arange(0, N, 1)
    y = np.arange(0, N, 1)
    
    x_mesh, y_mesh = np.meshgrid(x, y)
    
    z = terrain[x_mesh.ravel(), y_mesh.ravel()]
    
    #%%
    
    x = np.linspace(0, 1, np.shape(terrain)[0])
    
    #%%
    terrain = terrain[:N,:N]
    # Creates mesh of image pixels
    x = np.linspace(0, 1, np.shape(terrain)[0])
    y = np.linspace(0, 1, np.shape(terrain)[1])
    x_mesh, y_mesh = np.meshgrid(x,y)
    
    z = terrain
    #X = create_X(x_mesh, y_mesh, m)
    
    
    # Show the terrain
    plt.figure()
    plt.title('Terrain over Norway 1')
    plt.imshow(terrain, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
