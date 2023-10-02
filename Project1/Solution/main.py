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


# %%
    print('\n #### Task e) #### \n')

    np.random.seed(2500)
    
    N = 500
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))
    
    # With noise:
    z = FrankeFunction(x, y) + np.random.normal(0, 0.2, x.shape)

    # First reproduce figure similar figure 2.11 in hastie
    polynomal_orders = [i for i in range(1, 5)]
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

    # Make plot to show bias variance analysis of only OLS regression, making own function for
    # bootstrapping and cross validation

    n_boostraps = 100
    k = 5
    max_polydegree = 11
    polynomal_orders = [degree for degree in range(1,max_polydegree)]

    def run_task_e_f(regression_method, polynomal_orders, x, y, z, n_boostraps=None, k_folds=None,
                     lambda_values=None):

        if n_boostraps is not None:

            # Make dataframe with results
            bootstrap_df = pd.DataFrame(index=polynomal_orders)

            # lists to store plotting data
            error_liste = np.zeros(len(polynomal_orders))
            bias_liste = np.zeros(len(polynomal_orders))
            variance_liste = np.zeros(len(polynomal_orders))
            polydegree_liste = np.zeros(len(polynomal_orders))

            for polyorder in polynomal_orders:
                bootstrap_class = LinRegression(polyorder, x, y, z)  # create class
                bootstrap_class.split_data(1 / 5)  # perform split of data

                y_pred, error, bias, variance = bootstrap_class.bootstrapping_train_model(n_boostraps)

                bootstrap_df.loc[polyorder, 'Bias'] = bias
                bootstrap_df.loc[polyorder, 'Variance'] = variance
                bootstrap_df.loc[polyorder, 'Error'] = error
                bootstrap_df.loc[polyorder, 'Bias + variance'] = bias + variance
                if error >= bias + variance:
                    bootstrap_df.loc[polyorder, 'Error >= Bias+Variance'] = True
                else:
                    bootstrap_df.loc[polyorder, 'Error >= Bias+Variance'] = False

                polydegree_liste[polyorder-1] = polyorder
                error_liste[polyorder-1] = error
                bias_liste[polyorder-1] = bias
                variance_liste[polyorder-1] = variance

            return bootstrap_df, polydegree_liste, error_liste, bias_liste, variance_liste

        # cross validation
        elif k_folds and regression_method is not None:

            # Dataframes to store in own code
            MSE_test_df = pd.DataFrame(index=polynomal_orders)
            R2_test_df = pd.DataFrame(index=polynomal_orders)
            #optimal_beta_df = pd.DataFrame(index=polynomal_orders)

            #Scikit own dataframes to store inf
            scikit_MSE_test_df = pd.DataFrame(index=polynomal_orders)
            scikit_r2_test_df = pd.DataFrame(index=polynomal_orders)

            # need to find optimal beta for ridge and lasso
            summary_df = pd.DataFrame(index=polynomal_orders)

            l = int((max(polynomal_orders) + 1) * (max(polynomal_orders) + 2) / 2)


            for polyorder in polynomal_orders:

                cross_validation_class = LinRegression(polyorder, x, y, z)

                if lambda_values is None:  # OLS

                    mean_B_parameters, mean_MSE_test, mean_MSE_train, mean_R2_test, mean_R2_train \
                        = cross_validation_class.cross_validation_train_model(k, regression_method)

                    #optimal_beta_df.loc[polyorder] = list(itertools.chain(mean_B_parameters,
                     #                                                     [np.nan for _ in range(l - len(mean_B_parameters))]))

                    MSE_test_df.loc[polyorder, 'OLS'] = mean_MSE_test
                    R2_test_df.loc[polyorder, 'OLS'] = mean_R2_test

                    scikit_MSE_test_df.loc[polyorder, 'OLS'] = cross_validation_class.scikit_cross_validation_train_model(k, regression_method=regression_method)[0]
                    scikit_r2_test_df.loc[polyorder, 'OLS'] = cross_validation_class.scikit_cross_validation_train_model(k, regression_method=regression_method)[1]


                else: # Ridge and Lasso

                    for la in lambda_values:
                        mean_B_parameters, mean_MSE_test, mean_MSE_train, mean_R2_test, mean_R2_train \
                            = cross_validation_class.cross_validation_train_model(k, regression_method, lmb=la)

                        #optimal_beta_df.loc[polyorder, la] = list(itertools.chain(mean_B_parameters,[np.nan for _ in range(l - len(mean_B_parameters))]))

                        MSE_test_df.loc[polyorder, la] = mean_MSE_test
                        R2_test_df.loc[polyorder, la] = mean_R2_test

                        scikit_MSE_test_df.loc[polyorder, la] = \
                        cross_validation_class.scikit_cross_validation_train_model(k,
                                                                                   regression_method=regression_method, lmb=la)[
                            0]
                        scikit_r2_test_df.loc[polyorder, la] = \
                        cross_validation_class.scikit_cross_validation_train_model(k,
                                                                                   regression_method=regression_method, lmb=la)[
                            1]


                    # lag summary
                    optimal_la_MSE = MSE_test_df.loc[polyorder].idxmin()
                    summary_df.loc[polyorder, 'Optimal lambda MSE'] = optimal_la_MSE
                    summary_df.loc[polyorder, 'Min test MSE'] = MSE_test_df.loc[polyorder, optimal_la_MSE]
                    optimal_la_R2 = R2_test_df.loc[polyorder].idxmax()
                    summary_df.loc[polyorder, 'Optimal lambda R2'] = optimal_la_R2
                    summary_df.loc[polyorder, 'Max test R2'] = R2_test_df.loc[polyorder, optimal_la_R2]

            return MSE_test_df, R2_test_df, scikit_MSE_test_df , scikit_r2_test_df, summary_df

        else:
            raise ValueError('Valid resampling method not entered')


    np.random.seed(2500)

    N = 500
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    # With noise:
    z = FrankeFunction(x, y) + np.random.normal(0, 0.2, x.shape)
    bootstrap_df,polydegree_liste, error_liste, bias_liste, variance_liste  = \
        run_task_e_f(regression_method='OLS', polynomal_orders=polynomal_orders, x=x, y=y, z=z, n_boostraps=n_boostraps)

    print(bootstrap_df)

    # Plot the bias variance analysis (e)

    plt.figure()
    plt.xlabel('Model complexity')
    plt.ylabel('Error score')
    plt.plot(polydegree_liste, error_liste, label='Error')
    plt.plot(polydegree_liste, bias_liste, label='bias')
    plt.plot(polydegree_liste, variance_liste, label='Variance')
    plt.legend()
    plt.show()

# %%
    print('\n #### Task f) #### \n')

    # Lambdas for run through with ridge and lasso
    nlambdas = 100
    lambdas = np.logspace(-5, 1, nlambdas)

    # Hente ut tat for å plotte Ridge
    MSE_test_df_Ridge, R2_test_df_Ridge, scikit_MSE_test_df_Ridge, scikit_r2_test_df_Ridge, summary_df_Ridge = run_task_e_f(
        regression_method='Ridge', polynomal_orders=polynomal_orders, x=x, y=y, z=z, k_folds=5, lambda_values=lambdas)

    print('-----------   Ridge --------------')
    print(MSE_test_df_Ridge)
    print(R2_test_df_Ridge)
    print(scikit_MSE_test_df_Ridge)
    print(scikit_r2_test_df_Ridge)
    print(summary_df_Ridge)  # gives optimal lambda for each model


    MSE_test_df_Lasso, R2_test_df_Lasso, scikit_MSE_test_df_Lasso, scikit_r2_test_df_Lasso, summary_df_Lasso = run_task_e_f(
        regression_method='Lasso', polynomal_orders=polynomal_orders, x=x, y=y, z=z, k_folds=5, lambda_values=lambdas)

    print('-----------   Lasso --------------')
    print(MSE_test_df_Lasso)
    print(R2_test_df_Lasso)
    print(scikit_MSE_test_df_Lasso)
    print(scikit_r2_test_df_Lasso)
    print(summary_df_Lasso)  # gives optimal lambda for each model

    """
    Plot ols, ridge and lasso for given model complexities. for ridge and lasso, the optimal
    lambda value has been found and put into the model. the x axis will be for different k_folds
    5-10
    """

    # For polynomial order of 10, k fold = 5-10


    MSE_OLS = []
    MSE_Ridge = []
    MSE_Lasso = []

    for k in range(5, 11):

        MSE_test_df, R2_test_df, scikit_MSE_test_df, scikit_r2_test_df, summary_df = run_task_e_f(
            regression_method='OLS', polynomal_orders=[10], x=x, y=y, z=z, k_folds=k)

        MSE_OLS.append(MSE_test_df['OLS'].to_list())

        MSE_test_df_Ridge, R2_test_df_Ridge, scikit_MSE_test_df_Ridge, scikit_r2_test_df_Ridge, summary_df_Ridge = run_task_e_f(
            regression_method='Ridge', polynomal_orders=[10], x=x, y=y, z=z, k_folds=k,
            lambda_values=[0.000040])  # fant denne gjennom å søke etter optimal lambda

        MSE_Ridge.append(MSE_test_df_Ridge[0.000040].to_list())

        MSE_test_df_Lasso, R2_test_df_Lasso, scikit_MSE_test_df_Lasso, scikit_r2_test_df_Lasso, summary_df_Lasso = run_task_e_f(
            regression_method='Lasso', polynomal_orders=[10], x=x, y=y, z=z, k_folds=k,
            lambda_values=[0.000040])  # hadde ikke tid til å kjøre for lasso, så fant ikke optimal lambda

        MSE_Lasso.append(MSE_test_df_Lasso[0.000040].to_list())

    plt.figure()
    plt.title('Cross validation for polynomial order 10')
    plt.xlabel('number of k folds')
    plt.ylabel('MSE score')
    plt.plot(range(5,11), MSE_OLS, 'r', label='OLS')
    plt.plot(range(5,11), MSE_Ridge, 'b', label='Ridge, lmb = 0.00004')
    plt.plot(range(5,11), MSE_Lasso, 'g', label='Lasso, lmb = 0.00004')  #finnes kanskje mer opt lambda, hadde ikke tid til å finne den
    plt.legend()
    plt.show()


    """
    Plot for model complexity on x axis, MSE on y_axis. bootstrap, cross validation
    and own code is included as lines. FIrst for OLS and k = 5,8 and 10
    """

    max_polydegree = 11
    polynomal_orders = [degree for degree in range(1,max_polydegree)]

    # Hente ut data for å plotte OLS
    MSE_test_df, R2_test_df, scikit_MSE_test_df, scikit_r2_test_df, summary_df = run_task_e_f(
        regression_method='OLS', polynomal_orders=polynomal_orders, x=x, y=y, z=z, k_folds=5)

    plt.figure()
    plt.title('kfold of 5, OLS')
    plt.xlabel('Model complexity')
    plt.ylabel('Mean Square Error (MSE)')
    plt.plot(polydegree_liste, error_liste, 'r', label='Bootstrapping')
    plt.plot(polydegree_liste, MSE_test_df, 'b', label='Own code cross validation')  # mse test was run on k = 5
    plt.plot(polydegree_liste, scikit_MSE_test_df, 'g', label='Scikit cross validation')  # mse scikit was run on k = 5
    plt.legend()
    plt.show()

    # Hente ut data for å plotte OLS
    MSE_test_df, R2_test_df, scikit_MSE_test_df, scikit_r2_test_df, summary_df = run_task_e_f(
        regression_method='OLS', polynomal_orders=polynomal_orders, x=x, y=y, z=z, k_folds=8)

    plt.figure()
    plt.title('kfold of 8, OLS')
    plt.xlabel('Model complexity')
    plt.ylabel('Mean Square Error (MSE)')
    plt.plot(polydegree_liste, error_liste, 'r', label='Bootstrapping')
    plt.plot(polydegree_liste, MSE_test_df, 'b',
             label='Own code cross validation')  # mse test was run on k = 5
    plt.plot(polydegree_liste, scikit_MSE_test_df, 'g',
             label='Scikit cross validation')  # mse scikit was run on k = 5
    plt.legend()
    plt.show()

    # Hente ut data for å plotte OLS
    MSE_test_df, R2_test_df, scikit_MSE_test_df, scikit_r2_test_df, summary_df = run_task_e_f(
        regression_method='OLS', polynomal_orders=polynomal_orders, x=x, y=y, z=z, k_folds=10)

    plt.figure()
    plt.title('kfold of 10, OLS')
    plt.xlabel('Model complexity')
    plt.ylabel('Mean Square Error (MSE)')
    plt.plot(polydegree_liste, error_liste, 'r', label='Bootstrapping')
    plt.plot(polydegree_liste, MSE_test_df, 'b',
             label='Own code cross validation')  # mse test was run on k = 5
    plt.plot(polydegree_liste, scikit_MSE_test_df, 'g',
             label='Scikit cross validation')  # mse scikit was run on k = 5
    plt.legend()
    plt.show()








