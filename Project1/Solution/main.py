from Franke_function import FrankeFunction
from LinRegression import LinRegression
from Plotting import Plotting
from setup import save_fig, data_path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')


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


#%% Task c) Lasso regression

    max_polydegree = 5
    degrees = np.arange(1, max_polydegree+1)
    
    nlambdas = 100
    lambdas = np.logspace(-5, 4, nlambdas)
    
    num_lambdas_to_plot = 7 # Number of lambdas we want to plot
    indices_to_plot = np.round(np.linspace(0, len(lambdas) - 1,
                                           num_lambdas_to_plot)).astype(int)
    
    MSE_test_lasso_df = pd.DataFrame(index=degrees)
    MSE_train_lasso_df = pd.DataFrame(index=degrees)
    
    R2_test_lasso_df = pd.DataFrame(index=degrees)
    R2_train_lasso_df = pd.DataFrame(index=degrees)
    
    lasso_summary = pd.DataFrame(index=degrees)
    
    # Sjekk ut GridSearch
    for polydegree in degrees:
        print(f"Polynomial degree: {polydegree}")
        
        MSE_train_scores_lasso = []
        MSE_test_scores_lasso = []
        
        R2_train_scores_lasso = []
        R2_test_scores_lasso = []
        
        beta_parameters_lasso = []
        
        Lasso_regression = LinRegression(polydegree, x, y, z) #create class
        Lasso_regression.split_data(1/5) # perform split of data
            
        Lasso_regression.scale(scaling_method='StandardScaling') # Scaling data
        
        for i, la in enumerate(lambdas):
            
            Lasso_regression.train_model(train_on_scaled=True, regression_method='Lasso', la=la)
            
            beta_parameters_lasso.append(Lasso_regression.beta)
            
            Lasso_regression.predict_training()

            Lasso_regression.predict_test()

            MSE_training = Lasso_regression.MSE(Lasso_regression.y_train, 
                                                Lasso_regression.y_pred_train)
            MSE_test = Lasso_regression.MSE(Lasso_regression.y_test, 
                                            Lasso_regression.y_pred_test)
            
            MSE_train_scores_lasso.append(MSE_training)
            MSE_test_scores_lasso.append(MSE_test)
            
            
            R2_training = Lasso_regression.R_squared(Lasso_regression.y_train, 
                                                     Lasso_regression.y_pred_train)
            R2_test = Lasso_regression.R_squared(Lasso_regression.y_test, 
                                                 Lasso_regression.y_pred_test)
            
            R2_train_scores_lasso.append(R2_training)
            R2_test_scores_lasso.append(R2_test)

                
            # Getting out equally spaced lambdas to make plots of
            if i in indices_to_plot:
                MSE_train_lasso_df.loc[polydegree, la] = MSE_training
                MSE_test_lasso_df.loc[polydegree, la] = MSE_test
                
                R2_train_lasso_df.loc[polydegree, la] = R2_training
                R2_test_lasso_df.loc[polydegree, la] = R2_test
            
        lasso_summary.loc[polydegree, "Min test MSE"] = np.min(MSE_test_scores_lasso)
        min_index = np.argmin(MSE_test_scores_lasso)
        lasso_summary.loc[polydegree, "Lambda"] = lambdas[min_index]
        lasso_summary.loc[polydegree, "Train MSE"] = MSE_train_scores_lasso[min_index]
        lasso_summary.loc[polydegree, "R2 test"] = R2_test_scores_lasso[min_index]
        lasso_summary.loc[polydegree, "R2 train"] = R2_train_scores_lasso[min_index]
        
        # Uncomment to plot all lambdas for each polynomial degree:
        # plt.figure()
        # plt.title(f"MSE Lasso, Polydegree: {polydegree}")
        # plt.plot(np.log10(lambdas), MSE_train_scores_lasso, 'r--', label = 'Train')
        # plt.plot(np.log10(lambdas), MSE_test_scores_lasso, 'b--', label = 'Test')
        # plt.legend()
        # plt.show()
        
        # plt.figure()
        # plt.title(f"R2 Lasso, Polydegree: {polydegree}")
        # plt.plot(np.log10(lambdas), R2_train_scores_lasso, 'r--', label = 'Train')
        # plt.plot(np.log10(lambdas), R2_test_scores_lasso, 'b--', label = 'Test')
        # plt.legend()
        # plt.show()
    

            
            
            
            
    
    
    