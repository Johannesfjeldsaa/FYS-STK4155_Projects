import numpy as np
from LinRegression import LinRegression
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"""
Make a class that will find min MSE scores (+R2 scores) for wat you want

inspired by Daniel Haas code

må nå skrives om til å ta inn GD class
"""

cm = 1/2.54

def GridSearch_LinReg_epochs_batchsize(p, x, y, learning_rate, batch_sizes, n_epochs, lmbda=0,
                                       regression_method='OLS', approach='approach',
                                       optimization=None, momentum=0.0, k=5,
                                       rho1=0.9, rho2=0.99, plot_grid=True):
    """
    Function for calculating mse and r2 scores using crossvalidation as training method.
    Can retrieve for different paths: OLS or Ridge, analytical or automatic differentiation and
    optimization methods no momentum, momentum, RMSprop, adagrad and adam.

    :param p: the order of polynomial fit for design matrix
    :param x: initail input data
    :param y: initail input data y(x)
    :param learning_rate: initial guess of step size
    :param batch_sizes: an array to test which one is performing better
    :param n_epochs: an array to test what is optimal epoch size
    :param lmbda: lmbda value for ridge regression, if not change lmbda = 0
    :param regression_method: chosen regression method: OLS or Ridge
    :param approach: Choose between analytical and automatic differentiation approach
    :param optimization: method used for optimization:no momentum,momentum,adagrad,RMSprop or adam
    :param momentum: momentum parameter, set to 0 if not changed
    :param k: number of crossvalidation kfolds
    :param rho1: parameter for momentum methods, usually set to 0.9
    :param rho2: parametor for momemntum methods, usually set to 0.99
    :param plot_grid: set to True, plot a heatmap for visualising results
    :return: mse scores and r2 scores for each combination of epoch and batchsize
    """
    mse_values = np.zeros((len(batch_sizes), len(n_epochs)))
    r2_values = np.zeros((len(batch_sizes), len(n_epochs)))

    linreg = LinRegression(p, x, y)

    for i, M in enumerate(batch_sizes):
        for j, epochs in enumerate(n_epochs):
            #print(f"Computing batch size={M} and num. epochs={epochs}.")


            if approach == 'analytical':
                beta = SGD(linreg.X, linreg.y, batch_sizes[i], n_epochs[j], learning_rate,
                           momentum=momentum, tol=10**-8, regression_method=regression_method,
                           lmbd=lmbda, optimization=optimization, rho1=rho1, rho2=rho2)
            elif approach == 'automatic differentiation':
                beta = AD_SGD(linreg.X, linreg.y, batch_sizes[i], n_epochs[i], learning_rate,
                              momentum=momentum, tol=10**-8, regression_method=regression_method,
                              lmbd=lmbda, optimization=optimization, rho1=rho1, rho2=rho2)
            else:
                raise Exception('No valid approach given')

            linreg.beta = beta  # nå forandrer vel beta seg så cross_val blir gjort på den beta som er funnet?

            mse, r2 = linreg.scikit_cross_validation_train_model(k=k, regression_method=regression_method, lmb=lmbda)   #also takes lmb, when Ridge given

            mse_values[i][j] = mse
            r2_values[i][j] = r2

    if plot_grid:
        fig, ax = plt.subplots(figsize=(13*cm, 12*cm))
        sns.heatmap(mse_values,annot=True,ax=ax,cmap="viridis",cbar_kws={'label': 'MSE'},
            yticklabels=batch_sizes, xticklabels=n_epochs)

        #ax.set_title("MSE")  # have title on score bar
        ax.set_ylabel("Batch size")
        ax.set_xlabel("Epochs")
        plt.tight_layout()
        info = "_sol" + regression_method + f"_opt{optimization}_mom{momentum}_eta_0{learning_rate}_approach{approach}_lmbda{lmbda}"

        plt.savefig("figures/gridsearch_linreg_MSE_epoch_batchsize" + info + ".png")

        fig, ax = plt.subplots(figsize=(12 * cm, 12 * cm))
        sns.heatmap(r2_values, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': '$R^2$'},
            yticklabels=batch_sizes, xticklabels=n_epochs)
        #ax.set_title("$R^2$")   # have title on the scorebar
        ax.set_ylabel("Batch size")
        ax.set_xlabel("Epochs")
        plt.tight_layout()
        plt.savefig("figures/gridsearch_linreg_R2_epoch_batchsize" + info + ".png")

    return mse_values, r2_values