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

def GridSearch_LinReg_epochs_batchsize(betas, X, y, num_batches, n_epochs, solution_dict, plot_grid=True):
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

    optimization = solution_dict.get('optimization')
    approach = solution_dict.get('approach')
    iteration_method = solution_dict.get('iteration')

    mse_values = np.zeros((len(num_batches), len(n_epochs)))
    r2_values = np.zeros((len(num_batches), len(n_epochs)))


    for i, M in enumerate(num_batches):
        for j, epochs in enumerate(n_epochs):
            print(f"Computing num. batches={int(M)} and num. epochs={epochs}.")

            y_pred = np.dot(X, betas[j])

            mse = (1.0/len(y)) * np.sum((y - y_pred)**2)

            mean_true_y = np.mean(y)
            R2 = 1 - np.sum((y-y_pred)**2) / np.sum((y - mean_true_y)**2)


            mse_values[i][j] = mse
            r2_values[i][j] = R2

    if plot_grid:
        fig, ax = plt.subplots(figsize=(13*cm, 12*cm))
        sns.heatmap(mse_values,annot=True,ax=ax,cmap="viridis",cbar_kws={'label': 'MSE'},
            yticklabels=num_batches, xticklabels=n_epochs)


        ax.set_title(optimization)  # have title on score bar
        ax.set_ylabel("Number of batches")
        ax.set_xlabel("Epochs")
        plt.tight_layout()
        solution = f'{optimization}_{approach}_{iteration_method}'

        plt.savefig(f'figures/gridsearch_linreg_MSE_epoch_batchsize_{solution}.png')

        fig, ax = plt.subplots(figsize=(12 * cm, 12 * cm))
        sns.heatmap(r2_values, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': '$R^2$'},
            yticklabels=num_batches, xticklabels=n_epochs)
        ax.set_title(optimization)   # have title on the scorebar
        ax.set_ylabel("Number of batches")
        ax.set_xlabel("Epochs")
        plt.tight_layout()
        plt.savefig(f"figures/gridsearch_linreg_R2_epoch_batchsize_{solution}.png")

    return mse_values, r2_values, solution