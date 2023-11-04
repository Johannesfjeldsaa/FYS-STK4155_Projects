import numpy as np
from LinRegression import LinRegression
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"""
Gridsearch will find where combination of epochs and batches give best performance.

Heatmaps are created to visualise this.

inspired by Daniel Haas code

"""

cm = 1/2.54

def GridSearch_LinReg_epochs_batchsize(betas, X, y, num_batches, n_epochs, solution_dict, plot_grid=True):
    """

    :param betas: betas found from different optimization methods (outside functions)
    :param X: original design matrix
    :param y: orginal y variables
    :param num_batches: list of number of batches
    :param n_epochs:  list of number of epochs
    :param solution_dict: dictionary cointaining information about analysis
    :param plot_grid: True if heatmap is wanted as visualization
    :return: calculated MSE and R2 values and a solution information
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