import numpy as np
from LinRegression import LinRegression
import matplotlib.pyplot as plt
import seaborn as sns
from GD_and_SGD_analytical import GD, SGD
from GD_and_SGD_AD import AD_GD, AD_SGD


"""
Make a class that will find min MSE scores (+R2 scores) for wat you want

inspired by Daniel Haas code
"""

cm = 1/2.54

## må utvides så vi kan kjøre med flere metoder.
def GridSearch_LinReg_epochs_batchsize(p, x, y, learning_rate, batch_sizes, n_epochs, lmbda=0,
                                       regression_method='OLS', approach='approach',
                                       optimization=None, momentum=0.0, k=5,
                                       rho1=None, rho2=None, plot_grid=True):
    mse_values = np.zeros((len(batch_sizes), len(n_epochs)))
    r2_values = np.zeros((len(batch_sizes), len(n_epochs)))

    for i, M in enumerate(batch_sizes):
        for j, epochs in enumerate(n_epochs):
            print(f"Computing batch size={M} and num. epochs={epochs}.")
            linreg = LinRegression(p, x, y)
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