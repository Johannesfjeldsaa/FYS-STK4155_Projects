from GD_and_SGD_AD import AD_GD, AD_SGD
from GD_and_SGD_analytical import GD, SGD
from LinRegression import LinRegression
import matplotlib.pyplot as plt

""""
Plotting functions
"""

def plot_SGD_pred_model_MSE_epoch_batch(X, x, y, dict, approach, regression_method):   #utvides senere til å ta inn Ridge
        for key, value in dict.items():
            beta, _ = SGD(X, y, value[0], value[1], 0.1, 0.3,
                            10**-8, regression_method=regression_method, optimization=key)
            y_pred = X@beta

            plt.plot(sorted(x), sorted(y_pred), label=key)

        info = f'Predicted model from {approach} SGD analysis for {regression_method}'

        plt.plot(x[::15],y[::15],'ro')
        plt.title(info)
        plt.xlabel('x value')
        plt.ylabel('y value')
        plt.legend()
        plt.show()


def plot_SGD_MSE_convergence_epoch_batch(X, y, dict, approach, regression_method):  # utvides senere til å ta inn Ridge
    for key, value in dict.items():
        _, MSE = SGD(X, y, value[0], value[1], 0.1, 0.3, 10**-8,
                     regression_method=regression_method, optimization=key)

        plt.plot(range(len(MSE))[::1000], MSE[::1000], label=key)  # prøvde på en konvergensgraf

    info = f'Convergencegraph for {approach} SGD analysis for {regression_method}'

    plt.title(info)
    plt.xlabel('Iteration number')
    plt.ylabel('MSE scores')
    plt.axis([0,2000, 0, 50])
    plt.legend()
    plt.show()


