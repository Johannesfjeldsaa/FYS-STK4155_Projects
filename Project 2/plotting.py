import numpy as np
import matplotlib.pyplot as plt
import math
from prepropeces import Handle_Files

""""
Plotting functions
"""

def plot_SGD_MSE_convergence_epoch_batch(optimization_methods, cost_scores):

    for i, optimization in enumerate(optimization_methods):
        plt.plot(range(len(cost_scores[i]))[::500], cost_scores[i][::500], label=optimization)

    plt.title('Convergencegraph')
    plt.xlabel('Iteration number')
    plt.ylabel('MSE scores')
    plt.axis(([0, 1000, 0, 3]))
    plt.legend()
    plt.show()


class Plott_creator:
    def __init__(self):
        self.file_handler = Handle_Files()

    def plot_regression(self, x, y, y_preds, title, save_fig=False, show_history=False):
        def plot_scatter_set_apperence(x, y, title):
            fig, ax = plt.subplots()
            ax.scatter(x, y, c='b', marker='o', label='Data')
            ax.set_xlabel('x value')
            ax.set_ylabel('y value')
            ax.set_title(title)
            return fig, ax

        if show_history:
            if type(y_preds) is not list:
                raise TypeError('y_preds must be a list of lists')
            else:
                fig, ax = plot_scatter_set_apperence(x, y, title)
                for t, y_pred in enumerate(y_preds[:-1]):
                    ax.plot(x, y_pred, 'r-',
                            alpha=t / len(y_preds))
                y_pred = y_preds[-1]
                ax.plot(x, y_pred, 'g-', label='Final Prediction')
        else:
            fig, ax = plot_scatter_set_apperence(x, y, title)
            if type(y_preds) is list:
                y_pred = y_preds[-1]
            else:
                try:
                    y_pred = y_preds
                except:
                    raise TypeError(f"Prediction must be a list of lists or a list, not {type(y_preds)}")

            ax.plot(x, y_pred, 'g-', label='Final prediction')

        plt.legend()
        if save_fig:
            self.file_handler.save_fig(title)

        plt.show()

    def plot_path_weights(self, costfunction, X, y, weights, title, save_fig=False):
        """
        Plot the path of the weights during gradient descent.

        Inspired by: https://github.com/mravanba/comp551-notebooks/blob/master/GradientDescent.ipynb
        :param weights:
        :param title:
        :param save_fig:
        :return:
        """

        def magnitude(x):
            return int(math.log10(x))
        def add_margin_to_lim(lim, upper=True):
            if upper:
                lim += 10 ** magnitude(lim)
            elif upper is False:
                lim -= 10 ** magnitude(lim)
            return lim
        def plot_contour_of_cost(costfunction, X, y, w0_lim, w1_lim):
            # create figure and ax
            fig, ax = plt.subplots()

            # create values with w0_lim on x-axis and w1 for y
            w0_range = np.linspace(w0_lim[0], w0_lim[1], 50)
            w1_range = np.linspace(w1_lim[0], w1_lim[1], 50)
            w0_grid, w1_grid = np.meshgrid(w0_range, w1_range)
            zg = np.zeros_like(w0_grid)

            # use values to create values for the z direction based on the cost function
            for i in range(50):
                for j in range(50):
                    zg[i, j] = costfunction(X, y, np.array([w0_grid[i, j], w1_grid[i, j]]))

            ax.contour(w0_grid, w1_grid, zg, 50)

            return fig, ax

        if type(weights) is not list:
            raise TypeError('weights must be a list of lists or a list, not {}'.format(type(weights)))

        weigths = np.vstack(weights)
        w0_lim = [add_margin_to_lim(weigths[:,0].min(), False),
                  add_margin_to_lim(weigths[:,0].max())]
        w1_lim = [add_margin_to_lim(weigths[:, 1].min(), False),
                  add_margin_to_lim(weigths[:,1].max())]

        # Plot the contour of the cost function
        fig, ax = plot_contour_of_cost(costfunction, X, y, w0_lim, w1_lim)

        # Plot the weights
        ax.plot(weigths[:,0], weigths[:,1], '.r', alpha=.8)
        ax.plot(weigths[:,0], weigths[:,1], '-r', alpha=.3)
        ax.set_xlabel(r'$w_0$')
        ax.set_ylabel(r'$w_1$')
        ax.set_title(title)
        ax.set_xlim(w0_lim[0], w0_lim[1])
        ax.set_ylim(w1_lim[0], w1_lim[1])
        if save_fig:
            self.file_handler.save_fig(title)
        plt.show()



