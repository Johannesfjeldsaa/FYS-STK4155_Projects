import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

"""
A class which contains functions for plotting MSE and R2 scores against polynomial order. It also
contains a class for plotting betaparameters associated with a certain polynomial order.
"""
class Plotting:

    def __init__(self, poly_degree, MSE_scores, R2_scores, beta_parameters):

        if poly_degree == 0:
            raise ValueError('This regression does not contain any predictors')

        self.poly_degree = poly_degree
        self.MSE_scores = MSE_scores
        self.R2_scores = R2_scores
        self.beta_parameters = beta_parameters
        self.number_of_degrees = len(range(self.poly_degree))

    def plot_MSE_scores(self):
        """
        Function plotting MSE_scores against polynomial order
        Return: plot
        """
        fig, ax = plt.subplots()

        ax.set_xlabel('Polynomial degree')
        ax.set_ylabel('Predicted mean square error')

        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.xaxis.set_ticks(np.arange(1, (self.number_of_degrees + 1), 1))

        ax.plot(np.arange(1,self.number_of_degrees +1), self.MSE_scores,
                alpha=0.7, lw=2, color='r',label='MSE score')
        plt.show()

    def plot_R2_scores(self):
        """
        Function for plotting R2 scores against polynomial order
        :return: plot
        """
        fig, ax = plt.subplots()

        ax.set_xlabel('Polynomial degree')
        ax.set_ylabel('Predicted R squared score')

        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.xaxis.set_ticks(np.arange(1, (self.number_of_degrees + 1), 1))

        ax.plot(np.arange(1, self.number_of_degrees + 1),
                self.R2_scores, alpha=0.7, lw=2, color='b', label='R2 score')

        plt.show()

    def plot_betaparams_polynomial_order(self):
        """
        Function for plotting betaparams of the different polynomial orders
        :return: plot
        """
        fig, ax = plt.subplots()

        ax.set_xlabel('Number of beta parameters')
        ax.set_ylabel('Beta parameters')

        for degree, beta_coefficients in zip(np.arange(1, self.number_of_degrees +1),
                                             self.beta_parameters):
            number_parameters = len(beta_coefficients)
            ax.plot(range(number_parameters), beta_coefficients,
                    label=f'polynomial of order {degree}')
        plt.legend()
        plt.show()


