import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from setup import save_fig


def signif(x, p):
    """
    Rounds numbers to specified number of significant digits.
    (Copied from https://stackoverflow.com/questions/18915378/rounding-to-significant-figures-in-numpy/59888924#59888924)
     Parameters:
    x (float or list): The number or numbers to round
    p (int): number of significant digits to round to
    """
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

class Plotting:
    
    """
    A class which contains functions for plotting MSE and R2 scores against polynomial order. It also
    contains a class for plotting betaparameters associated with a certain polynomial order.
    """

    def __init__(self, poly_degrees, MSE_scores, R2_scores, beta_parameters, 
                 MSE_training=None):

        if poly_degrees == 0:
            raise ValueError('This regression does not contain any predictors')

        self.poly_degrees = poly_degrees
        self.MSE_scores = MSE_scores
        self.MSE_training = MSE_training
        self.R2_scores = R2_scores
        self.beta_parameters = beta_parameters
        #self.number_of_degrees = len(range(self.poly_degree))

    def plot_MSE_scores(self, la=None, save_filename=None):
        """
        Function plotting MSE_scores against polynomial order
        Return: plot
        """
        fig, ax = plt.subplots()

        ax.set_xlabel('Polynomial degree')
        ax.set_ylabel('Predicted mean square error')

        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.xaxis.set_ticks(self.poly_degrees)
        
        if la is None:
            ax.plot(self.poly_degrees, self.MSE_scores,
                    alpha=0.7, lw=2, color='r',label='MSE score')
        else:
            ax.plot(self.poly_degrees, self.MSE_scores.loc[:, la],
                    alpha=0.7, lw=2, color='r',label='MSE score')
        
        if save_filename is not None:
            save_fig(save_filename)
        
        plt.show()
        
    def plot_MSE_test_and_training(self, la=None, save_filename=None):
        """
        Function plotting MSE_scores against polynomial order
        Return: plot
        """
        fig, ax = plt.subplots()
        plt.text(5, 0.035, 'High bias \nLow variance \n <-----', weight='bold',fontsize=10)
        plt.text(30, 0.035, 'Low bias \nHigh variance \n ----->',weight='bold', fontsize=10)
        ax.set_xlabel('Polynomial degree')
        ax.set_ylabel('Mean Square Error (MSE)')

        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.xaxis.set_ticks(self.poly_degrees[::4])


        if la is None:
            ax.plot(self.poly_degrees, self.MSE_scores,
                    alpha=0.7, lw=2, color='r',label='Test')
            ax.plot(self.poly_degrees, self.MSE_training,
                    alpha=0.7, lw=2, color='b',label='Training')


        else:
            ax.plot(self.poly_degrees, self.MSE_scores.loc[:, la],
                    alpha=0.7, lw=2, color='r',label='MSE score')
        plt.legend(loc='center right')
        
        if save_filename is not None:
            save_fig(save_filename)
        
        plt.show()

    def plot_R2_scores(self, la=None, save_filename=None):
        """
        Function for plotting R2 scores against polynomial order
        :return: plot
        """
        fig, ax = plt.subplots()

        ax.set_xlabel('Polynomial degree')
        ax.set_ylabel('Predicted R squared score')

        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.xaxis.set_ticks(self.poly_degrees)
        
        if la is None:
            ax.plot(self.poly_degrees, self.R2_scores, 
                    alpha=0.7, lw=2, color='b', label='R2 score')
        else:
            ax.plot(self.poly_degrees, self.R2_scores.loc[:, la], 
                    alpha=0.7, lw=2, color='b', label='R2 score')
            
        if save_filename is not None:
            save_fig(save_filename)
            
        plt.show()

    def plot_betaparams_polynomial_order(self, save_filename=None):
        """
        Function for plotting betaparams of the different polynomial orders
        :return: plot
        """
        fig, ax = plt.subplots()

        ax.set_xlabel('Beta parameter number')
        ax.set_ylabel('Beta parameters')
        
        for poly_degree in self.poly_degrees:
            
            ax.plot(self.beta_parameters.columns, 
                    self.beta_parameters.loc[poly_degree, :],
                    label=f'Polynomial degree {poly_degree}')

        plt.legend()
        plt.tight_layout()
        
        if save_filename is not None:
            save_fig(save_filename)
        
        plt.show()
        
    def plot_MSE_for_all_lambdas(self, poly_degree, save_filename=None):
        """
        Function for plotting MSE scores for all lambdas for one
        chosen polynomial degree, for either ridge or lasso regression results.

        Parameters
        ----------
        poly_degree : TYPE
            DESCRIPTION.

        """
        fig, ax = plt.subplots()
        
        ax.plot(self.MSE_scores.columns, self.MSE_scores.loc[poly_degree, :],
                 alpha=0.7, lw=2, color='r', label='MSE')
        
        ax.set_ylabel("Mean Squared Error")
        ax.set_xlabel("Lambda")

        ax.set_xscale('log')
        
        plt.tight_layout()
        
        if save_filename is not None:
            save_fig(save_filename)
        
        
        plt.show()
    
    def plot_R2_for_all_lambdas(self, poly_degree, save_filename=None):
        """
        Function for plotting R2 scores for all lambdas for one
        chosen polynomial degree, for either ridge or lasso regression results.

        Parameters
        ----------
        poly_degree : TYPE
            DESCRIPTION.

        """
        fig, ax = plt.subplots()
        
        ax.plot(self.R2_scores.columns, self.R2_scores.loc[poly_degree, :],
                 alpha=0.7, lw=2, color='r', label='MSE')
        
        ax.set_ylabel("Mean Squared Error")
        ax.set_xlabel("Lambda")

        ax.set_xscale('log')
        
        plt.tight_layout()
        
        if save_filename is not None:
            save_fig(save_filename)
        
        
        plt.show()
        
    def plot_MSE_some_lambdas(self, lambdas_to_plot, num_plot_columns=3, save_filename=None):
        """
        Function for plotting the Mean Squared Error as a function of polynomial
        degree for specified lambda values.

        """
        num_plot_rows = int(np.ceil(len(lambdas_to_plot)/num_plot_columns))
        
        fig, axes = plt.subplots(num_plot_rows, num_plot_columns,
                                 figsize=(12, 7), sharey=True, sharex=True)
                                 #layout="constrained")
        

        for la, ax in zip(lambdas_to_plot, axes.ravel()):
            #la_rounded = 
            title = fr"$\lambda$ = {signif(la, 3)}"
            ax.set_title(title)

            ax.plot(self.MSE_scores.index, self.MSE_scores.loc[:, la], 
                    alpha=0.7, lw=2, color='r', label='MSE')
            
            plt.xticks(self.poly_degrees)
            
        fig.supxlabel("Polynomial degree")
        fig.supylabel("Mean Squared Error (MSE)")
        
        if save_filename is not None:
            save_fig(save_filename)
        
        
        
    def plot_R2_some_lambdas(self, lambdas_to_plot, num_plot_columns=3, save_filename=None):
        """
        Function for plotting the R squared as a function of polynomial
        degree for specified lambda values.

        """
        num_plot_rows = int(np.ceil(len(lambdas_to_plot)/num_plot_columns))
        
        fig, axes = plt.subplots(num_plot_rows, num_plot_columns,
                                 figsize=(12, 7), sharey=True, sharex=True,
                                 layout="constrained")
        

        for la, ax in zip(lambdas_to_plot, axes.ravel()):
            #la_rounded = 
            title = fr"$\lambda$ = {signif(la, 3)}"
            ax.set_title(title)

            ax.plot(self.R2_scores.index, self.R2_scores.loc[:, la], 
                    alpha=0.7, lw=2, color='r', label='MSE')
            
            plt.xticks(self.poly_degrees)
            
        fig.supxlabel("Polynomial degree")
        fig.supylabel("Mean Squared Error (MSE)")
        
        if save_filename is not None:
            save_fig(save_filename)
        
        
    
    
