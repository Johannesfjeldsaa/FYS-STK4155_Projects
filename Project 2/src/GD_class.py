# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:14:44 2023

@author: vildesn
"""

import jax
import jax.numpy as jnp
import numpy as np
#import matplotlib.pyplot as plt

from jax import grad as jax_grad

class GradientDescent:
    """
    Class for performing gradient descent, either stochastic or normal.
    :param learning_rate: the learning rate for the gradient descent algorithm.
    :param tol: optional, the tolerance for the convergence check. Default: 1e-3
    :param cost_function: cost function used in gradient descent algorithm. 
    Should be a function. Default: None. If None, a simple version of the class
    will be set up, for use of the calculate_change method only.
    :param analytical_gradient: Function for the analytical gradient of the 
    cost function. Default: None. If None, the gradient will be calculated
    using jax.
    :param skip_convergence_check: False if the algorithm should check for 
    converge, True if not. Defaul: False.
    :param record_history: If True, the estimated beta parameters and 
    corresponding cost scores at each iteration are recorded. 
    """

    def __init__(self, learning_rate, tol=1e-3, cost_function=None, 
                 analytic_gradient=None, 
                 skip_convergence_check=False, record_history=False):
        
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        
        if cost_function is not None:
            self.tol = tol
            self.cost_function = cost_function
            self.skip_convergence_check = skip_convergence_check
            self.record_history = record_history
    
            if self.record_history is True:
                self.betas = []
                self.cost_scores = []
            
            if analytic_gradient is not None:
                if callable(analytic_gradient):
                    self.calc_gradient = analytic_gradient
                else:
                    raise ValueError("Analytical gradient must be a function")
                    
            else:
                # When defining the cost function, the third parameter must be 
                # the one to differentiate by
                self.calc_gradient = jax_grad(self.cost_function, 2)
    
    def learning_schedule(self, method):
        """ 
        Adjusts the learning rate during training according to the selected 
        learning schedule and updates the learning rate in-place. 
        :param method: Learning rate schedule method. 
        Options: 'Fixed learning rate' or 'Linear decay'. 
        :param iteration: Current iteration of the training process. 
        :param num_iter: Total number of iterations for the training process. 
        """
        if method == "Fixed learning rate":
            pass
        elif method == "Linear decay":
            alpha = self.iteration / (self.max_iter)
            self.learning_rate = (1 - alpha) * self.initial_learning_rate \
                + alpha * self.initial_learning_rate * 0.01 
        else:
            raise ValueError("Not a valid learning schedule!")
            
        
    def check_convergence(self, gradient, iteration):
        """
        Checks if the gradient descent has converged by comparing the gradient 
        norm with a tolerance.
        :param gradient: The current gradient. 
        :param iteration: Current iteration in gradient descent. 
        :return: Boolean indicating if convergence is reached. 
        If True, prints the number of iterations used before convergence.
        """
        if jnp.linalg.norm(gradient) <= self.tol:
            print(f"Converged after {iteration} iterations")
            return True
        else:
            return False
        
    def calculate_change(self, gradient, learning_rate=None):
        """ 
        Calculates the change in parameters using the current gradient and 
        learning rate. 
        :param gradient: Current gradient. 
        :param learning_rate: Current learning rate. If None, uses the object's 
        learning rate (self.learning_rate). 
        :return: The calculated change. 
        """
        
        if learning_rate is None:
            learning_rate = self.learning_rate
  
        self.change = learning_rate * gradient
        
        return self.change
    
    def record(self, beta, cost_score):
        """ Records the current parameters (betas) and cost score at each 
        iteration of the gradient descent. 
        :param beta: Current parameters. 
        :param cost_score: Current cost score. 
        """
        self.betas.append(beta)
        self.cost_scores.append(cost_score)
            
        
    def iterate_full(self, X, target, max_iter, schedule_method):
        """
        Runs the gradient descent algorithm for a specified number of iterations.
        :param X: The input data. 
        :param target: Target values.
        :param max_iter: Maximum number of iterations for the gradient descent.
        :param schedule_method: Learning rate schedule method. 
        Options: 'Fixed learning rate' or 'Linear decay'. 
        :return: The resulting beta parameters. 
        If record_history is True, records the parameters and corresponding 
        cost scores at each iteration. 
        """
            
        beta = np.random.rand(jnp.shape(X)[1])
        beta = jnp.array(beta)
        self.iteration = 0
        self.max_iter = max_iter
            
        for i in range(max_iter):
            gradient = self.calc_gradient(X, target, beta)

            if self.skip_convergence_check is False:
                if self.check_convergence and self.check_convergence(gradient, self.iteration):
                    break
            
            self.iteration += 1
            
            self.learning_schedule(schedule_method)
            
            change = self.calculate_change(gradient)
            beta = beta - change
            
            if self.record_history is True:
                cost_score = self.cost_function(X, target, beta)
                
                self.record(beta, cost_score)

        if self.skip_convergence_check is False:
            if self.iteration == self.max_iter:
                print(f"Did not converge in {max_iter} iterations")
            
        return beta
    
    
    def iterate_minibatch(self, X, target, max_epoch, num_batches, 
                          schedule_method):
        """
        Runs stochastic gradient descent algorithm for a specified number of 
        epochs.
        :param X: The input data. 
        :param target: Target values.
        :param max_epoch: Maximum number of epochs.
        :param num_batches: The number of batches to split data in.
        :param schedule_method: Learning rate schedule method. 
        Options: 'Fixed learning rate' or 'Linear decay'. 
        :return: The resulting beta parameters. 
        If record_history is True, records the parameters and corresponding 
        cost scores at each iteration. 
        """

        beta = np.random.rand(jnp.shape(X)[1])
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_epoch*num_batches
        
        n = jnp.shape(X)[0]
        indices = np.arange(n)

        for epoch in range(max_epoch):
            np.random.shuffle(indices)

            batches = jnp.array_split(indices, num_batches)

            for i in range(num_batches):
                
                self.iteration += 1
                
                X_batch = X[batches[i], :]
                y_batch = target[batches[i]]

                gradient = self.calc_gradient(X_batch, y_batch, beta)
                
                self.learning_schedule(schedule_method)
                
                change = self.calculate_change(gradient)
                beta = beta - change
                
                if self.record_history is True:
                    cost_score = self.cost_function(X_batch, y_batch, beta)
                    self.record(beta, cost_score)
                
            if self.skip_convergence_check is False:
                total_gradient = self.calc_gradient(X, target, beta)
                if self.check_convergence(total_gradient, self.epoch):
                    break

            self.epoch += 1

        if self.skip_convergence_check is False:
            if self.epoch == max_epoch:
                print(f"Did not converge in {max_epoch} epochs")
                
        return beta

    
    def iterate(self, X, target, iteration_method, max_iter=None, 
                max_epoch=None, num_batches=None,
                schedule_method="Fixed learning rate"):
        """
        Performs model training iterations based on specified method, either 
        stochastic of normal.
        :param X: Input data.
        :param target: Target values.
        :param iteration_method: Method of iterations: 'Full' for normal 
        gradient descent or 'Stochastic' for stochastic gradient descent.
        :param max_iter: Maximum number of iterations for normal gradient 
        descent. Defaults to 10000 if not provided.
        :param max_epoch: Maximum number of epochs for stochastic gradient 
        descent. Defaults to 128 if not provided.
        :param num_batches: Number of minibatches for stochastic gradient descent. 
        Defaults to 10 if not provided.
        :param schedule_method: Learning rate adjustment method. Can be either
        'Fixed learning rate' or 'Linear decay'. Default is 'Fixed learning rate'.
        :return: Final beta parameters after training. 
        If record_history is True, records the parameters and corresponding 
        cost scores at each iteration. 
        """
                
        if iteration_method == "Full":
            max_iter = max_iter if not None else 10000
            
            self.beta = self.iterate_full(X, target, max_iter, schedule_method)
        
        elif iteration_method == "Stochastic":
            max_epoch = max_epoch if not None else 128
            num_batches = num_batches if not None else 10
            
            self.beta = self.iterate_minibatch(X, target, max_epoch, 
                                               num_batches, schedule_method)
            
        return self.beta
        


class GradientDescentMomentum(GradientDescent):
    """
    Class implementing gradient descent optimization algorithm with momentum.
    Inherits from the 'GradientDescent' parent class.

    :param momentum: fraction of the change from the previous time step to add
    to the current change. Value should lie between 0 and 1. Higher value means 
    more momentum.
    """
    
    def __init__(self, momentum, **kwargs):
        super().__init__(**kwargs)
        
        self.momentum = momentum
        self.change = 0
        
    def calculate_change(self, gradient, learning_rate=None):
        """ 
        Calculates the change in parameters using the momentum algorithm. 
        :param gradient: Current gradient. 
        :param learning_rate: Current learning rate. If None, uses the object's 
        learning rate (self.learning_rate). 
        :return: The calculated change. 
        """
        
        if learning_rate is None:
            learning_rate = self.learning_rate
            
        self.change = self.momentum*self.change + learning_rate*gradient
        return self.change
    
    
class GradientDescentAdagrad(GradientDescent):
    """
    Class implementing gradient descent optimimizing using the Adagrad algorithm.
    Inherits from the 'GradientDescent' parent class.

    :param momentum: fraction of the change from the previous time step to add
    to the current change. Value should lie between 0 and 1. Higher value means 
    more momentum.
    :param delta: Small constant for numerical stability.

    """

    def __init__(self, delta, momentum, **kwargs):
        super().__init__(**kwargs)
        
        self.delta = delta
        self.momentum = momentum
        self.change = 0
        self.acc_squared_gradient = 0
        
    def calculate_change(self, gradient, learning_rate=None):
        """ 
        Calculates the change in parameters using the Adagrad algorithm. 
        :param gradient: Current gradient. 
        :param learning_rate: Current learning rate. If None, uses the object's 
        learning rate (self.learning_rate). 
        :return: The calculated change. 
        """
        
        if learning_rate is None:
            learning_rate = self.learning_rate
         
        self.acc_squared_gradient = self.acc_squared_gradient + gradient*gradient
        
        self.change = (learning_rate/(self.delta + jnp.sqrt(self.acc_squared_gradient)))*gradient \
            + self.momentum*self.change

        return self.change
    
    
class GradientDescentRMSprop(GradientDescent):
    """
    Class implementing gradient descent optimimizing using the RMSProp algorithm.
    Inherits from the 'GradientDescent' parent class.

    :param delta: Small constant added for numerical stability.
    :param rho: Decay rate used to determine the extent of the moving average.
    :param n_inputs: The number of inputs in the model.
    """
    
    def __init__(self, delta, rho, **kwargs):
        super().__init__(**kwargs)
        
        self.delta = delta
        self.rho = rho
        self.change = 0
        self.acc_squared_gradient = 0
        
    def calculate_change(self, gradient, learning_rate=None):
        """ 
        Calculates the change in parameters using the RMSProp algorithm. 
        :param gradient: Current gradient. 
        :param learning_rate: Current learning rate. If None, uses the object's 
        learning rate (self.learning_rate). 
        :return: The calculated change. 
        """
        
        if learning_rate is None:
            learning_rate = self.learning_rate

        self.acc_squared_gradient = (self.rho*self.acc_squared_gradient + 
                                     (1-self.rho) * gradient**2)
        # Calculate the update
        self.change = (gradient*learning_rate) / (self.delta + jnp.sqrt(self.acc_squared_gradient))
        
        return self.change

        
class GradientDescentADAM(GradientDescent):
    """
    Class implementing gradient descent optimimizing using the Adam algorithm.
    Inherits from the 'GradientDescent' parent class.

    :param delta: Small constant added for numerical stability.
    :param rho1: Exponential decay rate for the first moment estimates.
    :param rho2: Exponential decay rate for the second-moment estimates.
    """

    def __init__(self, delta, rho1, rho2, **kwargs):
        super().__init__(**kwargs)
        
        self.delta = delta
        self.rho1 = rho1
        self.rho2 = rho2
        self.change = 0
        self.first_moment = 0
        self.second_moment = 0
        self.iter_adam = 0
        
    def calculate_change(self, gradient, learning_rate=None):
        """ 
        Calculates the change in parameters using the RMSProp algorithm. 
        :param gradient: Current gradient. 
        :param learning_rate: Current learning rate. If None, uses the object's 
        learning rate (self.learning_rate). 
        :return: The calculated change. 
        """
        
        if learning_rate is None:
            learning_rate = self.learning_rate
            
        self.iter_adam += 1

        self.first_moment = self.rho1*self.first_moment + (1 - self.rho1)*gradient
        self.second_moment = self.rho2*self.second_moment + (1-self.rho2)*(gradient*gradient)
        
        first_term = self.first_moment/(1.0 - self.rho1**self.iter_adam)
        second_term = self.second_moment/(1.0 - self.rho2**self.iter_adam)

        self.change = learning_rate*first_term/(jnp.sqrt(second_term)+self.delta)

        return self.change

if __name__ == "__main__":
    def make_design_matrix(x, degree):
        "Creates the design matrix for the given polynomial degree and input data"
        
        X = np.zeros((len(x), degree+1))
        
        for i in range(X.shape[1]):
            X[:,i] = np.power(x, i)
            
        return X

    def cost_function_OLS(X, y, beta):
        n = len(y)  # Define the number of data points
        return (1.0/n) * jnp.sum((y - jnp.dot(X, beta))**2)

    # def cost_function_OLS(X, y, beta):
    #     return (1.0/n)*jnp.sum((y-X.dot(beta))**2)
    
    def analytical_gradient(X, y, beta):
        n = len(y)
        return (2.0/n)*jnp.dot(X.T, ((jnp.dot(X, beta))-y))
    
    np.random.seed(1342)
    
    true_beta = [2, 0.5, 3.2]
    
    n = 100
    
    x = jnp.linspace(0, 1, n)
    y = jnp.sum(jnp.asarray([x ** p * b for p, b in enumerate(true_beta)]),
                    axis=0) + 0.1 * np.random.normal(size=len(x))
    
    # Making a design matrix to use for linear regression part
    degree = 2
    X = make_design_matrix(x, degree)
    
    learning_rate = 0.1
    tol=1e-3
    beta_guess = np.random.rand(3)
    
    X = jnp.array(X)  # Convert X to a JAX array
    y = jnp.array(y)  # Convert y to a JAX array
    
    np.random.seed(505)
    momentum=0.5
    delta=1e-8
    rho1 = 0.9
    rho2 = 0.99
    grad_descent_rms = GradientDescentADAM(delta, rho1, rho2, X=X, y=y, 
                                       learning_rate=learning_rate, tol=tol, 
                                       cost_function=cost_function_OLS,
                                       analytic_gradient=analytical_gradient,
                                       record_history=True)
    
    max_iter = 100000
    max_epochs = 500
    # beta_calculated = grad_descent_rms.iterate(iteration_method="Full", 
    #                                             max_iter=max_iter)
    beta_calculated = grad_descent_rms.iterate(iteration_method="Stochastic", 
                                                max_epoch=max_epochs,
                                                num_batches=10)
    print(beta_calculated)
    
    print(grad_descent_rms.cost_scores)

    