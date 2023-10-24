# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:14:44 2023

@author: vildesn
"""
#import autograd.numpy as np
#from autograd import grad


# import jax
import jax.numpy as jnp
import numpy as np
#import matplotlib.pyplot as plt

from jax import grad as jax_grad

class GradientDescent:
    
    def __init__(self, X, y, learning_rate, tol, cost_function, 
                 analytic_gradient=None, iteration_method="Normal",
                 record=False):
        
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.tol = tol
        self.cost_function = cost_function
        self.iteration_method = iteration_method
        
        if record == "All":
            self.betas = []
            self.cost_scores = []
        elif record == "Cost scores":
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
        if method == "Fixed learning rate":
            pass
        elif method == "Linear decay":
            alpha = self.iteration / (self.max_iter)
            self.learning_rate = (1 - alpha) * self.initial_learning_rate \
                + alpha * self.initial_learning_rate * 0.01 
        elif method == "Exponential decay":
            #Legg inn senere
            pass
        else:
            raise ValueError("Not a valid learning schedule!")
            
        
    def check_convergence(self, gradient, iteration):
        if np.linalg.norm(gradient) <= self.tol:
            print(f"Converged after {iteration} iterations")
            return True
        else:
            return False
        
    def calculate_change(self, gradient):
        self.change = self.learning_rate * gradient
        return self.change
    
    def record(self, beta):
        try:
            self.betas.append[beta]
        except NameError:
            pass

        

    def iterate_full(self, max_iter, schedule_method):
            
        beta = np.random.rand(np.shape(self.X)[1])
        self.iteration = 0
        self.max_iter = max_iter
            
        for i in range(max_iter):
            gradient = self.calc_gradient(self.X, self.y, beta)
            
            if self.check_convergence(gradient, self.iteration):
                break
            
            self.iteration += 1
            
            self.learning_schedule(schedule_method)
            
            change = self.calculate_change(gradient)
            beta = beta - change

                
        if self.iteration == self.max_iter:
            print(f"Did not converge in {max_iter} iterations")
            
        return beta
    
    
    def iterate_minibatch(self, max_epoch, num_batches, schedule_method):

        beta = np.random.rand(np.shape(self.X)[1])
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_epoch*num_batches
        
        n = np.shape(self.X)[0]
        indices = np.arange(n)
        
        #batch_size = int(n/num_batches)

        for epoch in range(max_epoch):
            np.random.shuffle(indices)
            batches = np.array_split(indices, num_batches)

            for i in range(num_batches):
                
                X_batch = self.X[batches[i], :]
                y_batch = self.y[batches[i]]

                gradient = self.calc_gradient(X_batch, y_batch, beta)
                
                self.learning_schedule(schedule_method)
                
                change = self.calculate_change(gradient)
                beta = beta - change
                
                self.iteration += 1
            
            # Fiks riktig konvergenskriterie
            total_gradient = self.calc_gradient(self.X, self.y, beta)
            if self.check_convergence(total_gradient, self.epoch):
                break
            
            self.epoch += 1
            
        if self.epoch == max_epoch:
            print(f"Did not converge in {max_epoch} epochs")
                
        return beta

    
    def iterate(self, iteration_method, max_iter=None, max_epoch=None, num_batches=None,
                schedule_method="Fixed learning rate"):
        
        if iteration_method == "Full":
            max_iter = max_iter if not None else 10000
            
            self.beta = self.iterate_full(max_iter, schedule_method)
        
        elif iteration_method == "Stochastic":
            max_epoch = max_epoch if not None else 128
            num_batches = num_batches if not None else 10
            
            self.beta = self.iterate_minibatch(max_epoch, num_batches, schedule_method)
            
        return self.beta
        


class GradientDescentMomentum(GradientDescent):
    
    def __init__(self, momentum, **kwargs):
        super().__init__(**kwargs)
        
        self.momentum = momentum
        self.change = 0
        
    def calculate_change(self, gradient):
        self.change = self.momentum*self.change + self.learning_rate*gradient
        return self.change
    
    
class GradientDescentAdagrad(GradientDescent):
    # Sjekk hvorfor så sakte konvergering
    def __init__(self, delta, momentum, **kwargs):
        super().__init__(**kwargs)
        
        self.delta = delta
        self.momentum = momentum
        self.change = 0
        self.acc_squared_gradient = 0
        
    def calculate_change(self, gradient):
        self.acc_squared_gradient = self.acc_squared_gradient + gradient*gradient
        
        self.change = (self.learning_rate/(self.delta + np.sqrt(self.acc_squared_gradient)))*gradient \
            + self.momentum*self.change

        return self.change
    
    
class GradientDescentRMSprop(GradientDescent):
    
    def __init__(self, delta, rho, **kwargs):
        super().__init__(**kwargs)
        
        self.delta = delta
        self.rho = rho
        self.change = 0
        p = np.shape(self.X)[1]
        # This does not work yet
        self.acc_squared_gradient = np.zeros(p)
        
    def calculate_change(self, gradient):
        
        self.acc_squared_gradient = (self.rho*self.acc_squared_gradient + 
                                     (1-self.rho) * gradient**2)
        # Calculate the update
        self.change = (gradient*self.learning_rate) / (self.delta + np.sqrt(self.acc_squared_gradient))
        
        return self.change

        
class GradientDescentADAM(GradientDescent):
    # Fiks for stokastisk
    def __init__(self, delta, rho1, rho2, **kwargs):
        super().__init__(**kwargs)
        
        self.delta = delta
        self.rho1 = rho1
        self.rho2 = rho2
        self.change = 0
        self.first_moment = 0
        self.second_moment = 0
        
    def calculate_change(self, gradient):
        
        self.first_moment = self.rho1*self.first_moment + (1 - self.rho1)*gradient
        self.second_moment = self.rho2*self.second_moment + (1-self.rho2)*(gradient*gradient)
        
        first_term = self.first_moment/(1.0 - self.rho1**self.iteration)
        second_term = self.second_moment/(1.0 - self.rho2**self.iteration)

        self.change = self.learning_rate*first_term/(np.sqrt(second_term)+self.delta)
        
        return self.change

if __name__ == "__main__":
    def make_design_matrix(x, degree):
        "Creates the design matrix for the given polynomial degree and input data"
        
        X = np.zeros((len(x), degree+1))
        
        for i in range(X.shape[1]):
            X[:,i] = np.power(x, i)
            
        return X

    def cost_function_OLS(X, y, beta):
        return (1.0/n)*np.sum((y-X.dot(beta))**2)
    
    def analytical_gradient(X, y, beta):
        return (2.0/n)*np.dot(X.T, ((np.dot(X,beta))-y))
    
    np.random.seed(1342)
    
    true_beta = [2, 0.5, 3.2]
    
    n = 100
    
    x = np.linspace(0, 1, n)
    y = np.sum(np.asarray([x ** p * b for p, b in enumerate(true_beta)]),
                    axis=0) + 0.1 * np.random.normal(size=len(x))
    
    # Making a design matrix to use for linear regression part
    degree = 2
    X = make_design_matrix(x, degree)
    
    learning_rate = 0.1
    tol=1e-3
    beta_guess = np.random.rand(3)

    np.random.seed(505)
    momentum=0.5
    delta=1e-8
    rho1 = 0.9
    rho2 = 0.99
    grad_descent_rms = GradientDescent(X=X, y=y, 
                                             learning_rate=learning_rate, tol=tol, 
                                             cost_function=cost_function_OLS)
    
    max_iter = 100000
    max_epochs = 500
    beta_calculated = grad_descent_rms.iterate(iteration_method="Full", 
                                                max_iter=max_iter)
    # beta_calculated = grad_descent_rms.iterate(iteration_method="Stochastic", 
    #                                            max_epoch=max_epochs,
    #                                            num_batches=10)
    print(beta_calculated)
    

    