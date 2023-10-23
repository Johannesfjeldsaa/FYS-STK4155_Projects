# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:14:44 2023

@author: vildesn
"""
import autograd.numpy as np
from autograd import grad

class GradientDescent:
    
    def __init__(self, X, y, learning_rate, tol, cost_function, analytic_gradient=None, iteration_method="Normal"):
        
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.tol = tol
        self.cost_function = cost_function
        self.iteration_method = iteration_method
        
        if analytic_gradient is not None:
            if callable(analytic_gradient):
                self.calc_gradient = analytic_gradient
            else:
                raise ValueError("Analytical gradient must be a function")
                
        else:
            # When defining the cost function, the third parameter must be 
            # the one to differentiate by
            self.calc_gradient = grad(self.cost_function, 2)
            
        
    def check_convergence(self, gradient, iteration):
        if np.linalg.norm(gradient) <= self.tol:
            print(f"Converged after {iteration} iterations")
            return True
        else:
            return False
        
    def calculate_change(self, gradient):
        self.change = self.learning_rate * gradient
        return self.change

    def iterate_full(self, max_iter):
            
        beta = np.random.rand(np.shape(self.X)[1])
        self.iteration = 0
            
        for i in range(max_iter):
            gradient = self.calc_gradient(self.X, self.y, beta)
            
            if self.check_convergence(gradient, self.iteration):
                break
            
            self.iteration += 1
            
            change = self.calculate_change(gradient)
            beta = beta - change
                
        if self.iteration == max_iter:
            print(f"Did not converge in {max_iter} iterations")
            
        return beta
    
    
    def iterate_minibatch(self, max_epoch, num_batches):
        # Hvordan introdusere learning schedule for å få denne til å konvergere
        beta = np.random.rand(np.shape(self.X)[1])
        self.epoch = 0
        
        n = np.shape(self.X)[0]
        batch_size = int(n/num_batches)

        for epoch in range(max_epoch):

            for i in range(num_batches):
                random_ind = batch_size*np.random.randint(num_batches)
                # Pick out belonging design matrix and y values to the randomized batch
                X_batch = X[random_ind:random_ind+batch_size]
                y_batch = y[random_ind:random_ind+batch_size]

                gradient = self.calc_gradient(X_batch, y_batch, beta)
                
                change = self.calculate_change(gradient)
                beta = beta - change
            
            total_gradient = self.calc_gradient(self.X, self.y, beta)
            if self.check_convergence(total_gradient, self.epoch):
                break
            
            self.epoch += 1
            
        if self.epoch == max_epoch:
            print(f"Did not converge in {max_epoch} epochs")
                
        return beta

    
    def iterate(self, iteration_method, max_iter=None, max_epoch=None, num_batches=None):
        
        if iteration_method == "Full":
            max_iter = max_iter if not None else 10000
            
            self.beta = self.iterate_full(max_iter)
        
        elif iteration_method == "Stochastic":
            max_epoch = max_epoch if not None else 128
            num_batches = num_batches if not None else 10
            
            self.beta = self.iterate_minibatch(max_epoch, num_batches)
            
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
    def __init__(self, delta, **kwargs):
        super().__init__(**kwargs)
        
        self.delta = delta
        self.change = 0
        self.acc_squared_gradient = 0
        
    def calculate_change(self, gradient):
        self.acc_squared_gradient = self.acc_squared_gradient + gradient*gradient
        
        self.change = (self.learning_rate/(self.delta + np.sqrt(self.acc_squared_gradient)))*gradient

        return self.change
    
    
class GradientDescentRMSprop(GradientDescent):
    
    def __init__(self, delta, rho, **kwargs):
        super().__init__(**kwargs)
        
        self.delta = delta
        self.rho = rho
        self.change = 0
        p = np.shape(self.X)[1]
        # This does not work yet
        self.acc_squared_gradient = np.zeros(shape=(p, p))
        
    def calculate_change(self, gradient):
        
        self.acc_squared_gradient = (self.rho*self.acc_squared_gradient + 
                                     (1-self.rho) * gradient**2)
        # Calculate the update
        self.change = (gradient*self.learning_rate) / (self.delta + np.sqrt(self.acc_squared_gradient))
        
        return self.change

        
class GradientDescentADAM(GradientDescent):
    
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

    np.random.seed(1342)
    delta=1e-8
    rho1 = 0.9
    rho2 = 0.99
    grad_descent_rms = GradientDescent(X=X, y=y, 
                                             learning_rate=learning_rate, tol=tol, 
                                             cost_function=cost_function_OLS,
                                             analytic_gradient=analytical_gradient)
    
    max_iter = 100000
    max_epochs = 1000
    beta_calculated = grad_descent_rms.iterate(iteration_method="Stochastic", max_epoch=max_epochs,
                                               num_batches=5)
    print(beta_calculated)
    

    