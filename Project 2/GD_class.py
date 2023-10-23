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
        
    def iterate(self, learning_rate, max_iter):
            
        beta = np.random.rand(np.shape(self.X)[1])
        iteration = 0
            
        for i in range(max_iter):
            gradient = self.calc_gradient(self.X, self.y, beta)
            beta -= learning_rate * gradient
                
            if self.check_convergence(gradient, iteration):
                break
                
            iteration += 1
                
        if iteration == max_iter:
            print(f"Did not converge in {max_iter} iterations")
            
        return beta
