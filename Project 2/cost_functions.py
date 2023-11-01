# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:05:52 2023

@author: vildesn
"""

import jax.numpy as jnp

from jax import grad as jax_grad

class Cost_Functions:
    def __init__(self, cost_function, target):
        self.target = target
        self.func_name = cost_function
        self.cost_function = self.set_cost_function()
        
        self.cost_function_grad = jax_grad(self.cost_function)
    
    def __str__(self):
        return self.func_name

    def set_cost_function(self):
        if self.func_name == 'LogReg':
            return self.CostLogReg(self.target)

        else:
            raise ValueError('Cost function is not available.' 
                             'Expected: LogReg, not {}' .format(self.func_name))

    def CostLogReg(self, target):
    
        def func(pred):
            
            return -(1.0 / target.shape[0]) * jnp.sum(
                (target * jnp.log(pred + 10e-10)) + ((1 - target) * jnp.log(1 - pred + 10e-10)))
    
        return func