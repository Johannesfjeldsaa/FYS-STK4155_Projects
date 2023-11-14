# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:05:52 2023

@author: vildesn
"""

import jax.numpy as jnp

from jax import grad as jax_grad

def grad_mse(target, pred):
        
    n = len(target) # Number of inputs
        
    return 2/n * (pred - target)

def grad_cost_logreg(target, pred):
    
    return (pred - target)/(pred * (1 - pred))

