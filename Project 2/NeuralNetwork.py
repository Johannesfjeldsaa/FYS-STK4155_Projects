# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:17:59 2023

@author: vildesn
"""

# import numpy as np


# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# y_xor = np.array([0, 1, 1, 0])
# y_and = np.array([0, 0, 0, 1])
# y_or = np.array([0, 1, 1, 1])

# import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from jax import grad as jax_grad
import jax.numpy as jnp

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def feed_forward(X):
    # weighted sum of inputs to the hidden layer
    z_h = np.matmul(X, hidden_weights) + hidden_bias
    # activation in the hidden layer
    a_h = sigmoid(z_h)
    
    # weighted sum of inputs to the output layer
    z_o = np.matmul(a_h, output_weights) + output_bias
    
    # activation of the output layer
    a_o = sigmoid(z_o)
    
    return a_o

# we obtain a prediction by taking the class with the highest likelihood
def predict(X):
    output = feed_forward(X)
    
    if output > 0.5:
        return 1
    else:
        return 0
    
def CostLogReg(target):

    def func(pred):
        
        return -(1.0 / target.shape[0]) * jnp.sum(
            (target * jnp.log(pred + 10e-10)) + ((1 - target) * jnp.log(1 - pred + 10e-10)))

    return func

#print(CostLogReg(np.array([0]))(np.array([0])))

target = jnp.array([0.])

cost_func = CostLogReg(target)
cost_func_derivative = jax_grad(cost_func)

cost_func_derivative(jnp.array([0.1]))

#%%
def back_propagation():
    pass

# ensure the same random numbers appear every time
np.random.seed(0)

# Design matrix
X = np.array([[0, 0], [0, 1], [1, 0],[1, 1]], dtype=np.float64)

# The XOR gate
yXOR = np.array([0, 1 ,1, 0])
# The OR gate
yOR = np.array([0, 1 ,1, 1])
# The AND gate
yAND = np.array([0, 0 ,0, 1])

# Defining the neural network
n_features = 2
n_hidden_neurons = 2
n_outputs = 1

# we make the weights normally distributed using numpy.random.randn
# weights and bias in the hidden layer
hidden_weights = np.random.randn(n_features, n_hidden_neurons)
hidden_bias = np.zeros(n_hidden_neurons) + 0.01

# weights and bias in the output layer
output_weights = np.random.randn(n_hidden_neurons, n_outputs)
output_bias = np.zeros(n_outputs) + 0.01

x = [0, 1]

outputs = feed_forward(x)
print(outputs)

predictions = predict(x)
print(predictions)

#%% 

#Set up the cost function (cross entropy for classification of binary cases).

#Calculate the gradients needed for the back propagation part.

#Use the gradients to train the network in the back propagation part. Think of using automatic differentiation.

#Train the network and study your results and compare with results obtained either with scikit-learn or TensorFlow.

#%%
# import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def feed_forward(X):
    # weighted sum of inputs to the hidden layer
    z_h = np.matmul(X, hidden_weights) + hidden_bias
    # activation in the hidden layer
    a_h = sigmoid(z_h)
    
    # weighted sum of inputs to the output layer
    z_o = np.matmul(a_h, output_weights) + output_bias
    # softmax output
    # axis 0 holds each input and axis 1 the probabilities of each category
    probabilities = sigmoid(z_o)
    return probabilities

# we obtain a prediction by taking the class with the highest likelihood
def predict(X):
    probabilities = feed_forward(X)
    return np.argmax(probabilities, axis=1)

# ensure the same random numbers appear every time
np.random.seed(0)

# Design matrix
X = np.array([ [0, 0], [0, 1], [1, 0],[1, 1]],dtype=np.float64)

# The XOR gate
yXOR = np.array( [ 0, 1 ,1, 0])
# The OR gate
yOR = np.array( [ 0, 1 ,1, 1])
# The AND gate
yAND = np.array( [ 0, 0 ,0, 1])

# Defining the neural network
n_inputs, n_features = X.shape
n_hidden_neurons = 2
n_categories = 2
n_features = 2

# we make the weights normally distributed using numpy.random.randn

# weights and bias in the hidden layer
hidden_weights = np.random.randn(n_features, n_hidden_neurons)
hidden_bias = np.zeros(n_hidden_neurons) + 0.01

# weights and bias in the output layer
output_weights = np.random.randn(n_hidden_neurons, n_categories)
output_bias = np.zeros(n_categories) + 0.01

probabilities = feed_forward(X)
print(probabilities)


predictions = predict(X)
print(predictions)
     
     