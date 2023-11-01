# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 09:26:42 2023

@author: vildesn
"""

from NeuralNetwork import Neural_Network
import numpy as np
import jax.numpy as jnp

np.random.seed(1234)

# Create design matrix
X = jnp.array([[0, 0],
               [0, 1],
               [1, 0],
               [1, 1]])
# Since X is our input we need it to be a vector, hence we reshape it 
#X = X.reshape((X.shape[0]*X.shape[1], 1))

# The XOR gate
target_XOR = jnp.array( [ 0, 1 ,1, 0])
# The OR gate
target_OR = jnp.array( [ 0, 1 ,1, 1])
# The AND gate
target_AND = jnp.array( [ 0, 0 ,0, 1])

#%%
n_hidden_layers = 1
n_hidden_nodes = 2
n_outputs = 1

ffnn = Neural_Network(X, target_OR, n_hidden_layers, n_hidden_nodes, n_outputs, activation_function='sigmoid')
str(ffnn)

print(f'Output layer has {ffnn.output_layer.n_nodes} node and uses the {str(ffnn.output_layer.activation_function)} as activation function')



print('Weights hidden layer 1:\n'
      f'{ffnn.hidden_layers[0].weights}')
print('Biases hidden layer 1:\n'
      f'{ffnn.hidden_layers[0].biases}')

print('weights output layer:\n'
      f'{ffnn.output_layer.weights}')
print('Biases output layer:\n'
      f'{ffnn.output_layer.biases}')



#%%

ffnn.feed_forward()
print('Weights hidden layer 1:\n'
      f'{ffnn.hidden_layers[0].weights}')
print('Biases hidden layer 1:\n'
      f'{ffnn.hidden_layers[0].biases}')

print('weights output layer:\n'
      f'{ffnn.output_layer.weights}')
print('Biases output layer:\n'
      f'{ffnn.output_layer.biases}')

print("Final prediction:\n"
      f"{ffnn.output_layer.output}")

#%%



