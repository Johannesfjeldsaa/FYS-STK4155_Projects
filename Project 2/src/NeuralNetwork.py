
import numpy as np
import jax.numpy as jnp

from activation_functions import Activation_Functions
from GD_class import *
from copy import deepcopy

    
class Dense_Layer:
    """
    Class for a dense layer in a neural network. Each layer consists of n_nodes with their own bias (float).
    Further each has weights vector of same length as n_inputs.
    """

    def __init__(self, n_inputs, n_nodes, activation_function, optimizer, 
                 weights=None, biases=None):
        

        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.activation_function = Activation_Functions(activation_function)
        self.weight_optimizer=deepcopy(optimizer)
        self.bias_optimizer=deepcopy(optimizer)
        
        self.constructed_from_scratch = True if weights or biases is None else False
        
        if self.constructed_from_scratch:
            # initiate weights of network randomly, scale by 0.1 to keep small
            # Initiate biases to 0.01
            self.weights = .1 * np.random.randn(self.n_inputs, self.n_nodes)
            self.biases = np.zeros((1, self.n_nodes)) #+ 0.01
        else:
            # Brukes om man allerede har trent en modell.
            self.weights = weights
            self.biases = biases

        self.output = None
        self.output_pre_activation = None
        
        self.delta = None
        self.dC_dW = None
        self.dC_db = None

    def forward_propagation(self, inputs):
        """
        Forward propagation of the network.
        :param inputs: Input data
        :return: Output of the layer
        """

        self.output_pre_activation = jnp.dot(inputs, self.weights) + self.biases
        self.output = self.activation_function.activation_function(self.output_pre_activation)

        return self.output
    
    def calculate_gradients(self, delta_next, weights_next, input_value, lmbd):
        
        da_dz = self.activation_function.grad_activation_function(self.output_pre_activation)

        self.delta = np.matmul(delta_next, weights_next.T) * da_dz
 
        self.dC_dW = jnp.matmul(input_value.T, self.delta)
        self.dC_db = jnp.sum(self.delta, axis=0)
        
        self.dC_dW = self.dC_dW + lmbd * self.weights

    
    def backward_propagation(self, learning_rate):

        change_W = self.weight_optimizer.calculate_change(self.dC_dW, learning_rate)
        change_b = self.bias_optimizer.calculate_change(self.dC_db, learning_rate)
    
        self.weights = self.weights - change_W
        self.biases = self.biases - change_b
        
class Output_Layer(Dense_Layer):
    
    def __init__(self, grad_cost_function, **kwargs):
        super().__init__(**kwargs)
        
        self.grad_cost_function = grad_cost_function
        
    def calculate_gradients(self, input_value, target, lmbd):
        
        dC_da = self.grad_cost_function(target, self.output)
        da_dz = self.activation_function.grad_activation_function(self.output_pre_activation)
        
        self.delta = dC_da * da_dz # Elementwise multiplication

        self.dC_dW = jnp.matmul(input_value.T, self.delta)
        self.dC_db = jnp.sum(self.delta, axis=0)
        
        # Adding a regularization term
        self.dC_dW = self.dC_dW + lmbd * self.weights
        
    
class Neural_Network:

    def __init__(self, n_inputs, n_hidden_layers, n_hidden_nodes, n_outputs,
                 grad_cost_function,
                 learning_rate=0.1,
                 lmbd=0.0,
                 activation_function_hidden='ReLU', 
                 activation_function_output='Sigmoid',
                 optimizer=None,
                 weights=None, biases=None,
                 classification_problem=False):

        # Initiate the basics of the network
        self.n_inputs = n_inputs
        
        self.activation_function_hidden = activation_function_hidden
        self.activation_function_output = activation_function_output
        
        self.grad_cost_function = grad_cost_function
        
        if optimizer is None:
            self.optimizer = GradientDescentADAM(delta=1e-8, rho1=0.9, 
                                                 rho2=0.99,
                                                 learning_rate=learning_rate)
        elif isinstance(optimizer, GradientDescent):
            self.optimizer = optimizer
        else:
            raise ValueError(f"Expected an instance of GradientDescent class, not {type(optimizer)}")
            
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.lmbd = lmbd
        
        if weights or biases is None:
            self.construct_network_from_scratch = True
            self.construct_layer = self.construct_layer_from_scratch
            self.construct_output_layer = self.construct_output_layer_from_scratch
        else:
            self.construct_network_from_scratch = False
            self.construct_layer = self.construct_layer_from_previous_training
            self.construct_output_layer = self.construct_output_layer_from_previous_training
            self.weights = weights
            self.biases = biases

        # Initiate the hidden nodes and layers
        self.n_hidden_layers = n_hidden_layers
        if isinstance(n_hidden_nodes, int) or isinstance(n_hidden_nodes, float):
            n_hidden_nodes = [int(n_hidden_nodes)] * self.n_hidden_layers
        elif len(n_hidden_nodes) != self.n_hidden_layers:
            raise ValueError('n_hidden_nodes must be integer or a list of same length as number of hidden layers')

        self.n_hidden_nodes = n_hidden_nodes
        self.n_outputs = n_outputs

        self.hidden_layers = self.initiate_hidden_layers()
        self.output_layer = self.initiate_output_layer()
        
        self.classification_problem = classification_problem
        

    def __str__(self):
        return (f"Neural Network with {self.n_hidden_layers} hidden layers and {self.n_hidden_nodes} nodes per layer. "
                f"The activation function for hidden layers is {self.activation_function_hidden}."
                f"The activation function for the output layer is {self.activation_function_output}.")

    def initiate_hidden_layers(self):
        """
        Initiates hidden layers of the network. If the nodes per layer is given as an integer, all hidden layers
        will have the same number of nodes, then we create a list of same length as n_hidden_layers.
        If the nodes per layer is given as a list it has to have same number of elements as n_hidden_layers.

        The first layer is initiated with n_inputs the same length as the input layer.
        The rest of the layers are initiated with n_inputs the same length as the previous hidden layer.

        :return: Hidden layers of the network
        """

        hidden_layers = []
        for i in range(self.n_hidden_layers):
            if i == 0:
                n_inputs = self.n_inputs
            else:
                n_inputs = self.n_hidden_nodes[i - 1]

            if self.construct_network_from_scratch:
                hidden_layers.append(self.construct_layer(n_inputs=n_inputs,
                                                          n_hidden_nodes=self.n_hidden_nodes[i]))
            else:
                hidden_layers.append(self.construct_layer(n_inputs=n_inputs,
                                                          n_hidden_nodes=self.n_hidden_nodes[i],
                                                          layer_indx=i))

        return hidden_layers

    def initiate_output_layer(self):
        """
        Initiates the output layer of the network.
        :return: Output layer of the network
        """
        if self.n_hidden_layers == 0:
            n_inputs=self.n_inputs
        else:
            n_inputs=self.hidden_layers[-1].n_nodes
                
        if self.construct_network_from_scratch:

            return self.construct_output_layer(n_inputs=n_inputs,
                                               n_hidden_nodes=self.n_outputs)
        else:
            return self.construct_output_layer(n_inputs=n_inputs,
                                               n_hidden_nodes=self.n_outputs,
                                               layer_indx=-1)
        
    def construct_output_layer_from_scratch(self, n_inputs, n_hidden_nodes):
        """
        Constructs a layer without weights and biases.
        :param n_inputs: Number of inputs to the layer
        :param n_hidden_nodes: Number of nodes in the layer
        :param activation_function: Activation function of the layer
        :return: Layer with random weights and biases set to 0.01
        """
        return Output_Layer(grad_cost_function=self.grad_cost_function, 
                           n_inputs=n_inputs,
                           n_nodes=n_hidden_nodes,
                           activation_function=self.activation_function_output,
                           optimizer=self.optimizer)

    def construct_output_layer_from_previous_training(self, n_inputs, n_hidden_nodes, layer_indx):
        """
        Constructs a layer with weights and biases.
        :param n_inputs: Number of inputs to the layer
        :param n_hidden_nodes: Number of nodes in the layer
        :param activation_function: Activation function of the layer
        :param layer_indx: Index of the layer
        :return: Layer with weights and biases
        """
        return Output_Layer(grad_cost_function=self.grad_cost_function, 
                           n_inputs=n_inputs,
                           n_nodes=n_hidden_nodes,
                           activation_function=self.activation_function_output,
                           optimizer=self.optimizer,
                           weights=self.weights[layer_indx],
                           biases=self.biases[layer_indx])

    def construct_layer_from_scratch(self, n_inputs, n_hidden_nodes):
        """
        Constructs a layer without weights and biases.
        :param n_inputs: Number of inputs to the layer
        :param n_hidden_nodes: Number of nodes in the layer
        :param activation_function: Activation function of the layer
        :return: Layer with random weights and biases set to 0.01
        """
        return Dense_Layer(n_inputs=n_inputs,
                           n_nodes=n_hidden_nodes,
                           activation_function=self.activation_function_hidden,
                           optimizer=self.optimizer)

    def construct_layer_from_previous_training(self, n_inputs, n_hidden_nodes, layer_indx):
        """
        Constructs a layer with weights and biases.
        :param n_inputs: Number of inputs to the layer
        :param n_hidden_nodes: Number of nodes in the layer
        :param activation_function: Activation function of the layer
        :param layer_indx: Index of the layer
        :return: Layer with weights and biases
        """
        return Dense_Layer(n_inputs=n_inputs,
                           n_nodes=n_hidden_nodes,
                           activation_function=self.activation_function_hidden,
                           optimizer=self.optimizer,
                           weights=self.weights[layer_indx],
                           biases=self.biases[layer_indx])
    
    def classify(self):
        """
        Performs binary classification on the output of the output layer. 
        Values higher that 0.5 are set to 1 and lower values are set to 0.
        """
        self.output_layer.output = jnp.where(self.output_layer.output > 0.5, 1.0, 0.0)

    def feed_forward(self, X):
        """
        Performs a feed forward pass of the neural network.
        :param X: The input data to the neural network.
        """
        
        if self.n_hidden_layers == 0:
            
            self.output_layer.forward_propagation(X)
        
        else:
            
            for i in range(self.n_hidden_layers):
                
                if i == 0:
                    input_value = X
                else:
                    input_value = self.hidden_layers[i-1].output
                
                self.hidden_layers[i].forward_propagation(input_value)
            
            self.output_layer.forward_propagation(self.hidden_layers[-1].output)
            
        
    def feed_backward(self, X, target):
        """
        Executes a single pass of the backward propagation algorithm, first
        calculating the gradients for all the layers, then updating all weights
        and biases.
        adjusting the network's internal weights and biases to minimize error
        Performs a feed forward pass of the neural network.
        :param X: The input data to the neural network.
        """
        
        if self.n_hidden_layers == 0:
            self.output_layer.calculate_gradients(input_value=X,
                                                  target=target,
                                                  lmbd=self.lmbd)
            
            self.output_layer.backward_propagation(learning_rate=self.learning_rate)
            
            
        else:
            # Calculate all the necessary gradients:
            self.output_layer.calculate_gradients(input_value=self.hidden_layers[-1].output,
                                                  target=target,
                                                  lmbd=self.lmbd)
            
            for i in reversed(range(self.n_hidden_layers)):
                
                layer = self.hidden_layers[i]
                
                if i == (self.n_hidden_layers - 1):
                    delta = self.output_layer.delta
                    weights = self.output_layer.weights
                    
                else:
                    delta = self.hidden_layers[i+1].delta
                    weights = self.hidden_layers[i+1].weights
                    
                if i == 0:
                    input_value = X
                else:
                    input_value = self.hidden_layers[i-1].output
    
                layer.calculate_gradients(delta_next=delta, 
                                          weights_next = weights,
                                          input_value = input_value,
                                          lmbd=self.lmbd)
                
            # Update the weights of all the layers:
            for layer in [self.output_layer] + self.hidden_layers:
                layer.backward_propagation(learning_rate=self.learning_rate)
                
            
    def learning_schedule(self, method, iteration, num_iter):
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
            alpha = iteration / (num_iter)
            self.learning_rate = (1 - alpha) * self.initial_learning_rate \
                + alpha * self.initial_learning_rate * 0.01 
        
        
    def train(self, X, target, num_iter=1000, 
              n_minibatches=None,
              method="Fixed learning rate"):
        """ 
        Trains the neural network over a given number of iterations. 
        :param X: The input data. 
        :param num_iter: The number of iterations for training. Default is 1000.
        If using stochastic method, num_iter becomes the number of epochs.
        :param method: The learning schedule method to be used for updating the
        learning rate. Options: 'Fixed learning rate' or 'Linear decay'. 
        Default is 'Fixed learning rate'. 
        :param n_minimatches: Number of minibatches to use if stochastic.
        Default is None, in which case the full dataset is used.
        :return: None. The weights and biases of the network are updated in-place.
        """
        
        if n_minibatches is None:
            
            for i in range(num_iter):
                
                self.learning_schedule(method, iteration=i, num_iter=num_iter)
    
                self.feed_forward(X)
                self.feed_backward(X, target)
        
        else:
            # Using stochastic method
            
            indices = np.arange(X.shape[0])
            
            iteration = 0
            
            for epoch in range(num_iter):
                
                np.random.shuffle(indices)
                
                batches = jnp.array_split(indices, n_minibatches)
                
                for i in range(n_minibatches):
                    
                    iteration += 1
                    
                    X_batch = X[batches[i], :]
                    target_batch = target[batches[i]]
                    
                    self.learning_schedule(method, iteration=iteration, 
                                           num_iter=num_iter)
                    
                    self.feed_forward(X_batch)
                    self.feed_backward(X_batch, target_batch)

    
    def predict(self, X):
        """
        Predicts the output for the provided input after training the network. 
        :param X: The input data you want to predict an output from.
        :return: The predicted output from the network. 
        For classification problems, the output from the output layer is made 
        binary (either 0 or 1) before returning.
        For regression problems, the output from the output layer is directly
        returned.
        """

        self.feed_forward(X)
        
        if self.classification_problem:
            self.classify()
        
        return self.output_layer.output
        
        
        
        
        
