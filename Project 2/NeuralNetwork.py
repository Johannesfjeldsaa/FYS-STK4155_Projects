
import numpy as np
import jax.numpy as jnp

from activation_functions import Activation_Functions
from cost_functions import Cost_Functions


class Dense_Layer:
    """
    Class for a dense layer in a neural network. Each layer consists of n_nodes with their own bias (float).
    Further each has weights vector of same length as n_inputs.
    """
    def __init__(self, n_inputs, n_nodes, activation_function, weights=None, biases=None):

        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.activation_function = Activation_Functions(activation_function)
        self.constructed_from_scratch = True if weights or biases is None else False

        if self.constructed_from_scratch:
            # initiate weights of network randomly, scale by 0.1 to keep small
            # Initiate biases to zero
            self.weights = .1 * np.random.randn(self.n_inputs, self.n_nodes)
            self.biases = np.zeros((1, self.n_nodes)) + 0.01
        else:
            # Brukes om man allerede har trent en modell.
            # Legg til kontroll av dimensjoner og type
            self.weights = weights
            self.biases = biases

        self.output = None
        self.output_pre_activation = None

    def forward_propagation(self, inputs):
        """
        Forward propagation of the network.
        :param inputs: Input data
        :return: Output of the layer
        """
        self.output_pre_activation = jnp.dot(inputs, self.weights) + self.biases
        self.output = self.activation_function.activation_function(self.output_pre_activation)
        
        return self.output
    
    def calculate_derivatives(self, delta):
        
        pass
    
    def backward_propagation(self, delta, learning_rate):
        
        self.calculate_derivatives(delta=delta)
        
        self.weights = self.weights - learning_rate * self.dC_dW
        self.biases = self.biases - learning_rate * self.dC_db
        
class Output_Layer(Dense_Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def backward_propagation(self, learning_rate):
        pass
    
    def calculate_derivatives(self):
        pass


class Neural_Network:

    def __init__(self, X, target, n_hidden_layers, n_hidden_nodes, n_outputs,
                 activation_function='ReLU', cost_function='LogReg', 
                 weights=None, biases=None,
                 classification_problem=False):

        # Initiate the basics of the network
        self.X = X
        self.target = target
        self.activation_function = activation_function
        self.cost_function = Cost_Functions(cost_function, self.target)
        
        
        if weights or biases is None:
            self.construct_network_from_scratch = True
            self.construct_layer = self.construct_layer_from_scratch
        else:
            self.construct_network_from_scratch = False
            self.construct_layer = self.construct_layer_from_previous_training
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
                f"The activation function is {self.activation_function}.")

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
                n_inputs = self.X.shape[1]
            else:
                n_inputs = self.n_hidden_nodes[i - 1]

            if self.construct_network_from_scratch:
                hidden_layers.append(self.construct_layer(n_inputs=n_inputs,
                                                          n_hidden_nodes=self.n_hidden_nodes[i],
                                                          activation_function=self.activation_function))
            else:
                hidden_layers.append(self.construct_layer(n_inputs=n_inputs,
                                                          n_hidden_nodes=self.n_hidden_nodes[i],
                                                          activation_function=self.activation_function,
                                                          layer_indx=i))

        return hidden_layers

    def initiate_output_layer(self):
        """
        Initiates the output layer of the network.
        :return: Output layer of the network
        """
        if self.construct_network_from_scratch:
            return self.construct_layer(n_inputs=self.hidden_layers[-1].n_nodes,
                                        n_hidden_nodes=self.n_outputs,
                                        activation_function='sigmoid')
        else:
            return self.construct_layer(n_inputs=self.hidden_layers[-1].n_nodes,
                                        n_hidden_nodes=self.n_outputs,
                                        activation_function='sigmoid',
                                        layer_indx=-1)

    def construct_layer_from_scratch(self, n_inputs, n_hidden_nodes, activation_function):
        """
        Constructs a layer without weights and biases.
        :param n_inputs: Number of inputs to the layer
        :param n_hidden_nodes: Number of nodes in the layer
        :param activation_function: Activation function of the layer
        :return: Layer with random weights and biases set to 0.01
        """
        return Dense_Layer(n_inputs=n_inputs,
                           n_nodes=n_hidden_nodes,
                           activation_function=activation_function)

    def construct_layer_from_previous_training(self, n_inputs, n_hidden_nodes, activation_function, layer_indx):
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
                           activation_function=activation_function,
                           weights=self.weights[layer_indx],
                           biases=self.biases[layer_indx])
    
    def classify(self):
        
        self.output_layer.output = [1 if output >= 0.5 else 0 for output in self.output_layer.output]

    def feed_forward(self):
        for i in range(self.n_hidden_layers):
            
            if i == 0:
                input_value = self.X
            else:
                input_value = self.hidden_layers[i-1].output
            
            self.hidden_layers[i].forward_propagation(input_value)
        
        self.output_layer.forward_propagation(self.hidden_layers[-1].output)
        
        if self.classification_problem:
            self.classify()

        #self.output = self.activation_function(jnp.dot(self.X, self.weights) + self.biases)
        #pass
        
        
    def feed_backward(self):
        
        self.output_layer.backward_propagation(learning_rate=self.learning_rate)
        
        for i in reversed(range(self.n_hidden_layers)):
            
            layer = self.hidden_layers[i]
            
            if i == (self.n_hidden_layers - 1):
                delta = self.output_layer.delta
                
            else:
                delta=self.hidden_layers[i+1].delta
            
            layer.backward_propagation(delta=delta, 
                                       learning_rate=self.learning_rate)

        
            
            
            
        
            
        
        pass
        # dz_dW = self.hidden_layers[-1].output
        # da_dz = self.output_layer.grad_activation_function(x=self.output_layer.output_pre_activation)
        # dC_da = (self.output_layer.output - self.target)/(self.output_layer.output(1 - self.output_layer.output))
        
        
