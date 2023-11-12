import torch
import torch.nn as nn



class Neural_Network_PyTorch(nn.Module):
    """
    Neural Network class for classification using the PyTorch framework.

    Resources:
    - Project 2/NeuralNetwork.py
        - Project 2/Test NeuralNetwork.ipynb
    - https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
    """
    def __init__(self, n_inputs, n_hidden_layers, n_hidden_nodes, n_outputs,
                 activation_function_hidden_layers='RelU', activation_function_output_layer='sigmoid'):

        super(Neural_Network_PyTorch, self).__init__()

        # Initiate the number of inputs
        self.n_inputs = n_inputs

        # Initiate the activation functions
        self.activation_function_hidden_layers = self.set_activation_function(activation_function_hidden_layers)
        self.activation_function_output_layer = self.set_activation_function(activation_function_output_layer)

        # Initiate the hidden nodes and layers
        self.n_hidden_layers = n_hidden_layers
        if isinstance(n_hidden_nodes, int) or isinstance(n_hidden_nodes, float):
            n_hidden_nodes = [int(n_hidden_nodes)] * self.n_hidden_layers
        elif len(n_hidden_nodes) != self.n_hidden_layers:
            raise ValueError('n_hidden_nodes must be integer or a list of same length as number of hidden layers')

        self.n_hidden_nodes = n_hidden_nodes
        self.n_outputs = n_outputs

        self.initiate_hidden_layers()
        self.initiate_output_layer()

    #def __str__(self):
     #   return (f"Neural Network with {self.n_hidden_layers} hidden layers and {self.n_hidden_nodes} nodes per layer. "
      #          f"The activation function is {self.activation_function_hidden_layers} for the hidden layers and "
       #         f"{self.activation_function_output_layer} for the output layer.")


    def set_activation_function(self, activation_function):
        if activation_function == 'sigmoid':
            return nn.Sigmoid()
        elif activation_function == 'tanh':
            return nn.Tanh()
        elif activation_function == 'ReLU':
            return nn.ReLU()
        elif activation_function == 'Leaky ReLU':
            return nn.LeakyReLU()
        elif activation_function is None:
            def dummy(x):
                return x
            return dummy
        else:
            raise ValueError('activation_function must be one of: '
                             'sigmoid, tanh, ReLU, Leaky ReLU not {}. \n'.format(activation_function)
                             + 'If you want to use no activation function, set activation_function to None.')


    def initiate_hidden_layers(self):
        """
        Initiates hidden layers of the network. If the nodes per layer is given as an integer, all hidden layers
        will have the same number of nodes, then we create a list of same length as n_hidden_layers.
        If the nodes per layer is given as a list it has to have same number of elements as n_hidden_layers.

        The first layer isi initated with n_inputs the same length as the input layer.
        The rest of the layers are initiated with n_inputs the same length as the previous hidden layer.
        The output layer is initialized with n_inputs the same length as the last hidden layer.

        :return: Hidden layers of the network
        """
        for i in range(self.n_hidden_layers):
            if i == 0:
                n_inputs = self.n_inputs
            else:
                n_inputs = self.n_hidden_nodes[i - 1]
            
            setattr(self, f'hidden{i}', nn.Linear(n_inputs, self.n_hidden_nodes[i]))
            setattr(self, f'activation{i}', self.activation_function_hidden_layers)


    def initiate_output_layer(self):
        """
        Initiates the output layer of the network.
        :return: Output layer of the network
        """
        self.linear_output = nn.Linear(self.n_hidden_nodes[-1], self.n_outputs)
        print(self.activation_function_output_layer)
        self.activated_output = self.activation_function_output_layer

    def feed_forward(self, X):
        """
        Feed forward through the network.
        :return: output of the network prior to the activation function of the output layer
        """

        input = X
        for i in range(self.n_hidden_layers):
            
            linear_output = getattr(self, f'hidden{i}')(input)
            non_linear_output = getattr(self, f'activation{i}')(linear_output)
            
            input = non_linear_output

        # output layer
        linear_output = self.linear_output(input)
        output = self.activated_output(linear_output)

        return output

    def classify(self, unclassified_output):
        """
        Classifies the output of the network.
        :param unclassified_output: output of the network prior to the activation function of the output layer
        :return: classified output of the network
        """
        return [1 if output >= 0.5 else 0 for output in unclassified_output]

