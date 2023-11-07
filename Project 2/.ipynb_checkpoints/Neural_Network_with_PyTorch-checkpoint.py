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
                 activation_function_hidden_layers, activation_function_output_layer):

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

        self.hidden_layers = self.initiate_hidden_layers()
        self.output_layer = self.initiate_output_layer()

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
        else:
            raise ValueError('activation_function must be one of: sigmoid, tanh, ReLU, Leaky ReLU not {}'.format(activation_function))


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

        hidden_layers = []
        for i in range(self.n_hidden_layers):
            if i == 0:
                n_inputs = self.n_inputs
            else:
                n_inputs = self.n_hidden_nodes[i - 1]

            hidden_layers.append([nn.Linear(n_inputs, self.n_hidden_nodes[i]), self.activation_function_hidden_layers])

        return hidden_layers

    def initiate_output_layer(self):
        """
        Initiates the output layer of the network.
        :return: Output layer of the network
        """
        return [nn.Linear(self.n_hidden_nodes[-1], self.n_outputs), self.activation_function_output_layer]

    def feed_forward(self, X):
        """
        Feed forward through the network.
        :return: output of the network prior to the activation function of the output layer
        """

        input = X
        for layer in self.hidden_layers:
            linear_output = layer[0](input)
            non_linear_output = layer[1](linear_output)
            # set non_linear_output as input for next layer
            input = non_linear_output

        # output layer
        linear_output = self.output_layer[0](input)
        output = self.output_layer[1](linear_output)

        return output


