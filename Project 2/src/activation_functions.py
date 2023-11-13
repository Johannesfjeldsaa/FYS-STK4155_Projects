import numpy as np
class Activation_Functions:
    def __init__(self, activation_function):
        self.func_name = activation_function
        self.activation_function, self.grad_activation_function = self.set_activation_function()

    def __str__(self):
        return self.func_name

    def set_activation_function(self):
        if self.func_name == 'sigmoid':
            return self.sigmoid, self.grad_sigmoid
        elif self.func_name == 'tanh':
            return self.tanh, self.grad_tanh
        elif self.func_name == 'ReLU':
            return self.ReLU, self.grad_ReLU
        elif self.func_name == 'Leaky ReLU':
            return self.Leaky_ReLU, self.grad_Leaky_ReLU
        elif self.func_name == "Linear":
            return self.linear, self.grad_linear
        else:
            raise ValueError('Activation function is not available.' 
                             'Expected: sigmoid, relu, leaky_relu, Linear, not {}' .format(self.func_name))

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Resources:
        - https://www.youtube.com/watch?v=Xvg00QnyaIY&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=30

        :param x: x-axis coordinate
        :return: sigmoid function value [0, 1]
        """
        return 1 / (1 + np.exp(-x))

    def grad_sigmoid(self, x=None, y=None):
        """
        Calculate the gradient of the sigmoid function. y is the output of the sigmoid function and the preferred
        method of calculating the gradient. x is also possible, but slower.

        Resources:
        - https://www.youtube.com/watch?v=7f7xnJoCj98

        :param x: x-axis value
        :param y: output of the sigmoid function
        :return: gradient of the sigmoid function
        """
        if y is not None:
            return y * (1 - y)
        elif x is not None:
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        else:
            raise ValueError('Either x or y must be given')

    def tanh(self, x):
        """
        Hyperbolic tangent (tanh) activation function.

        Resources:
        - https://www.youtube.com/watch?v=Xvg00QnyaIY&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=30

        :param x: x-axis coordinate
        :return: Tangent activation function value [-1, 1]
        """
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def grad_tanh(self, x):
        """
        Calculate the gradient of the tanh function.

        Resources:
        - https://www.youtube.com/watch?v=7f7xnJoCj98

        :param x: x-axis value
        :return: gradient of the tanh function
        """
        return 1 - self.tanh(x) ** 2

    def ReLU(self, x):
        """
        Rectified Linear Unit (ReLU) activation function.

        Resources:
        - https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
        - https://www.youtube.com/watch?v=Xvg00QnyaIY&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=30

        :param x: x-axis coordinate
        :return: function value [0, inf]
        """
        #return max(0.0, x)
        return np.maximum(0.0, x)

    def grad_ReLU(self, x):
        """
        Calculate the gradient of the ReLU function.

        Resources:
        - https://www.youtube.com/watch?v=P7_jFxTtJEo

        :param x: x-axis value
        :return: gradient of the ReLU function
        """
        #return 1.0 if x >= 0.0 else 0.0
        return np.where(x >= 0.0, 1.0, 0.0)

    def Leaky_ReLU(self, x):
        """
        Leaky Rectified Linear Unit (Leaky ReLU) activation function.

        Resources:
        - https://www.youtube.com/watch?v=Xvg00QnyaIY&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=30

        :param x: x-axis coordinate
        :return: function value
        """
        #return max(0.01 * x, x)
        return np.maximum(0.01 * x, x)

    def grad_Leaky_ReLU(self, x):
        """
        Calculate the gradient of the Leaky ReLU function.

        Resources:
        - https://www.youtube.com/watch?v=KTNqXwkLuM4

        :param x: x-axis value
        :return: gradient of the Leaky ReLU function
        """
        #return 1.0 if x >= 0.0 else 0.01
        return np.where(x >= 0.0, 1.0, 0.01)
    
    def linear(self, x):
        
        return x
    
    def grad_linear(self, x):
        
        return 1.0

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    activation_functions = ['sigmoid', 'tanh', 'ReLU', 'Leaky ReLU']

    for af in activation_functions:
        func = Activation_Functions(af)
        x = np.linspace(-10, 10, 100)
        y = [func.activation_function(x_i) for x_i in x]
        y_grad = [func.grad_activation_function(x_i) for x_i in x]

        plt.hlines(0, -10, 10, color='black', linestyles='--')
        plt.plot(x, y, label=af)
        plt.plot(x, y_grad, label='Gradient of {}'.format(af))
        plt.grid()
        plt.legend()
        plt.show()