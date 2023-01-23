import numpy as np

class Network:
    def __init__(self, topology):
        self.activation_functions = {"sigmoid": self.sigmoid_activation}
        self.weights = [None]
        self.biases = [None]
        self.topology = topology
        self.L = len(self.topology)
        for l in range(1, self.L):
            self.biases.append(np.random.randn(self.topology[l], 1))
            self.weights.append(np.random.randn(self.topology[l], self.topology[l-1]))

        # self.weights = [None, 
        #                 [[3,  2],
        #                  [6, 2],
        #                  [3, 4]],
        #                 [[4, 2, 3]]]
        # self.biases = [None,
        #                [[3],
        #                 [2],
        #                 [4]], 
        #                [[4]]]

        print(self.weights)
        print()
        print(self.biases)
        print()
    def forwardprop(self, activation_function, X):
        activation_function = self.activation_functions[activation_function]
        self.neuron_activations = [X]
        self.weighted_sums = [None]
        A = X
        for l in range(1, self.L):
            W = self.weights[l]
            print(W)
            B = self.biases[l]
            print(B)
            print(A)
            Z = np.dot(W, A) + B
            print(Z)
            A = activation_function(Z)

    def sigmoid_activation(self, Z):
        return 1/(1+np.exp(-Z))

    def cost_function(self, h, y):
        return np.array([sum(j) for j in [(y[i]-h[i])**2 for i in range(len(h))]])/len(h)

    def gradient_descent(self, epochs, learning_parameter):
        for e in range(epochs):
            cost = self.cost_function()
            gradient = self.backprop(cost)
            weight_gradient = gradient[0]
            bias_gradient = gradient[1]


            for l in range(self.L):
                self.weights[l] -= learning_parameter * weight_gradient[l]
                self.biases[l] -= learning_parameter * bias_gradient[l]

    def backprop(self, cost):
        for l in range(self.L):
            pass



net = Network((5, 10, 15, 5))
net.forwardprop("sigmoid", [[3], 
                            [2], 
                            [5],
                            [6],
                            [2]])