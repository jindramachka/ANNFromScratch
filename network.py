import numpy as np

class Network:
    def __init__(self, layers):
        self.activation_functions = {"sigmoid": self.sigmoid_activation}
        self.weights = [None]
        self.biases = [None]
        self.L = len(layers)
        for l in range(1, self.L):
            self.biases.append(np.random.randn(layers[l], 1))
            self.weights.append(np.random.randn(layers[l], layers[l-1]))

        self.weights = [None, 
                       [[3,  2],
                        [6, 2],
                        [3, 4]],
                        [[4, 2, 3]]]
        self.biases = [None,
                       [[3],
                        [2],
                        [4]], 
                        [[4]]]

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
            print(A)

            print()
            self.neuron_activations.append(A)
            self.weighted_sums.append(Z)

        print()
        print(self.neuron_activations)
        print(self.weighted_sums)

    def sigmoid_activation(self, Z):
        return 1/(1+np.exp(-Z))

net = Network((2, 3, 1))
net.forwardprop("sigmoid", [[3], [2]])