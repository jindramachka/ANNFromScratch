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
            print("Weights for layer l:")
            print(W)

            B = self.biases[l]
            print("Biases for layer l:")
            print(B)

            print("Activations for layer l-1:")
            print(A)
            Z = np.dot(W, A) + B
            print("Weighted sums for layer l")
            print(Z)

            A = activation_function(Z)
            print("Activations for layer l:")
            print(A)

            print()
            self.neuron_activations.append(A)
            self.weighted_sums.append(Z)

        print()
        print("All biases:")
        print(self.biases)
        print("All weights:")
        print(self.weights)
        print("Weighted sums of all neurons:")
        print(self.weighted_sums)
        print("Activations of all neurons: ")
        print(self.neuron_activations)

    def sigmoid_activation(self, Z):
        return 1/(1+np.exp(-Z))

    def sigmoid_derivative(self, z):
        return self.sigmoid_activation(z)*(1-self.sigmoid_activation(z))

    def cost_function(self, h, y):
        pass

    def cost_derivative(self, a):
        pass

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
        activation_gradient = []
        for l in range(self.L-1, -1, -1):
            p_d_prev = []
            for j in range(l):
                a = self.neuron_activations[j]
                z = self.weighted_sums[j]
                if j == self.L-1:
                    dCda = self.cost_derivative(a)
                    # a = self.neuron_activations[j]
                    # z = self.weighted_sums[j]
                    # dCdw = a*self.sigmoid_derivative()*self.cost_derivative(a)
                    # dCdb = self.sigmoid_derivative()*self.cost_derivative(a)
                else:
                    dCda = [self.weights*z for j in range(l)]
                p_d_prev.append(dCda)


net = Network((3, 2, 1))
net.forwardprop("sigmoid", [[3], 
                            [2], 
                            [5]])