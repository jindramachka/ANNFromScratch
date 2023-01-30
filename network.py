import numpy as np
np.random.seed(5)

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
        self.weighted_sums = [0]
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

        self.gradient_descent(3, 0.5)

    def sigmoid_activation(self, Z):
        return 1/(1+np.exp(-Z))

    def sigmoid_derivative(self, z):
        # return self.sigmoid_activation(z)*(1-self.sigmoid_activation(z))
        return z*(1-z) # The self.sigmoid_activation(z) has already been applied on z during forward propagation

    def cost_function(self, h, y):
        return sum([np.dot(y[i]-h[i], y[i]-h[i]) for i in range(len(h))])/len(h)

    def cost_derivative(self, hi, yi):
        return 2/len(hi) * np.dot((hi - yi), hi) 

    def gradient_descent(self, epochs, learning_parameter):
        for e in range(epochs):
            cost = self.cost_function([1, 1], [1, 1])
            nabla_wb = self.backprop()
            nabla_w = nabla_wb[0]
            nabla_b = nabla_wb[1]

            for l in range(1, self.L):
                self.weights[l] -= learning_parameter * nabla_w[l]
                self.biases[l] -= learning_parameter * nabla_b[l]

        print("Final weights: ")
        print(self.weights)
        print("Final biases: ")
        print(self.biases)

    def backprop(self):
        nabla_a = []
        nabla_aL = [[self.cost_derivative(hi, 1)] for hi in self.neuron_activations[self.L-1].T]

        # Calculation of nabla_a given that we know the elements of nabla_a for the last layer
        print()
        nabla_a.append(nabla_aL)
        nabla_aj = nabla_aL
        for l in range(self.L-1, 0, -1):

            print(l-1)
            print(self.neuron_activations[l-1])
            print(l)
            print(self.neuron_activations[l])

            nabla_ak = [] # nabla_a for layer l-1
            for k in range(len(self.neuron_activations[l-1])):   
                # print(f"k: {k}")
                pdC_pdak = 0
                for j in range(len(self.neuron_activations[l])):
                    zj = self.weighted_sums[l][j][0]
                    wjk = self.weights[l][j][k]
                    pdC_pdaj = nabla_aj[j][0]
                    pdC_pdak += wjk*self.sigmoid_derivative(zj)*pdC_pdaj

                    print(f"k: {k}")
                    print(self.neuron_activations[l-1][k])
                    print(f"j: {j}")
                    print(self.neuron_activations[l][j])
                    print(f"zj: {zj}")
                    print(f"wjk: {wjk}")

                nabla_ak.append([pdC_pdak])
            
            print()
            print(nabla_aj)
            print(nabla_ak)

            nabla_a.append(nabla_ak)
            nabla_aj = nabla_ak # nabla_a for layer l
        nabla_a.reverse()

        print()
        print("nabla_a:")
        print(nabla_a)

        nabla_b, nabla_w, nabla_bj, nabla_wj = [None], [None], [], []
        for l in range(1, self.L):
            nabla_bj = []
            nabla_wj = []
            for j in (range(len(self.neuron_activations[l]))):
                aj = self.neuron_activations[l][j][0]
                zj = self.weighted_sums[l][j][0]
                pdC_pdbj = self.sigmoid_derivative(zj)*nabla_a[l][j][0]
                pdC_pdwjk = aj*pdC_pdbj
                nabla_bj.append([pdC_pdbj])
                nabla_wj.append([pdC_pdwjk for k in range(len(self.weights[l][j]))])
            nabla_b.append(np.array(nabla_bj))
            nabla_w.append(np.array(nabla_wj))

        print("nabla_b:")
        print(nabla_b)
        print("nabla_w:")
        print(nabla_w)

        return nabla_w, nabla_b

net = Network((3, 4, 4, 1))
net.forwardprop("sigmoid", [[3], 
                            [2], 
                            [5]])