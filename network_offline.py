import numpy as np
import tensorflow as tf
# np.random.seed(5)

class Network:
    def __init__(self, topology):
        self.activation_functions = {"sigmoid": (self.sigmoid_activation,
                                                 self.sigmoid_derivative),
                                     "relu": (self.ReLU_activation,
                                              self.ReLU_derivative),
                                     "softmax": (self.softmax_activation,
                                                 self.softmax_derivative)}
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
        
        # print(self.weights)
        # print()
        # print(self.biases)
        # print()

    def forwardprop(self, X):
        self.neuron_activations = [X]
        self.weighted_sums = [0]
        A = X
        for l in range(1, self.L):
            activation_function = self.activation_functions[l][0]
            W = self.weights[l]
            # print("Weights for layer l:")
            # print(W)
            B = self.biases[l]
            # print("Biases for layer l:")
            # print(B)
            # print("Activations for layer l-1:")
            # print(A)
            Z = np.dot(W, A) + B
            # print("Weighted sums for layer l")
            # print(Z)
            A = activation_function(Z)
            # print("Activations for layer l:")
            # print(A)
            # print()
            self.neuron_activations.append(A)
            self.weighted_sums.append(Z)
        # print()
        # print("All biases:")
        # print(self.biases)
        # print("All weights:")
        # print(self.weights)
        # print("Weighted sums of all neurons:")
        # print(self.weighted_sums)
        # print("Activations of all neurons: ")
        # print(self.neuron_activations)

    def sigmoid_activation(self, Z):
        return 1/(1+np.exp(-Z))

    def sigmoid_derivative(self, z):
        return self.sigmoid_activation(z)*(1-self.sigmoid_activation(z))
    
    def ReLU_activation(self, Z):
        return np.maximum(Z, 0)
    
    def ReLU_derivative(self, z):
        return z > 0
    
    def softmax_activation(self, Z):
        return np.exp(Z) / sum(np.exp(Z))
    
    def softmax_derivative(self, z):
        return 

    def cost_function(self, h, y):
        # print(h[:3])
        # print(y[:3])
        # for i in range(5):
        #     print()
        #     print(y[i])
        #     print(h[i])
        #     print(y[i]-h[i])
        return sum([np.dot(y[i]-h[i], y[i]-h[i]) for i in range(len(h))])/len(h)

    def cost_derivative(self, hi, yi):
        # return 2/len(hi) * np.dot((yi - hi), -hi) 
        return hi-yi

    def gradient_descent(self, epochs, learning_parameter, h, y_train):
            # cost = self.cost_function([1, 1], [1, 1])
        cost = self.cost_function(h, y_train)

        print(f"Cost {cost}")
        # print(h[:3])
        # print(y_train[:3])
        nabla_wb = self.backprop(h, y_train)
        nabla_w = nabla_wb[0]
        nabla_b = nabla_wb[1]

        for l in range(1, self.L):
            self.weights[l] -= learning_parameter * nabla_w[l]
            self.biases[l] -= learning_parameter * nabla_b[l]

        # print("Final weights: ")
        # print(self.weights)
        # print("Final biases: ")
        # print(self.biases)
        # print("Final activations: ")
        # print(self.neuron_activations)
        

    def backprop(self, h, y_train):
        nabla_a = []
        nabla_aL = [[self.cost_derivative(hi, yi)] for hi, yi in zip(h, y_train)]

        # Calculation of nabla_a given that we know the elements of nabla_a for the last layer
        # print()
        nabla_a.append(nabla_aL)
        nabla_aj = nabla_aL

        # print(self.neuron_activations[self.L-1])
        # print(self.neuron_activations[self.L-1].T)
        # for hi in self.neuron_activations[self.L-1].T:
            # print(hi)
            # print(self.cost_derivative(hi, 1))

        # print(nabla_aL)

        for l in range(self.L-1, 0, -1):

            # print(l-1)
            # print(self.neuron_activations[l-1])
            # print(l)
            # print(self.neuron_activations[l])

            activation_derivative = self.activation_functions[l][1]
            nabla_ak = [] # nabla_a for layer l-1
            for k in range(len(self.neuron_activations[l-1])):   
                # print(f"k: {k}")
                pdC_pdak = 0
                for j in range(len(self.neuron_activations[l])):
                    zj = self.weighted_sums[l][j][0]
                    wjk = self.weights[l][j][k]
                    pdC_pdaj = nabla_aj[j][0]
                    pdC_pdak += wjk*activation_derivative(zj)*pdC_pdaj

                    # print(f"k: {k}")
                    # print(self.neuron_activations[l-1][k])
                    # print(f"j: {j}")
                    # print(self.neuron_activations[l][j])
                    # print(f"zj: {zj}")
                    # print(f"wjk: {wjk}")

                nabla_ak.append([pdC_pdak])
            
            # print()
            # print(nabla_aj)
            # print(nabla_ak)

            nabla_a.append(nabla_ak)
            nabla_aj = nabla_ak # nabla_a for layer l
        nabla_a.reverse()

        # print()
        # print("nabla_a:")
        # print(nabla_a)

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

        # print("nabla_b:")
        # print(nabla_b)
        # print("nabla_w:")
        # print(nabla_w)

        return nabla_w, nabla_b
    
    def predict(self, x):
        pass
    
    def train(self, activation_functions, X_train, y_train, epochs, learning_rate, flatten=False):
        self.activation_functions = [None if func == None else self.activation_functions[func] for func in activation_functions]
        for e in range(epochs):
            h = []
            print(f"Epoch {e}")
            for x in X_train:
                if flatten:
                    x = flatten_img(x)
                self.forwardprop(x)
                h.append(self.neuron_activations[self.L-1].T[0])
            h = np.array(h)
            self.gradient_descent(epochs, learning_rate, h, y_train)
            predictions = h.argmax(axis=1)
            # print(predictions[:3])
            targets = y_train.argmax(axis=1)
            # print(targets[:3])
            right = 0
            for i in range(len(h)):
                if predictions[i] == targets[i]:
                    right += 1
            print(right)
            accuracy = right/len(h)
            print(accuracy)

def flatten_img(img):
    flattened_img = []
    for row in img:
        for col in row:
            flattened_img.append([col])
    return flattened_img

def one_hot(data):
    encoded_data = []
    for num in data:
        encoded_num = []
        for i in range(max(data)+1):
            if i == num:
                encoded_num.append(1)
            else:
                encoded_num.append(0)
        encoded_data.append(encoded_num)
    return encoded_data

# fashion_mnist = tf.keras.datasets.fashion_mnist
# (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train_normalized = X_train / 255
X_train_flattened = X_train_normalized.reshape(X_train_normalized.shape[0], 784, 1)
print(X_train_flattened)

y_train_encoded = np.zeros((y_train.size, y_train.max()+1))
y_train_encoded[np.arange(y_train.size), y_train] = 1
# y_train_encoded = y_train_encoded.reshape(y_train.shape[0], 10, 1)
print(y_train_encoded)

net = Network((784, 30, 10))
net.train((None, "sigmoid", "sigmoid"), X_train_flattened, y_train_encoded, 500, 3)