import numpy as np
import tensorflow as tf
from random import shuffle
np.random.seed(10)

class Network:
    def __init__(self, topology):
        self.activation_functions = {"sigmoid": (self.sigmoid_activation,
                                                 self.sigmoid_derivative),
                                     "relu": (self.ReLU_activation,
                                              self.ReLU_derivative),
                                     "softmax": (self.softmax_activation,
                                                 self.softmax_derivative)}
        self.topology = topology
        self.L = len(self.topology)

        self.weights = [np.array([0])]
        self.biases = [np.array([0])]
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
        # print(X)
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
        return 
    
    def softmax_activation(self, Z):
        return np.exp(Z) / sum(np.exp(Z))
    
    def softmax_derivative(self, z):
        return 

    def cost_function(self, h, y):
        return sum([np.dot(y[i]-h[i], y[i]-h[i]) for i in range(len(h))])/len(h)

    def cost_derivative(self, hi, yi):
        # return 2/len(hi) * np.dot((yi - hi), -hi) 
        # return 2*(yi-hi)*-hi
        return hi-yi

    def gradient_descent(self, epochs, mini_batch, learning_parameter, h , y):
        nabla_wb = self.backprop(h, y)
        nabla_w = nabla_wb[0]
        nabla_b = nabla_wb[1]

        for l in range(1, self.L):
            self.weights[l] -= learning_parameter/len(mini_batch) * nabla_w[l]
            self.biases[l] -= learning_parameter/len(mini_batch) * nabla_b[l]

        # print("Final weights: ")
        # print(self.weights)
        # print("Final biases: ")
        # print(self.biases)
        # print("Final activations: ")
        # print(self.neuron_activations)

    def backprop(self, h, y):
        # print("now")
        nabla_a = []
        nabla_aL = np.array([[self.cost_derivative(hi, yi)] for hi, yi in zip(h, y)])

        # Calculation of nabla_a given that we know the elements of nabla_a for the last layer
        # print()
        nabla_a.append(nabla_aL)
        nabla_aj = nabla_aL

        for l in range(self.L-1, 0, -1):
            nabla_ak = []
            Z = self.weighted_sums[l]
            W = self.weights[l]
            nabla_ak = np.dot(W.T, self.sigmoid_derivative(Z) * nabla_aj)
            nabla_a.append(nabla_ak)
            nabla_aj = nabla_ak
        nabla_a.reverse()

        nabla_b, nabla_w, nabla_bj, nabla_wj = [0], [0], [], []
        for l in range(1, self.L):
            A = self.neuron_activations[l]
            Z = self.weighted_sums[l]
            W = self.weights[l]
            A_prev = self.neuron_activations[l-1]
            nabla_bj = self.sigmoid_derivative(Z)*nabla_a[l]
            # nabla_wj = A*nabla_bj
            nabla_wj = np.dot(nabla_bj, A_prev.T)
            nabla_b.append(nabla_bj)
            nabla_w.append(nabla_wj)
        
        # print(self.weights)

        # print("nabla_b:")
        # print(nabla_b)
        # print("nabla_w:")
        # print(nabla_w)
        # print(nabla_w[-1][-1].shape)
        # print(self.weights[-1][-1].shape)
        # print(nabla_b[:3])
        # print(nabla_b)
        return (nabla_w, nabla_b)
    
    def predict(self, x):
        pass
    
    def train(self, activation_functions, X_train, y_train, X_test, y_test, epochs, mini_batch_size, learning_rate, flatten=False):
        self.activation_functions = [None if func == None else self.activation_functions[func] for func in activation_functions]

        for e in range(epochs):
            training_data = list(zip(X_train, y_train))
            # shuffle(training_data)
            h = None
            hs = []
            print(f"Epoch {e}")


            mini_batches = []
            for mb in range(0, len(training_data), mini_batch_size):
                mini_batches.append(training_data[mb:mb+mini_batch_size])

            for mini_batch in mini_batches:
                # print(mini_batch)
                # print(len(mini_batch))
                i=0
                for x, y in mini_batch:

                    i+=1
                    if i % 10000 == 0:
                        print(i)
                    if flatten:
                        x = flatten_img(x)
                    self.forwardprop(x)

                    h = self.neuron_activations[self.L-1].T[0]
                    hs.append(h)
                    self.gradient_descent(epochs, mini_batch, learning_rate, h, y)
                # self.gradient_descent(epochs, mini_batch, learning_rate)

                # h.append(self.neuron_activations[self.L-1].T[0])
            # h = np.array(h)
            # self.gradient_descent(epochs, learning_rate, h, y_train)

            hs=[]
            for x, y in zip(X_test, y_test):

                self.forwardprop(x)

                h = self.neuron_activations[self.L-1].T[0]
                hs.append(h)
            hs = np.array(hs)
            cost = self.cost_function(hs, y_test)
            print(f"Cost {cost}")
            # print(hs[:3])
            predictions = hs.argmax(axis=1)
            # print(predictions[:3])
            targets = y_test.argmax(axis=1)
            # print(targets[:3])
            right = 0
            for i in range(len(hs)):
                if predictions[i] == targets[i]:
                    right += 1
            print(right)
            accuracy = right/len(hs)
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

fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
# mnist = tf.keras.datasets.mnist
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train_normalized = X_train / 255
X_train_flattened = X_train_normalized.reshape(X_train_normalized.shape[0], 784, 1)
X_test_normalized = X_test / 255
X_test_flattened = X_test_normalized.reshape(X_test_normalized.shape[0], 784, 1)

y_train_encoded = np.zeros((y_train.size, y_train.max()+1))
y_train_encoded[np.arange(y_train.size), y_train] = 1
# y_train_encoded = y_train_encoded.reshape(y_train.shape[0], 10, 1)
y_test_encoded = np.zeros((y_test.size, y_test.max()+1))
y_test_encoded[np.arange(y_test.size), y_test] = 1

net = Network((784, 16, 16, 10))

# net.train((None, "sigmoid","sigmoid"), X_train_flattened[:1], y_train_encoded[:1], X_test_flattened[:1], y_test_encoded[:1], 1, 1, 0.5)
net.train((None, "sigmoid", "sigmoid", "sigmoid"), X_train_flattened, y_train_encoded, X_test_flattened, y_test_encoded, 30, 10, 0.5)