import numpy as np
import tensorflow as tf
from random import shuffle
# np.random.seed(10)

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
        self.weights = [None]
        self.biases = [None]
        for l in range(1, self.L):
            self.biases.append(np.random.randn(self.topology[l], 1))
            self.weights.append(np.random.randn(self.topology[l], self.topology[l-1]))
            
    def forwardprop(self, X):
        self.neuron_activations = [X]
        self.weighted_sums = [None]
        A = X
        for l in range(1, self.L):
            activation_function = self.activation_functions[l][0]
            W = self.weights[l]
            B = self.biases[l]
            Z = np.dot(W, A) + B
            A = activation_function(Z)
            self.neuron_activations.append(A)
            self.weighted_sums.append(Z)

    def sigmoid_activation(self, Z):
        return 1/(1+np.exp(-Z))

    def sigmoid_derivative(self, z):
        return self.sigmoid_activation(z)*(1-self.sigmoid_activation(z))
    
    def ReLU_activation(self, Z):
        return np.maximum(Z, 0)
    
    def ReLU_derivative(self, z):
        return 1 if z > 0 else 0
    
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

    def backprop(self, h, y):
        nabla_a = []
        nabla_aL = np.array([[self.cost_derivative(hi, yi)] for hi, yi in zip(h, y)])

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

        nabla_b, nabla_w, nabla_bj, nabla_wj = [None], [None], [], []
        for l in range(1, self.L):
            Z = self.weighted_sums[l]
            W = self.weights[l]
            A_prev = self.neuron_activations[l-1]
            nabla_bj = self.sigmoid_derivative(Z)*nabla_a[l]
            nabla_wj = np.dot(nabla_bj, A_prev.T)
            nabla_b.append(nabla_bj)
            nabla_w.append(nabla_wj)
        
        return (nabla_w, nabla_b)
    
    def predict(self, x):
        pass
    
    def evaluate(self):
        pass
    
    def train(self, activation_functions, X_train, y_train, X_val, y_val, epochs, mini_batch_size, learning_rate, flatten=False):

        self.activation_functions = [None if func == None else self.activation_functions[func] for func in activation_functions]

        for e in range(epochs):
            training_data = list(zip(X_train, y_train))
            shuffle(training_data)
            h = None
    
            mini_batches = []
            for mb in range(0, len(training_data), mini_batch_size):
                mini_batches.append(training_data[mb:mb+mini_batch_size])

            for mini_batch in mini_batches:
                for x, y in mini_batch:
                    self.forwardprop(x)
                    h = self.neuron_activations[self.L-1].T[0]
                    self.gradient_descent(epochs, mini_batch, learning_rate, h, y)

            hs = []
            for x, y in zip(X_train, y_train):
                self.forwardprop(x)
                h = self.neuron_activations[self.L-1].T[0]
                hs.append(h)
            predictions = np.array(hs).argmax(axis=1)
            target_values = y_train.argmax(axis=1)
            accuracy = sum([1 for i in range(len(hs)) if predictions[i] == target_values[i]])/len(hs)
            cost = self.cost_function(hs, y_train)

            print(f"Epoch {e} -> Cost: {cost}, Accuracy: {accuracy}")

# def flatten_img(img):
#     """Flattens one image"""
#     flattened_img = []
#     for row in img:
#         for col in row:
#             flattened_img.append([col])
#     return flattened_img

# def one_hot(data):
#     encoded_data = []
#     for num in data:
#         encoded_num = []
#         for i in range(max(data)+1):
#             if i == num:
#                 encoded_num.append(1)
#             else:
#                 encoded_num.append(0)
#         encoded_data.append(encoded_num)
#     return encoded_data

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