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

    def stochastic_gradient_descent(self, activation_functions, X_train, y_train, epochs, mini_batch_size, learning_rate):

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
                    self.forward_propagation(x)
                    h = self.neuron_activations[self.L-1].T[0]
                    nabla_wb_x = self.backward_propagation(h, y)
                    nabla_w_x = nabla_wb_x[0]
                    nabla_b_x = nabla_wb_x[1]

                    for l in range(1, self.L):
                        self.weights[l] -= learning_rate * nabla_w_x[l]/len(mini_batch)
                        self.biases[l] -= learning_rate * nabla_b_x[l]/len(mini_batch)

            hs = []
            for x, y in zip(X_train, y_train):
                self.forward_propagation(x)
                h = self.neuron_activations[self.L-1].T[0]
                hs.append(h)
            predictions = np.array(hs).argmax(axis=1)
            target_values = y_train.argmax(axis=1)
            accuracy = sum([1 for i in range(len(hs)) if predictions[i] == target_values[i]])/len(hs)
            cost = self.cost_function(hs, y_train)

            print(f"Epoch {e} -> Cost: {cost}, Accuracy: {accuracy}")
            
    def forward_propagation(self, x):
        self.neuron_activations = [x]
        self.weighted_sums = [None]
        current_layer_A = x
        for l in range(1, self.L):
            activation_function = self.activation_functions[l][0]

            current_layer_W = self.weights[l]
            previous_layer_A = current_layer_A
            current_layer_B = self.biases[l]

            current_layer_Z = np.dot(current_layer_W, previous_layer_A) + current_layer_B
            current_layer_A = activation_function(current_layer_Z)

            self.neuron_activations.append(current_layer_A)
            self.weighted_sums.append(current_layer_Z)

    def backward_propagation(self, h, y):
        nabla_a = []
        last_layer_nabla_a = np.array([[self.cost_derivative(hi, yi)] for hi, yi in zip(h, y)])

        nabla_a.append(last_layer_nabla_a)
        current_layer_nabla_a = last_layer_nabla_a

        for l in range(self.L-1, 0, -1):
            previous_layer_nabla_a = []
            current_Z = self.weighted_sums[l]
            current_W = self.weights[l]
            previous_layer_nabla_a = np.dot(current_W.T, self.sigmoid_derivative(current_Z) * current_layer_nabla_a)
            nabla_a.append(previous_layer_nabla_a)
            current_layer_nabla_a = previous_layer_nabla_a
        nabla_a.reverse()

        nabla_b, nabla_w, current_layer_nabla_b, current_layer_nabla_w = [None], [None], [], []
        for l in range(1, self.L):
            current_layer_Z = self.weighted_sums[l]
            A_prev = self.neuron_activations[l-1]
            current_layer_nabla_b = self.sigmoid_derivative(current_layer_Z)*nabla_a[l]
            current_layer_nabla_w = np.dot(current_layer_nabla_b, A_prev.T)
            nabla_b.append(current_layer_nabla_b)
            nabla_w.append(current_layer_nabla_w)
        
        return (nabla_w, nabla_b)
    
    def predict(self):
        pass
    
    def evaluate(self):
        pass

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
        return 2*(hi-yi)

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
net.stochastic_gradient_descent((None, "sigmoid", "sigmoid", "sigmoid"), X_train_flattened, y_train_encoded, 30, 10, 0.5)