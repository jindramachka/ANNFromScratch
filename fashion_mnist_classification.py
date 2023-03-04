from network import Network
import numpy as np
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train_normalized = X_train / 255
X_train_flattened = X_train_normalized.reshape(X_train_normalized.shape[0], 784, 1)
X_test_normalized = X_test / 255
X_test_flattened = X_test_normalized.reshape(X_test_normalized.shape[0], 784, 1)

y_train_encoded = np.zeros((y_train.size, y_train.max()+1))
y_train_encoded[np.arange(y_train.size), y_train] = 1
# y_train_encoded = y_train_encoded.reshape(y_train.shape[0], 10, 1)
y_test_encoded = np.zeros((y_test.size, y_test.max()+1))
y_test_encoded[np.arange(y_test.size), y_test] = 1


net = Network((784, 128, 64, 10))
net.stochastic_gradient_descent((None, "sigmoid", "sigmoid", "sigmoid"), X_train_flattened, y_train_encoded, 5, 10, 0.5)
net.evaluate(X_test_flattened, y_test_encoded)