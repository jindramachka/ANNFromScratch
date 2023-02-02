import numpy as np
import sympy as sp

# Cost function

# 1 - columns as training examples, rows as neurons
h = np.array([[2, 3, 4],  # 3 training outputs of output neuron 1 (for training examples 1, 2 and 3)
              [4, 1, 3],  # 3 training outputs of output neuron 2
              [5, 3, 1],  # 3 training outputs of output neuron 3
              [2, 4, 1]]) # 3 training outputs of output neuron 4
y = np.array([[3, 1, 4],  # 3 desired outputs of output neuron 1
              [3, 5, 1],  # 3 desired outputs of output neuron 2
              [2, 2, 4],  # 3 desired outputs of output neuron 3
              [5, 2, 3]]) # 3 desired outputs of output neuron 4
# OR
# 2 - columns as neurons, rows as training examples (the correct way)
h = np.array([[2, 3, 4],  # 1st training output for output neurons 1, 2 and 3
              [4, 1, 3],  # 2nd training output for output neurons 1, 2 and 3
              [5, 3, 1],  # 3rd training output for output neurons 1, 2 and 3
              [2, 4, 1]]) # 4th training output for output neurons 1, 2 and 3
y = np.array([[3, 1, 4],  # 1st desired output for output neurons 1, 2 and 3
              [3, 5, 1],  # 2nd desired output for output neurons 1, 2 and 3
              [2, 2, 4],  # 3rd desired output for output neurons 1, 2 and 3
              [5, 2, 3]]) # 4th desired output for output neurons 1, 2 and 3
c=[]
for i in range(len(h)):
    c.append((y[i]-h[i])**2)
print(c)
print([(y[i]-h[i])**2 for i in range(len(h))]) 
print([sum(j) for j in [(y[i]-h[i])**2 for i in range(len(h))]]) # A list of four costs - 1 for each training example
c = sum([sum(j) for j in [(y[i]-h[i])**2 for i in range(len(h))]])/len(h)
print(c)

# print(np.subtract(y, h))
# print(np.square(np.subtract(y, h)))
# print(np.square(np.subtract(y, h)).mean())

# Derivative of the cost
# x1, y1, z1, a1, b1, c1 = sp.symbols("x1 y1 y1 a1 b1 c1")
# dc = sp.Derivative((sp.Vector(([x1, y1, z1]) - np.array([a1, b1, c1]))**2)/5, x1)
# print(f"{dc}")
# print(f"{dc.doit()}")

# 2 implemented
print()
for i in range(len(h)):
    e = y[i]-h[i]
    print(np.dot(e, e))
c = sum([np.dot(y[i]-h[i], y[i]-h[i]) for i in range(len(h))])/len(h)
print(c)

# Partial ifferentiation implemented
print()
h_for_diff = h.T
print(h_for_diff)
y_for_diff = y.T
print(y_for_diff)
nabla_h = []
for i in range(len(h_for_diff)):
    hi = h_for_diff[i]
    yi = y_for_diff[i]
    pdC_pdhi = 2/len(hi) * np.dot((hi-yi), hi)
    nabla_h.append(pdC_pdhi)
print(nabla_h)

a = [1,2,3,4]
b=  [5,6,7,8]
print(np.dot(a, b))

print(h.argmax(axis=1))

# softmax
a = np.array([[6], [8], [8], [1], [4], [4], [6]])
print(np.exp(a) / sum(np.exp(a)))
def softmax(z):
    return np.exp(a) / sum(np.exp(a))
# derivative
s = softmax(a).reshape(-1,1)
print(np.diagflat(s) - np.dot(s, s.T))

# sigmoid test
print()
def sigmoid_activation(Z):
    return 1/(1+np.exp(-Z))
def sigmoid_derivative(z):
    return sigmoid_activation(z)*(1-sigmoid_activation(z))
print(sigmoid_activation(a))
print(sigmoid_derivative(a[0]))

def sigmoid_activation(Z):
    return 1/(1+np.exp(-Z))

def sigmoid_derivative(z):
    return sigmoid_activation(z)*(1-sigmoid_activation(z))

def ReLU_activation(Z):
    return np.maximum(Z, 0)

def ReLU_derivative(z):
    return z > 0

def softmax_activation(Z):
    return np.exp(Z) / sum(np.exp(Z))

def softmax_derivative(z):
    s = softmax_activation(z).reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

activation_functions = {"sigmoid": (sigmoid_activation,
                                                 sigmoid_derivative),
                        "relu": (ReLU_activation,
                                ReLU_derivative),
                        "softmax": (softmax_activation,
                                    softmax_derivative)}
print()
print(a)
print(activation_functions["softmax"][1]([23]))