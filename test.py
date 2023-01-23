import numpy as np
import sympy as sp

# Cost function
h = np.array([[2, 3, 4],
              [4, 1, 3],
              [5, 3, 1],
              [2, 4, 1]])
y = np.array([[3, 1, 4],
              [3, 5, 1],
              [2, 2, 4],
              [5, 2, 3]])
c=[]
for i in range(len(h)):
    c.append((y[i]-h[i])**2)
print(c)
print([sum(j) for j in [(y[i]-h[i])**2 for i in range(len(h))]])
c = np.array([sum(j) for j in [(y[i]-h[i])**2 for i in range(len(h))]])/len(h)
print(c)

print(np.subtract(y, h))
print(np.square(np.subtract(y, h)))
print(np.square(np.subtract(y, h)).mean())

# Derivative of the cost
# x1, y1, z1, a1, b1, c1 = sp.symbols("x1 y1 y1 a1 b1 c1")
# dc = sp.Derivative((sp.Vector(([x1, y1, z1]) - np.array([a1, b1, c1]))**2)/5, x1)
# print(f"{dc}")
# print(f"{dc.doit()}")