import numpy as np
import matplotlib.pyplot as plt
import franke


x = np.random.rand(100)
y = np.random.rand(100)

f = franke.FrankeFunction(x, y)


X = np.zeros((len(x), 10))

X[:,0] = 1.0
X[:,1] = x
X[:,2] = x**2
X[:,3] = x**3
X[:,4] = x**4
X[:,5] = 1.0
X[:,6] = y
X[:,7] = y**2
X[:,8] = y**3
X[:,9] = y**4

print(X)














