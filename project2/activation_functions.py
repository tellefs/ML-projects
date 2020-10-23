# All activation functions and their derivatives

import numpy as np

def Sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def ReLU(x):
	return numpy.max(0,x)

def Softmax(x):
	exp_term = np.exp(x)
	return exp_term / np.sum(exp_term, axis=1, keepdims=True)

def ActivationFunction(x, activation_function):

	if(activation_function =="softmax"):
		return Softmax(x)
	elif(activation_function =="sigmoid"):
		return Sigmoid(x)
	elif(activation_function =="relu"):
		return ReLU(x)
	elif(activation_function =="tanh"):
		return np.tanh(x)
	elif(activation_function =="linear"):
		return x

def ActivationFunctionDeriv(x, activation_function):

	if(activation_function =="softmax"):
		return Softmax(x)*(1-Softmax(x))
	elif(activation_function =="sigmoid"):
		return Sigmoid(x)*(1-Sigmoid(x))
	elif(activation_function =="relu"):
		return np.heaviside(x, 0)
	elif(activation_function =="tanh"):
		return 1-np.tanh(x)**2
	elif(activation_function =="linear"):
		return 1