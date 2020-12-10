import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from random import random, seed

from src.data_processing import Data
from src.regression_methods import Fitting
from src.resampling_methods import Resampling

from src.statistical_functions import *
from src.print_and_plot import *
from src.neuralnetwork import NeuralNetwork
from src.activation_functions import *

from sklearn.neural_network import MLPRegressor #For Regression

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
from tensorflow.keras.models import Model
from tensorflow import keras
from keras.regularizers import l2

""" 

The folowing program performs the linear regression analysis with three options - OLS, Ridge and LASSO
The user might selesct one of the options by varying the regression_method variable taking values "OLS", "Ridge", "LASSO"
and switch one of the resampling options on by choosing between "cv" and "no resampling" for the resampling_method variable.
The output of the program is two grid search files for the test and training MSE and R2.

"""

np.random.seed(2016)

minimization_method = "matrix_inv"
lamb = 0.0

# Setting up the dataset
bind_eng = Data()
bind_eng.set_binding_energies("mass16.txt")

# Scaling the data
bind_eng.data_scaling()

# Setting matrices for the grid search
poly_deg = 3
num_poly = 23
num_lambda = 8
poly_deg_array = np.linspace(1, num_poly, num_poly, dtype = int)
lambda_array = [ 0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]

min_size = 100 	#size of each minibatch
n_epochs = 1000	#number of epochs
eta = 0.0001 	# learning rate
n_hidd_layers = 4

activation_function_hidd = "sigmoid" # tanh, sigmoid

nodes_in_hidd_layers = [50, 50, 70, 60]

eta_array = [0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0.00001, 0.000001]
lambda_array = [ 0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
num_lambda = len(lambda_array)
num_eta = len(eta_array)

min_el, min_eta, min_lamb, imin, jmin = 1000, 0, 0, 0, 0

bind_eng.design_matrix(poly_deg)
bind_eng.test_train_split(0.2)

MSE_test = np.zeros((num_eta, num_lambda))
MSE_train = np.zeros((num_eta, num_lambda))
R2_test = np.zeros((num_eta, num_lambda))
R2_train = np.zeros((num_eta, num_lambda))

# Opening the files for reading and writing
filename_1 = "Files/Grid_search_NN_tanh_R2_tot.txt"
f_1 = open(filename_1, "w")
f_1.write("eta   lambda  R2train  R2test\n")
f_1.close()

filename_2 = "Files/Grid_search_NN_tanh_MSE_tot.txt"
f_2 = open(filename_2, "w")
f_2.write("eta   lambda  MSEtrain  MSEtest\n")
f_2.close()

f_1 = open(filename_1, "a")
f_2 = open(filename_2, "a")

for i in range(num_eta):
	print("Eta: ", eta_array[i])
	for j in range(num_lambda):
		NN = NeuralNetwork(
				bind_eng.X_train, 
				bind_eng.z_train, 
				n_hidden_layers = n_hidd_layers, 
				n_hidden_neurons = nodes_in_hidd_layers, 
				epochs = n_epochs, 
				batch_size = min_size, 
				eta = eta_array[i], 
				lmbd=lambda_array[j], 
				hidden_act_func=activation_function_hidd)
		NN.train()

		z_predict = np.ravel(NN.predict(bind_eng.X_test))
		z_tilde = np.ravel(NN.predict(bind_eng.X_train))
		z_plot = np.ravel(NN.predict(bind_eng.X))

		# Filling up the matrices
		MSE_test[i, j] = MSE(bind_eng.z_test, z_predict)
		MSE_train[i, j] = MSE(bind_eng.z_train, z_tilde)
		R2_test[i, j] = R2(bind_eng.z_test, z_predict)
		R2_train[i, j] = R2(bind_eng.z_train, z_tilde)

		f_1.write("{0} {1} {2} {3}\n".format(eta_array[i], lambda_array[j], R2(bind_eng.z_train, z_tilde), R2(bind_eng.z_test, z_predict)))
		f_2.write("{0} {1} {2} {3}\n".format(eta_array[i], lambda_array[j], MSE(bind_eng.z_train, z_tilde), MSE(bind_eng.z_test, z_predict)))

		# Tracking the optimal parameters
		if(MSE(bind_eng.z_test, z_predict) <= min_el):
			min_el = MSE(bind_eng.z_test, z_predict)
			min_eta = eta_array[i]
			min_lamb = lambda_array[j]
			imin = i
			jmin = j
		print(j)

f_1.close()
f_2.close()

# Printing scores
print("Optimal eta:")	
print(min_eta)
print("Optimal lambda:")	
print(min_lamb)
print("--------------------------------")
print("Scores:")
print("Training MSE:")
print(MSE_train[imin, jmin])
print("Training R2:")
print(R2_test[imin, jmin])
print("Test MSE:")
print(MSE_test[imin, jmin])
print("Test R2:")
print(R2_test[imin, jmin])

# Rescaling the data back
bind_eng.data_rescaling()