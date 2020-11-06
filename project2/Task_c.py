import matplotlib.pyplot as plt
import numpy as np
from random import random, seed

from src.statistical_functions import *
from src.data_processing import Data
from src.print_and_plot import *
from src.regression_methods import Fitting
from src.resampling_methods import Resampling
from src.neuralnetwork import NeuralNetwork
from src.activation_functions import *

''' Task c

	The following file contains the code used to perform all studies
	for the third task of the project. Runs simple analysis for
	different activation functions.

	activation_function_hidd takes values "tanh", "relu" "leaky relu", "sigmoid"
'''

# Setting the Franke's function
poly_deg = 5
N = 30
seed = 2021
alpha = 0.001
lamb = 0.0

min_size = 5 #size of each minibatch
n_epochs = 1000 #number of epochs
eta = 0.001 # learning rate

# Setting Data
franke_data = Data()
franke_data.set_grid_franke_function(N,N,False)
franke_data.set_franke_function()
franke_data.add_noise(alpha, seed)

# Scaling data
franke_data.data_scaling()

activation_function_hidd = "tanh"
# -----------------------------------------------NN---------------------------------------------

franke_data.design_matrix(poly_deg)
franke_data.test_train_split(0.2)

n_hidd_layers = 1
nodes_in_hidd_layers = [50]

if(activation_function_hidd=="tanh"
	or activation_function_hidd=="relu"
	or activation_function_hidd=="leaky relu"):
	eta = 0.001
	n_epochs = 1000

NN = NeuralNetwork(
	franke_data.X_train, franke_data.z_train,
	n_hidden_layers = n_hidd_layers, n_hidden_neurons = nodes_in_hidd_layers,
	epochs =n_epochs, eta = eta,
	hidden_act_func=activation_function_hidd)

NN.train()
z_predict = np.ravel(NN.predict(franke_data.X_test))
z_tilde = np.ravel(NN.predict(franke_data.X_train))
z_plot = np.ravel(NN.predict(franke_data.X))

print("Scores:")
print("Training MSE:")
print(MSE(franke_data.z_train, z_tilde))
print("Training R2:")
print(R2(franke_data.z_train, z_tilde))
print("Test MSE:")
print(MSE(franke_data.z_test, z_predict))
print("Test R2:")
print(R2(franke_data.z_test, z_predict))

franke_data.z_scaled = z_plot # This is for plotting
franke_data.data_rescaling()

# surface plot
surface_plot(franke_data.x_rescaled, franke_data.y_rescaled, franke_data.z_mesh, franke_data.z_rescaled)
