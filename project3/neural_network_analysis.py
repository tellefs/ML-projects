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

"""

The folowing program performs the FFNN analysis by studying how the scores behave with the increasing number of
layers for a given number of neurons per layer (study_option == "number of layers"), number of neurons
study_option == "number of neurons" and performs a grid search for the selected values of lambda and learning rate eta
study_option == "grid search" or simple analysis for the chosen parameters for study_option == "simple analysis".
Two options for the activation function are available (tanh and sigmoid).

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

study_option = "grid search" # "number of layers", "number of neurons", "grid search", "simple analysis"
activation_function_hidd = "sigmoid" # "tanh", "sigmoid"

n_hidd_layers = 4
nodes_in_hidd_layers = [50, 50, 70, 60]

# Setting arrays for the grid search
eta_array = [0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0.00001, 0.000001]
lambda_array = [ 0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
num_lambda = len(lambda_array)
num_eta = len(eta_array)

min_el, min_eta, min_lamb, imin, jmin = 1000, 0, 0, 0, 0

bind_eng.design_matrix(poly_deg)
bind_eng.test_train_split(0.2)


# -----------------------------------------------hidden layers---------------------------------------------
# Study of R2 and MSE dependence on the number of hidden layers
if(study_option == "number of layers"):
	# Setting files
	filename_1 = "Files/NN_R2_layers.txt"
	f_1 = open(filename_1, "w")
	f_1.write("layers   R2train  R2test\n")
	f_1.close()

	filename_2 = "Files/NN_MSE_layers.txt"
	f_2 = open(filename_2, "w")
	f_2.write("layers   MSEtrain  MSEtest\n")
	f_2.close()

	f_1 = open(filename_1, "a")
	f_2 = open(filename_2, "a")

	# Setting number of hidden layers and neurons in each layer
	n_hidd_layers = 1
	nodes_in_hidd_layers = [50]

	for i in range(20):
		print(n_hidd_layers)
		NN = NeuralNetwork(
			bind_eng.X_train,
			bind_eng.z_train,
			n_hidden_layers = n_hidd_layers,
			n_hidden_neurons = nodes_in_hidd_layers,
			epochs = n_epochs,
			batch_size = min_size,
			eta = eta,
			lmbd=lamb)
		NN.train()
		z_predict = np.ravel(NN.predict(bind_eng.X_test))
		z_tilde = np.ravel(NN.predict(bind_eng.X_train))
		z_plot = np.ravel(NN.predict(bind_eng.X))

		f_1.write("{0} {1} {2}\n".format(n_hidd_layers, R2(bind_eng.z_train, z_tilde), R2(bind_eng.z_test, z_predict)))
		f_2.write("{0} {1} {2}\n".format(n_hidd_layers, MSE(bind_eng.z_train, z_tilde), MSE(bind_eng.z_test, z_predict)))
		n_hidd_layers = n_hidd_layers+1
		nodes_in_hidd_layers.append(50)
	f_1.close()
	f_2.close()
# ----------------------------------------------- number of neurons ---------------------------------------------
elif(study_option == "number of neurons"):
	# Setting files
	filename_1 = "Files/NN_R2_neurons.txt"
	f_1 = open(filename_1, "w")
	f_1.write("neurons   R2train  R2test\n")
	f_1.close()

	filename_2 = "Files/NN_MSE_neurons.txt"
	f_2 = open(filename_2, "w")
	f_2.write("neurons   MSEtrain  MSEtest\n")
	f_2.close()

	f_1 = open(filename_1, "a")
	f_2 = open(filename_2, "a")

	# Setting number of hidden layers and neurons in each layer
	n_hidd_layers = 5
	nodes_in_hidd_layers = [10, 10, 10, 10, 10]

	for i in range(9):
		NN = NeuralNetwork(
			bind_eng.X_train,
			bind_eng.z_train,
			n_hidden_layers = n_hidd_layers,
			n_hidden_neurons = nodes_in_hidd_layers,
			epochs = n_epochs,
			batch_size = min_size,
			eta = eta,
			lmbd=lamb)
		NN.train()
		z_predict = np.ravel(NN.predict(bind_eng.X_test))
		z_tilde = np.ravel(NN.predict(bind_eng.X_train))
		z_plot = np.ravel(NN.predict(bind_eng.X))

		f_1.write("{0} {1} {2}\n".format(n_hidd_layers, R2(bind_eng.z_train, z_tilde), R2(bind_eng.z_test, z_predict)))
		f_2.write("{0} {1} {2}\n".format(n_hidd_layers, MSE(bind_eng.z_train, z_tilde), MSE(bind_eng.z_test, z_predict)))
		for j in range(n_hidd_layers):
			nodes_in_hidd_layers[j]=nodes_in_hidd_layers[j]+10
	f_1.close()
	f_2.close()
# ----------------------------------------------- grid search ---------------------------------------------
# Grid search analysis
elif(study_option == "grid search"):

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
	print(R2_train[imin, jmin])
	print("Test MSE:")
	print(MSE_test[imin, jmin])
	print("Test R2:")
	print(R2_test[imin, jmin])
# ----------------------------------------------- Simple analysis ---------------------------------------------
# Performs a simple analysis of the scores with the parameters chosen in the beginning of the program
elif(study_option == "simple analysis"):
	NN = NeuralNetwork(
			bind_eng.X_train,
			bind_eng.z_train,
			n_hidden_layers = n_hidd_layers,
			n_hidden_neurons = nodes_in_hidd_layers,
			epochs = n_epochs,
			batch_size = min_size,
			eta = eta,
			lmbd=lamb,
			hidden_act_func=activation_function_hidd)
	NN.train()

	z_predict = np.ravel(NN.predict(bind_eng.X_test))
	z_tilde = np.ravel(NN.predict(bind_eng.X_train))
	z_plot = np.ravel(NN.predict(bind_eng.X))

	# Printing scores
	print("--------------------------------")
	print("Scores:")
	print("Training MSE:")
	print(MSE(bind_eng.z_train, z_tilde))
	print("Training R2:")
	print(R2(bind_eng.z_train, z_tilde))
	print("Test MSE:")
	print(MSE(bind_eng.z_test, z_predict))
	print("Test R2:")
	print(R2(bind_eng.z_test, z_predict))

# Rescaling the data back
bind_eng.data_rescaling()
