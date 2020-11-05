import numpy as np
from random import random, seed

from statistical_functions import *
from data_processing import Data
from print_and_plot import *
from regression_methods import Fitting
from resampling_methods import Resampling
from neuralnetwork import NeuralNetwork
from activation_functions import *

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

''' Task b

	The following file contains the code used to perform all studies 
	for the second task of the project. User defines which type of NN 
	will be used (self made NN, SKL and Keras NN).

	Depending on choice of study, either study of R2 and MSE is run for 
	different number of hidden layers ("hidden layers"), number of epochs ("epochs"),
	number of neurons ("neurons"), or ridge gridd search ("Ridge grid search")

	study: "simple analysis", "epochs", "hidden layers", "Ridge grid search", "neurons"
	NN_type = "self made", "skl", "keras"
''' 

# Setting the Franke's function
poly_deg = 5
N = 30
seed = 2021
alpha = 0.001
lamb = 0.001 

min_size = 5 	#size of each minibatch
n_epochs = 1000 	#number of epochs
eta = 0.001 	# learning rate

# user-defined settings
study = "simple analysis" 
NN_type = "self made" 

# Setting Data
franke_data = Data()
franke_data.set_grid_franke_function(N,N,False)
franke_data.set_franke_function()
franke_data.add_noise(alpha, seed)

# Scaling data
franke_data.data_scaling()

franke_data.design_matrix(poly_deg)
franke_data.test_train_split(0.2)

# -----------------------------------------------Simple study---------------------------------------------
# Simple study of R2 and MSE for different NN
if(study == "simple analysis"):

	if(NN_type == "self made"):

		# Setting number of hidden layers and neurons in each layer
		n_hidd_layers = 3
		nodes_in_hidd_layers = [50,50,50]

		NN = NeuralNetwork(
			franke_data.X_train, franke_data.z_train, 
			n_hidden_layers = n_hidd_layers, n_hidden_neurons = nodes_in_hidd_layers, 
			epochs = n_epochs, batch_size = min_size, 
			eta = eta, lmbd=lamb)

		NN.train()
		z_predict = np.ravel(NN.predict(franke_data.X_test))
		z_tilde = np.ravel(NN.predict(franke_data.X_train))
		z_plot = np.ravel(NN.predict(franke_data.X))

		plot_franke_test_train(
			z_predict,z_tilde, franke_data.X_test, 
			franke_data.X_train, franke_data.scaler, 
			franke_data.x_mesh, franke_data.y_mesh, 
			franke_data.z_mesh)

	elif(NN_type == "skl"):

		regr = MLPRegressor(
			hidden_layer_sizes=(50,50,50),activation='logistic',
			solver='adam', alpha=lamb, 
			batch_size=min_size, learning_rate_init=eta, 
			max_iter=n_epochs)

		regr.fit(franke_data.X_train, franke_data.z_train)
		z_predict = np.ravel(regr.predict(franke_data.X_test))
		z_tilde = np.ravel(regr.predict(franke_data.X_train))
		z_plot = np.ravel(regr.predict(franke_data.X))

	elif(NN_type == "keras"):

		model = Sequential()
		model.add(Dense(50, activation='sigmoid',kernel_regularizer=l2(lamb),kernel_initializer='normal'))
		model.add(Dense(50, activation='sigmoid', kernel_regularizer=l2(lamb), kernel_initializer='normal'))
		model.add(Dense(50,activation='sigmoid', kernel_regularizer=l2(lamb), kernel_initializer='normal'))
		model.add(Dense(1, activation='linear', kernel_regularizer=l2(lamb), kernel_initializer='normal'))

		#opt = keras.optimizers.Adam(learning_rate=0.001)
		# use in case user wants to change default eta = 0.01 for sgd
		opt = keras.optimizers.SGD(learning_rate=0.01) 

		model.compile(loss='mse', optimizer=opt, metrics=['mse','mae'])
		model.fit(franke_data.X_train, franke_data.z_train, epochs=n_epochs, batch_size=min_size)	

		z_predict = np.ravel(model.predict(franke_data.X_test))
		z_tilde = np.ravel(model.predict(franke_data.X_train))
		z_plot = np.ravel(model.predict(franke_data.X))

	# Printing scores
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
	surface_plot(franke_data.x_rescaled, franke_data.y_rescaled, franke_data.z_mesh, franke_data.z_rescaled)

# -----------------------------------------------epochs---------------------------------------------
# Study of R2 and MSE dependence on the number of epochs
if(study == "epochs"):

	filename_1 = "Files/NN_R2_epochs.txt"
	f_1 = open(filename_1, "w")
	f_1.write("eta   R2train  R2test\n")
	f_1.close()
	
	filename_2 = "Files/NN_MSE_epochs.txt"
	f_2 = open(filename_2, "w")
	f_2.write("eta   MSEtrain  MSEtest\n")
	f_2.close()

	f_1 = open(filename_1, "a")
	f_2 = open(filename_2, "a")

	epochs = np.arange(20,1020,20)

	# Setting number of hidden layers and neurons in each layer
	n_hidd_layers = 3
	nodes_in_hidd_layers = [50,50,50]

	for n_epochs in epochs:

		NN = NeuralNetwork(franke_data.X_train, franke_data.z_train, n_hidden_layers = n_hidd_layers, n_hidden_neurons = nodes_in_hidd_layers, epochs = n_epochs, batch_size = min_size, eta = eta, lmbd=lamb)
		NN.train()
		z_predict = np.ravel(NN.predict(franke_data.X_test))
		z_tilde = np.ravel(NN.predict(franke_data.X_train))
		z_plot = np.ravel(NN.predict(franke_data.X))
		f_1.write("{0} {1} {2}\n".format(n_epochs, R2(franke_data.z_train, z_tilde), R2(franke_data.z_test, z_predict)))
		f_2.write("{0} {1} {2}\n".format(n_epochs, MSE(franke_data.z_train, z_tilde), MSE(franke_data.z_test, z_predict)))
	f_1.close()
	f_2.close()


# -----------------------------------------------hidden layers---------------------------------------------
# Study of R2 and MSE dependence on the number of hidden layers
if(study == "hidden layers"):

	# Setting files
	filename_1 = "Files/NN_R2_layers.txt"
	f_1 = open(filename_1, "w")
	f_1.write("eta   R2train  R2test\n")
	f_1.close()
	
	filename_2 = "Files/NN_MSE_layers.txt"
	f_2 = open(filename_2, "w")
	f_2.write("eta   MSEtrain  MSEtest\n")
	f_2.close()

	f_1 = open(filename_1, "a")
	f_2 = open(filename_2, "a")

	# Setting number of hidden layers and neurons in each layer
	n_hidd_layers = 1
	nodes_in_hidd_layers = [50]

	for i in range(10):
		NN = NeuralNetwork(franke_data.X_train, franke_data.z_train, n_hidden_layers = n_hidd_layers, n_hidden_neurons = nodes_in_hidd_layers, epochs = n_epochs, batch_size = min_size, eta = eta, lmbd=lamb)
		NN.train()
		z_predict = np.ravel(NN.predict(franke_data.X_test))
		z_tilde = np.ravel(NN.predict(franke_data.X_train))
		z_plot = np.ravel(NN.predict(franke_data.X))

		f_1.write("{0} {1} {2}\n".format(n_hidd_layers, R2(franke_data.z_train, z_tilde), R2(franke_data.z_test, z_predict)))
		f_2.write("{0} {1} {2}\n".format(n_hidd_layers, MSE(franke_data.z_train, z_tilde), MSE(franke_data.z_test, z_predict)))
		n_hidd_layers = n_hidd_layers+1
		nodes_in_hidd_layers.append(50)	
	f_1.close()
	f_2.close()

# -----------------------------------------------neurons---------------------------------------------
# Study of R2 and MSE dependence on the number of neurons
if(study == "neurons"):

	# Setting files
	filename_1 = "Files/NN_R2_neurons.txt"
	f_1 = open(filename_1, "w")
	f_1.write("eta   R2train  R2test\n")
	f_1.close()
	
	filename_2 = "Files/NN_MSE_neurons.txt"
	f_2 = open(filename_2, "w")
	f_2.write("eta   MSEtrain  MSEtest\n")
	f_2.close()

	f_1 = open(filename_1, "a")
	f_2 = open(filename_2, "a")

	# Setting number of hidden layers and neurons in each layer
	n_hidd_layers = 1
	nodes_in_hidd_layers = [1]
	number_of_neurons = 1

	for i in range(50):
		NN = NeuralNetwork(
			franke_data.X_train, franke_data.z_train, 
			n_hidden_layers = n_hidd_layers, n_hidden_neurons = nodes_in_hidd_layers, 
			epochs = n_epochs, batch_size = min_size, 
			eta = eta, lmbd=lamb)

		NN.train()
		z_predict = np.ravel(NN.predict(franke_data.X_test))
		z_tilde = np.ravel(NN.predict(franke_data.X_train))
		z_plot = np.ravel(NN.predict(franke_data.X))

		f_1.write("{0} {1} {2}\n".format(number_of_neurons, R2(franke_data.z_train, z_tilde), R2(franke_data.z_test, z_predict)))
		f_2.write("{0} {1} {2}\n".format(number_of_neurons, MSE(franke_data.z_train, z_tilde), MSE(franke_data.z_test, z_predict)))
		number_of_neurons=number_of_neurons+1
		nodes_in_hidd_layers[0]=number_of_neurons

	f_1.close()
	f_2.close()

# -----------------------------------------------Ridge grid search---------------------------------------------
# Study of R2 and MSE dependence on eta and lambda for ridge

if(study == "Ridge grid search"):

	lambdas = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
	etas = [0.1, 0.01, 0.001,0.0001,0.00001,0.000001]

	# Setting files
	filename_1 = "Files/NN_Ridge_R2.txt"
	f_1 = open(filename_1, "w")
	f_1.write("lamb eta   R2train  R2test\n")
	f_1.close()
	
	filename_2 = "Files/NN_Ridge_MSE.txt"
	f_2 = open(filename_2, "w")
	f_2.write("lamb eta   MSEtrain  MSEtest\n")
	f_2.close()

	f_1 = open(filename_1, "a")
	f_2 = open(filename_2, "a")

	n_hidd_layers = 3
	nodes_in_hidd_layers = [50,50,50]

	for lamb in lambdas:
		for eta in etas:
			NN = NeuralNetwork(
				franke_data.X_train, franke_data.z_train, 
				n_hidden_layers = n_hidd_layers, n_hidden_neurons = nodes_in_hidd_layers, 
				epochs = n_epochs, batch_size = min_size, 
				eta = eta, lmbd=lamb)

			NN.train()			
			z_predict = np.ravel(NN.predict(franke_data.X_test))
			z_tilde = np.ravel(NN.predict(franke_data.X_train))
			z_plot = np.ravel(NN.predict(franke_data.X))

			f_1.write("{0} {1} {2} {3}\n".format(lamb, eta, R2(franke_data.z_train, z_tilde), R2(franke_data.z_test, z_predict)))
			f_2.write("{0} {1} {2}\n".format(lamb, eta, MSE(franke_data.z_train, z_tilde), MSE(franke_data.z_test, z_predict)))

	f_1.close()
	f_2.close()
	
