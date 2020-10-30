# The main program is used to collecting results for the Franke's function

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from imageio import imread

from sklearn.model_selection import train_test_split
import scipy.linalg as scl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
import sklearn.linear_model as skl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.linear_model import SGDRegressor

from statistical_functions import *
from data_processing import data
from print_and_plot import *
from regression_methods import fitting
from resampling_methods import resampling
from neuralnetwork import NeuralNetwork
from activation_functions import *

from sklearn.neural_network import MLPClassifier #For Classification
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

PolyDeg = 5
N = 30
seed = 2021
alpha = 0.001
lamb = 0.001 

Min = 5 #size of each minibatch
n_epochs = 1000 #number of epochs
eta = 0.001 # learning rate

regression_method ="OLS" # "OLS", "Ridge"

# Setting Data
FrankeData = data()
FrankeData.SetGridFrankeFunction(N,N,False)
FrankeData.SetFrankeFunction()
FrankeData.AddNoise(alpha, seed)

# Scaling data
FrankeData.DataScaling()

study = "simple analysis" # "epochs", "hidden layers", "Ridge grid search"
NN = "keras" # "self made", "skl", "keras"

FrankeData.DesignMatrix(PolyDeg)
FrankeData.TestTrainSplit(0.2)

# -----------------------------------------------Simple study---------------------------------------------
# Simple study of R2 and MSE for different NN
if(study == "simple analysis"):

	if(regression_method == "OLS"):
		lamb = 0.0

	if(NN == "self-made"):

		NumHiddLayers = 3
		NodesInHiddLayers = [50,50,50]

		NN = NeuralNetwork(FrankeData.X_train, FrankeData.z_train, n_hidden_layers = NumHiddLayers, n_hidden_neurons = NodesInHiddLayers, epochs = n_epochs, batch_size = Min, eta = eta, lmbd=lamb)
		NN.train()
		z_predict = np.ravel(NN.predict(FrankeData.X_test))
		z_tilde = np.ravel(NN.predict(FrankeData.X_train))
		z_plot = np.ravel(NN.predict(FrankeData.X))

	elif(NN == "skl"):

		regr = MLPRegressor(hidden_layer_sizes=(50,50,50),activation='logistic',solver='sgd', alpha=lamb, batch_size=Min, learning_rate_init=eta, max_iter=n_epochs).fit(FrankeData.X_train, FrankeData.z_train)
		z_predict = np.ravel(regr.predict(FrankeData.X_test))
		z_tilde = np.ravel(regr.predict(FrankeData.X_train))
		z_plot = np.ravel(regr.predict(FrankeData.X))

	elif(NN == "keras"):

		model = Sequential()
		model.add(Dense(50, activation='sigmoid', kernel_regularizer=l2(lamb),kernel_initializer='normal'))
		model.add(Dense(50, activation='sigmoid', kernel_regularizer=l2(lamb), kernel_initializer='normal'))
		model.add(Dense(50,activation='sigmoid', kernel_regularizer=l2(lamb), kernel_initializer='normal'))
		model.add(Dense(1, activation='linear', kernel_regularizer=l2(lamb), kernel_initializer='normal'))

		#opt = keras.optimizers.Adam(learning_rate=0.001)
		opt = keras.optimizers.SGD(learning_rate=0.01) # use in case user wants to change default eta = 0.01 for sgd

		model.compile(loss='mse', optimizer=opt, metrics=['mse','mae'])
		model.fit(FrankeData.X_train, FrankeData.z_train, epochs=n_epochs, batch_size=Min)	

		z_predict = np.ravel(model.predict(FrankeData.X_test))
		z_tilde = np.ravel(model.predict(FrankeData.X_train))
		z_plot = np.ravel(model.predict(FrankeData.X))

	# Printing scores
	print("Scores:")
	print("Training MSE:")
	print(MSE(FrankeData.z_train, z_tilde))
	print("Training R2:")
	print(R2(FrankeData.z_train, z_tilde))
	print("Test MSE:")
	print(MSE(FrankeData.z_test, z_predict))
	print("Test R2:")
	print(R2(FrankeData.z_test, z_predict))
	
	FrankeData.z_scaled = z_plot # This is for plotting
	FrankeData.DataRescaling()
	SurfacePlot(FrankeData.x_rescaled, FrankeData.y_rescaled, FrankeData.z_mesh, FrankeData.z_rescaled)

# -----------------------------------------------epochs---------------------------------------------
# Study of R2 and MSE dependence on the number of epochs
if(study == "epochs"):

	if(regression_method == "OLS"):
		lamb = 0.0

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

	NumHiddLayers = 3
	NodesInHiddLayers = [50,50,50]

	for n_epochs in epochs:

		NN = NeuralNetwork(FrankeData.X_train, FrankeData.z_train, n_hidden_layers = NumHiddLayers, n_hidden_neurons = NodesInHiddLayers, epochs = n_epochs, batch_size = Min, eta = eta, lmbd=lamb)
		NN.train()
		z_predict = np.ravel(NN.predict(FrankeData.X_test))
		z_tilde = np.ravel(NN.predict(FrankeData.X_train))
		z_plot = np.ravel(NN.predict(FrankeData.X))
		f_1.write("{0} {1} {2}\n".format(n_epochs, R2(FrankeData.z_train, z_tilde), R2(FrankeData.z_test, z_predict)))
		f_2.write("{0} {1} {2}\n".format(n_epochs, MSE(FrankeData.z_train, z_tilde), MSE(FrankeData.z_test, z_predict)))
	f_1.close()
	f_2.close()


# -----------------------------------------------hidden layers---------------------------------------------
# Study of R2 and MSE dependence on the number of hidden layers
if(study == "hidden layers"):

	if(regression_method == "OLS"):
		lamb = 0.0

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

	NumHiddLayers = 1
	NodesInHiddLayers = [50]

	for i in range(10):
		NN = NeuralNetwork(FrankeData.X_train, FrankeData.z_train, n_hidden_layers = NumHiddLayers, n_hidden_neurons = NodesInHiddLayers, epochs = n_epochs, batch_size = Min, eta = eta, lmbd=lamb)
		NN.train()
		z_predict = np.ravel(NN.predict(FrankeData.X_test))
		z_tilde = np.ravel(NN.predict(FrankeData.X_train))
		z_plot = np.ravel(NN.predict(FrankeData.X))

		f_1.write("{0} {1} {2}\n".format(NumHiddLayers, R2(FrankeData.z_train, z_tilde), R2(FrankeData.z_test, z_predict)))
		f_2.write("{0} {1} {2}\n".format(NumHiddLayers, MSE(FrankeData.z_train, z_tilde), MSE(FrankeData.z_test, z_predict)))
		NumHiddLayers = NumHiddLayers+1
		NodesInHiddLayers.append(50)	
	f_1.close()
	f_2.close()

# -----------------------------------------------Ridge grid search---------------------------------------------
# Study of R2 and MSE dependence on eta and lambda for ridge

if(study == "Ridge grid search"):

	lambdas = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
	etas = [0.1, 0.01, 0.001,0.0001,0.00001,0.000001]

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

	NumHiddLayers = 3
	NodesInHiddLayers = [50,50,50]

	for lamb in lambdas:
		for eta in etas:
			NN = NeuralNetwork(FrankeData.X_train, FrankeData.z_train, n_hidden_layers = NumHiddLayers, n_hidden_neurons = NodesInHiddLayers, epochs = n_epochs, batch_size = Min, eta = eta, lmbd=lamb)
			NN.train()			
			z_predict = np.ravel(NN.predict(FrankeData.X_test))
			z_tilde = np.ravel(NN.predict(FrankeData.X_train))
			z_plot = np.ravel(NN.predict(FrankeData.X))
			f_1.write("{0} {1} {2} {3}\n".format(lamb, eta, R2(FrankeData.z_train, z_tilde), R2(FrankeData.z_test, z_predict)))
			f_2.write("{0} {1} {2}\n".format(lamb, eta, MSE(FrankeData.z_train, z_tilde), MSE(FrankeData.z_test, z_predict)))

	f_1.close()
	f_2.close()
	
