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

activation_function_hidd = "tanh" # "tanh", "relu" or "leaky reku"
# -----------------------------------------------NN---------------------------------------------
 
FrankeData.DesignMatrix(PolyDeg)
FrankeData.TestTrainSplit(0.2)

NumHiddLayers = 1
NodesInHiddLayers = [50]

if(activation_function_hidd=="tanh" or activation_function_hidd=="relu" or activation_function_hidd=="leaky relu"):
	eta = 0.00001
	n_epochs = 100000
NN = NeuralNetwork(FrankeData.X_train, FrankeData.z_train, n_hidden_layers = NumHiddLayers, n_hidden_neurons = NodesInHiddLayers, epochs =n_epochs, eta = eta, HiddenActFunc=activation_function_hidd)
NN.train()
z_predict = np.ravel(NN.predict(FrankeData.X_test))
z_tilde = np.ravel(NN.predict(FrankeData.X_train))
z_plot = np.ravel(NN.predict(FrankeData.X))

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


""" # This can be used to plot Franke's function
FrankeData.DesignMatrix(PolyDeg)
FrankeData.TestTrainSplit(0.2)

NumHiddLayers = 3
NodesInHiddLayers = [50,50,50]

NN = NeuralNetwork(FrankeData.X_train, FrankeData.z_train, n_hidden_layers = NumHiddLayers, n_hidden_neurons = NodesInHiddLayers, epochs =n_epochs, eta = eta)
NN.train()
z_predict = np.ravel(NN.predict(FrankeData.X_test))
z_tilde = np.ravel(NN.predict(FrankeData.X_train))
z_plot = np.ravel(NN.predict(FrankeData.X))

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

#Plot_Franke_Test_Train(FrankeData.z_test,FrankeData.z_train, FrankeData.X_test, FrankeData.X_train, FrankeData.scaler, FrankeData.x_mesh, FrankeData.y_mesh, FrankeData.z_mesh)
"""

