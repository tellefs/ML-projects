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
from data_processing import Data
from print_and_plot import *
from regression_methods import Fitting
from resampling_methods import Resampling
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

activation_function_hidd = "tanh" # "tanh", "relu" or "leaky relu"
# -----------------------------------------------NN---------------------------------------------
 
franke_data.design_matrix(poly_deg)
franke_data.test_train_split(0.2)

n_hidd_layers = 1
nodes_in_hidd_layers = [50]

if(activation_function_hidd=="tanh" or activation_function_hidd=="relu" or activation_function_hidd=="leaky relu"):
	eta = 0.001
	n_epochs = 1000
NN = NeuralNetwork(franke_data.X_train, franke_data.z_train, n_hidden_layers = n_hidd_layers, n_hidden_neurons = nodes_in_hidd_layers, epochs =n_epochs, eta = eta, hidden_act_func=activation_function_hidd)
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
surface_plot(franke_data.x_rescaled, franke_data.y_rescaled, franke_data.z_mesh, franke_data.z_rescaled)


