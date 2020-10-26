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

PolyDeg = 5
N = 30
seed = 2021
alpha = 0.001
lamb = 0.001 

Min = 5 #size of each minibatch
n_epochs = 1000 #number of epochs
t0, t1 = 1, 1000

# For SGD_SKL and GD
eta = 0.01
Niterations = 10000

resampling_method = "cv" # "cv", "no resampling "
regression_method ="OLS" # "OLS", "Ridge"
minimization_method = "SGD" # "matrix_inv", "SGD", "GD", "SGD_SKL"

# Setting Data
FrankeData = data()
FrankeData.SetGridFrankeFunction(N,N,False)
FrankeData.SetFrankeFunction()
FrankeData.AddNoise(alpha, seed)

# Scaling data
FrankeData.DataScaling()
"""
model = resampling(FrankeData)

if(resampling_method == "cv"):

	model.Cross_Validation(5, seed, minimization_method, regression_method, PolyDeg, n_epochs=n_epochs, t0=t0, t1=t1, eta=eta, lamb=lamb, Min=Min, Niterations=Niterations)
	#model.Cross_Validation(5, seed, minimization_method, regression_method, PolyDeg, n_epochs=n_epochs, t0=t0, t1=t1, lamb=lamb)
elif(resampling_method == "no resampling"):

	model.NoResampling(seed, minimization_method, regression_method, PolyDeg, n_epochs=n_epochs, t0=t0, t1=t1, eta=eta, lamb=lamb, Min=Min, Niterations=Niterations)

FrankeData.z_scaled = model.z_plot # This is for plotting

# Rescale data
FrankeData.DataRescaling()

print("-------- ",minimization_method, "+", regression_method," -------")
print("Scores:")
print("Training MSE:")
print(model.MSE_train)
print("Training R2:")
print(model.R2_train)
print("Test MSE:")
print(model.MSE_test)
print("Test R2:")
print(model.R2_test)

SurfacePlot(FrankeData.x_rescaled, FrankeData.y_rescaled, FrankeData.z_mesh, FrankeData.z_rescaled)
"""
# -----------------------------------------------NN---------------------------------------------

FrankeData.DesignMatrix(PolyDeg)
FrankeData.TestTrainSplit(0.2)

NumHiddLayers = 1
NodesInHiddLayers = [50]

NN = NeuralNetwork(FrankeData.X_train, FrankeData.z_train, n_hidden_layers = NumHiddLayers, n_hidden_neurons = NodesInHiddLayers)
data_indices = np.arange(NN.n_inputs)
chosen_datapoints = np.random.choice(data_indices, NN.batch_size, replace=False)
NN.X_data = NN.X_data_full[chosen_datapoints]
NN.Y_data = NN.Y_data_full[chosen_datapoints]
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

