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

PolyDeg = 7
N = 30
seed = 2021
alpha = 0.001
lamb = 0.01 

Min = 5 #size of each minibatch
n_epochs = 1000 #number of epochs
t0, t1 = 1, 1000

# For SGD_SKL and GD
eta = 0.001
Niterations = 10000

resampling_method = "cv" # "cv", "no resampling "
regression_method ="Ridge" # "OLS", "Ridge"
minimization_method = "matrix_inv" # "matrix_inv", "SGD", "GD", "SGD_SKL"

# Setting Data
FrankeData = data()
FrankeData.SetGridFrankeFunction(N,N,False)
FrankeData.SetFrankeFunction()
FrankeData.AddNoise(alpha, seed)

# Scaling data
FrankeData.DataScaling()

model = resampling(FrankeData)

if(resampling_method == "cv"):

	model.Cross_Validation(5, n_epochs, t0, t1, seed, eta, minimization_method, regression_method, PolyDeg, lamb, Min, Niterations)
elif(resampling_method == "no resampling"):

	model.NoResampling( n_epochs, t0, t1, seed, eta, minimization_method, regression_method, PolyDeg, lamb, Min, Niterations)

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

