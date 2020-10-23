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

PolyDeg = 5
N = 30
seed = 2021
alpha = 0.001
lamb = 0.01

regression_method ="OLS" # "OLS", "Ridge"
minimization_method = "matrix_inv" # "SGD", "GD", "SGD_SKL"

# Setting Data
FrankeData = data()
FrankeData.SetGridFrankeFunction(N,N,False)
FrankeData.SetFrankeFunction()
FrankeData.AddNoise(alpha, seed)


# Scaling data
FrankeData.DataScaling()

# Set up design matrix
FrankeData.DesignMatrix(PolyDeg)
FrankeData.TestTrainSplit(0.2)

fit = fitting(FrankeData)


if (regression_method == "OLS"):
	fit.OLS()
elif(regression_method == "Ridge"):
	fit.Ridge(lamb)

FrankeData.z_scaled = fit.z_plot

# Rescaling data
FrankeData.DataRescaling()

# Print error estimations and plot the surfaces
print("--------Own inversion-------")
print("Beta:")
print(fit.beta)
print("Scores:")
PrintErrors(FrankeData.z_train,fit.z_tilde,FrankeData.z_test,fit.z_predict)
SurfacePlot(FrankeData.x_rescaled, FrankeData.y_rescaled, FrankeData.z_mesh, FrankeData.z_rescaled)

#-------------------------------------- Stochastic gradient descent ----------------------------------

Min = 5 #size of each minibatch
n_epochs = 1000 #number of epochs
t0, t1 = 1, 1000

# Scaling data
FrankeData.DataScaling()

# Set up design matrix
FrankeData.DesignMatrix(PolyDeg)
FrankeData.TestTrainSplit(0.2)

# Number of minibatches
NumMin = int(len(FrankeData.z_train)/Min)

# Splitting into minibatches
FrankeData.SplitMinibatch(True, NumMin)

fit.SGD(n_epochs, t0, t1, seed, regression_method, lamb)

FrankeData.z_scaled = fit.z_plot

# Rescaling data
FrankeData.DataRescaling()

print("--------Stochastic GD-------")
print("Beta:")
print(fit.beta)
print("Scores:")
PrintErrors(FrankeData.z_train,fit.z_tilde,FrankeData.z_test,fit.z_predict)
SurfacePlot(FrankeData.x_rescaled, FrankeData.y_rescaled, FrankeData.z_mesh, FrankeData.z_rescaled)

#-------------------------------------- Stochastic gradient descent skl ----------------------------------

# Scaling data
FrankeData.DataScaling()

# Set up design matrix
FrankeData.DesignMatrix(PolyDeg)
FrankeData.TestTrainSplit(0.2)
eta = 0.001

fit.SGD_SKL(eta, seed, regression_method, lamb)
FrankeData.z_scaled = fit.z_plot

# Rescaling data
FrankeData.DataRescaling()

print("--------Stochastic GD skl-------")
print("Beta:")
print(fit.beta)
print("Scores:")
PrintErrors(FrankeData.z_train,fit.z_tilde,FrankeData.z_test,fit.z_predict)
SurfacePlot(FrankeData.x_rescaled, FrankeData.y_rescaled, FrankeData.z_mesh, FrankeData.z_rescaled)

#-------------------------------------- Gradient descent ----------------------------------

# Scaling data
FrankeData.DataScaling()

# Set up design matrix
FrankeData.DesignMatrix(PolyDeg)
FrankeData.TestTrainSplit(0.2)

eta = 0.01
Niterations = 100000
fit.GD(Niterations, eta, seed, regression_method, lamb)
FrankeData.z_scaled = fit.z_plot

# Rescaling data
FrankeData.DataRescaling()

print("-------- GD -------")
print("Beta:")
print(fit.beta)
print("Scores:")
PrintErrors(FrankeData.z_train,fit.z_tilde,FrankeData.z_test,fit.z_predict)
SurfacePlot(FrankeData.x_rescaled, FrankeData.y_rescaled, FrankeData.z_mesh, FrankeData.z_rescaled)

#-------------------------------------- CV ----------------------------------


FrankeData.DataScaling()

model = resampling(FrankeData)
model.Cross_Validation(5, seed, minimization_method, regression_method, PolyDeg, n_epochs=n_epochs, t0=t0, t1=t1, eta=eta, lamb=lamb, Min=Min, Niterations=Niterations)

FrankeData.z_scaled = model.z_plot
FrankeData.DataRescaling()

print("-------- CV+OLS -------")
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

#-------------------------------------- No Resampling ----------------------------------

FrankeData.DataScaling()

model = resampling(FrankeData)
model.NoResampling(seed, minimization_method, regression_method, PolyDeg, n_epochs=n_epochs, t0=t0, t1=t1, eta=eta, lamb=lamb, Min=Min, Niterations=Niterations)

FrankeData.z_scaled = model.z_plot
FrankeData.DataRescaling()

print("-------- no resampling+OLS -------")
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
