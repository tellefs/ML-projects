import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from random import random, seed

from src.data_processing import Data
from src.regression_methods import Fitting
from src.statistical_functions import *

np.random.seed(2020)

# Setting up the dataset
bind_eng = Data()
bind_eng.set_binding_energies("mass16.txt")

# Scaling the data
bind_eng.data_scaling()

# Creating the design matrix
poly_deg = 19
bind_eng.design_matrix(poly_deg)

bind_eng.test_train_split(0.2)

# All we can use now is bind_eng.X_test, bind_eng.X_train, bind_eng.z_test, bind_eng.z_train

fit = Fitting(bind_eng)

fit.OLS()

print("Train R2 score:")
print(R2(bind_eng.z_train, fit.z_tilde))

print("Train MSE score:")
print(MSE(bind_eng.z_train, fit.z_tilde))

print("Test R2 score:")
print(R2(bind_eng.z_test, fit.z_predict))

print("Test MSE score:")
print(MSE(bind_eng.z_test, fit.z_predict))



# Rescaling the data back
bind_eng.data_rescaling()
