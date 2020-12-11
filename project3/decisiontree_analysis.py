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
poly_deg = 1
bind_eng.design_matrix(poly_deg)
deleted_matrix = np.delete(bind_eng.X,0,1)
bind_eng.X=deleted_matrix

bind_eng.test_train_split(0.2)

# All we can use now is bind_eng.X_test, bind_eng.X_train, bind_eng.z_test, bind_eng.z_train

fit = Fitting(bind_eng)

depth_values = np.linspace(1,14,14)
lambda_values = np.hstack((np.array([0.0]), np.logspace(-7,-1,7)))

min_mse_test, min_r2_test, min_mse_train, min_r2_train = 1000, 0, 0, 0
min_depth, min_lamb = 0, 0


filename_1 = 'Files/DecisionTree_test_MSE.txt'
filename_2 = 'Files/DecisionTree_test_R2.txt'
filename_3 = 'Files/DecisionTree_train_MSE.txt'
filename_4 = 'Files/DecisionTree_train_R2.txt'

f_1 = open(filename_1, "w")
f_1.write("lambda   depth  MSEtest\n")
f_1.close()
f_2 = open(filename_2, "w")
f_2.write("lambda   depth  R2test\n")
f_2.close()
f_3 = open(filename_3, "w")
f_3.write("lambda   depth  MSEtrain\n")
f_3.close()
f_4 = open(filename_4, "w")
f_4.write("lambda   depth  R2train\n")
f_4.close()

f_1 = open(filename_1, "a")
f_2 = open(filename_2, "a")
f_3 = open(filename_3, "a")
f_4 = open(filename_4, "a")

for i, depth in enumerate(depth_values):
    for j, lamb in enumerate(lambda_values):
        fit.decision_tree(depth=depth,lamb=lamb)

        f_1.write('{0} {1} {2}\n'.format(lamb, depth, MSE(bind_eng.z_test, fit.z_predict)))
        f_2.write('{0} {1} {2}\n'.format(lamb, depth, R2(bind_eng.z_test, fit.z_predict)))
        f_3.write('{0} {1} {2}\n'.format(lamb, depth, MSE(bind_eng.z_train, fit.z_tilde)))
        f_4.write('{0} {1} {2}\n'.format(lamb, depth, R2(bind_eng.z_train, fit.z_tilde)))

        # Tracking the optimal parameters
        if(MSE(bind_eng.z_test, fit.z_predict) <= min_mse_test):
            min_mse_test = MSE(bind_eng.z_test, fit.z_predict)
            min_r2_test = R2(bind_eng.z_test, fit.z_predict)
            min_mse_train = MSE(bind_eng.z_train, fit.z_tilde)
            min_r2_train = R2(bind_eng.z_train, fit.z_tilde)
            min_depth = depth
            min_lamb = lamb

f_1.close()
f_2.close()
f_3.close()
f_4.close()

# Printing scores
print("Optimal depth:")
print(min_depth)
print("Optimal lambda:")
print(min_lamb)
print("--------------------------------")
print("Scores:")
print("Training MSE:")
print(min_mse_train)
print("Training R2:")
print(min_r2_train)
print("Test MSE:")
print(min_mse_test)
print("Test R2:")
print(min_r2_test)

# Rescaling the data back
bind_eng.data_rescaling()
