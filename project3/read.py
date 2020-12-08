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

#fit.OLS()
#fit.XGB()
#fit.decision_tree(depth=10,lamb=0.0)


'''print("Train R2 score:")
print(R2(bind_eng.z_train, fit.z_tilde))

print("Train MSE score:")
print(MSE(bind_eng.z_train, fit.z_tilde))

print("Test R2 score:")
print(R2(bind_eng.z_test, fit.z_predict))

print("Test MSE score:")
print(MSE(bind_eng.z_test, fit.z_predict))'''

depth_values = np.linspace(1,10,10)
lambda_values = np.hstack((np.array([0.0]), np.logspace(-6,-1,6)))


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

f_1.close()
f_2.close()
f_3.close()
f_4.close()


# Rescaling the data back
bind_eng.data_rescaling()
