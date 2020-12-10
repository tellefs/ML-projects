import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from random import random, seed

from src.data_processing import Data
from src.regression_methods import Fitting
from src.resampling_methods import Resampling

from src.statistical_functions import *
from src.print_and_plot import *


"""

The folowing program performs the linear regression analysis with three options - OLS, Ridge and LASSO
The user might selesct one of the options by varying the regression_method variable taking values "OLS", "Ridge", "LASSO"
and switch one of the resampling options on by choosing between "cv" and "no resampling" for the resampling_method variable.
The output of the program is two grid search files for the test and training MSE and R2.

"""

seed = 2020

minimization_method = "matrix_inv"
regression_method = "Ridge"               # "OLS", "Ridge", "LASSO"
resampling_method = "no resampling" 	  # "cv" or "no resampling"
lamb = 0.001

# Setting up the dataset
bind_eng = Data()
bind_eng.set_binding_energies("mass16.txt")

# Scaling the data
bind_eng.data_scaling()

# Setting matrices and arrays for the grid search
num_poly = 23
poly_deg_array = np.linspace(1, num_poly, num_poly, dtype = int)
lambda_array = [ 0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
num_lambda = len(lambda_array)

MSE_test = np.zeros((num_poly, num_lambda))
MSE_train = np.zeros((num_poly, num_lambda))
R2_test = np.zeros((num_poly, num_lambda))
R2_train = np.zeros((num_poly, num_lambda))

min_el, min_pol, min_lamb, imin = 1000, 0, 0, 0

# Opening the files for reading and writing
filename_1 = "Files/Grid_search_Ridge_R2_tot.txt"
f_1 = open(filename_1, "w")
f_1.write("poly_degree   lambda  R2train  R2test\n")
f_1.close()

filename_2 = "Files/Grid_search_Ridge_MSE_tot.txt"
f_2 = open(filename_2, "w")
f_2.write("poly_degree   lambda  MSEtrain  MSEtest\n")
f_2.close()

f_1 = open(filename_1, "a")
f_2 = open(filename_2, "a")

# Setting up the model
model = Resampling(bind_eng)

# ------------------------------------ Grid search analysis ------------------------------------

for poly_deg in poly_deg_array:
	#print("Polynomial degree: ", poly_deg)
	for i in range(num_lambda):
		# Creating the design matrix
		bind_eng.design_matrix(poly_deg)

		# Choosing between two resampling methods
		if(resampling_method == "cv"):
			model.cross_validation(
				11, seed, minimization_method,
				regression_method, poly_deg, lamb=lambda_array[i])

		elif(resampling_method == "no resampling"):
			model.no_resampling(
				seed, minimization_method,
				regression_method, poly_deg, lamb=lambda_array[i])

		# Filling up the matrices
		MSE_test[poly_deg-1, i] = model.MSE_test
		MSE_train[poly_deg-1, i] = model.MSE_train
		R2_test[poly_deg-1, i] = model.R2_test
		R2_train[poly_deg-1, i] = model.R2_train

		f_1.write("{0} {1} {2} {3}\n".format(poly_deg, lambda_array[i], model.R2_train, model.R2_test))
		f_2.write("{0} {1} {2} {3}\n".format(poly_deg, lambda_array[i], model.MSE_train, model.MSE_test))

		# Tracking the optimal parameters
		if(model.MSE_test <= min_el):
			min_el = model.MSE_test
			min_pol = poly_deg
			min_lamb = lambda_array[i]
			imin = i


f_1.close()
f_2.close()

# Printing the scores
print(r"Optimal polynomial degree:")
print(min_pol)
print(r"Optimal lambda:")
print(min_lamb)
print("--------------------------------")
print("Scores:")
print("Training MSE: ", MSE_train[min_pol-1, imin])
print("Training R2: ", R2_train[min_pol-1, imin])
print("Test MSE: ", MSE_test[min_pol-1, imin])
print("Test R2: ", R2_test[min_pol-1, imin])

# Rescaling the data back
bind_eng.data_rescaling()
