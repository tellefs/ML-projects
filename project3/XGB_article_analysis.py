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
from src.neuralnetwork import NeuralNetwork
from src.activation_functions import *

"""

The folowing program performs the grid search linear regression analysis with XGBoost,
using the total binding energy data, to compare with the results from the papers
discussed in the report.

"""

np.random.seed(2020)

#values to perform gridsearch on
depth_values = np.linspace(1,10,10)
lambda_values = np.hstack((np.array([0.0]), np.logspace(-4,0,5)))
learning_rates = np.linspace(0, 1, 50)

#values to extract from gridsearch
min_mse_test, min_r2_test, min_mse_train, min_r2_train = 1000, 0, 0, 0


# Setting up the dataset
bind_eng = Data()
bind_eng.set_binding_energies("mass16.txt")

# Tracing the article nuclei
bind_eng.find_indeces()

# Getting the total binding energies
bind_eng.z_flat = bind_eng.z_flat*bind_eng.A_numpy

# Scaling the data
bind_eng.data_scaling()


# Creating the design matrix. Workaround to create the XGB and DT design matrices
poly_deg = 1
bind_eng.design_matrix(poly_deg)
deleted_matrix = np.delete(bind_eng.X,0,1)
bind_eng.X = deleted_matrix

# Picking AME16 data without article values
bind_eng.set_new_training_set()

# Picking article values only
bind_eng.set_new_test_set()

# Splitting into the test and training set
bind_eng.test_train_split(0.2)


fit = Fitting(bind_eng)

# -----------------------------------------------Analysis---------------------------------------------

#loop to find optimal values
for i, depth in enumerate(depth_values):
	for j, lamb in enumerate(lambda_values):
		for k, eta in enumerate(learning_rates):
			depth = int(depth)
			fit.XGB(max_depth=depth,reg_lambda=lamb, learning_rate=eta)

        # Tracking the optimal parameters
			if(MSE(bind_eng.z_test, fit.z_predict) <= min_mse_test):
				min_mse_test = MSE(bind_eng.z_test, fit.z_predict)
				min_r2_test = R2(bind_eng.z_test, fit.z_predict)
				min_mse_train = MSE(bind_eng.z_train, fit.z_tilde)
				min_r2_train = R2(bind_eng.z_train, fit.z_tilde)
				opt_depth = depth
				opt_lamb = lamb
				opt_eta = eta
				z_predict_article = fit.xgb_regression.predict(np.array(bind_eng.X_article))
				z_predict_total = fit.xgb_regression.predict(np.array(bind_eng.X_copy))
				z_predict = fit.z_predict
				z_tilde = fit.z_tilde
				z_plot = fit.z_plot


# Printing scores
print("--------------------------------")
print("Scores:")
print("Training MSE:")
print(MSE(bind_eng.z_train, z_tilde))
print("Training R2:")
print(R2(bind_eng.z_train, z_tilde))
print("Test MSE:")
print(MSE(bind_eng.z_test, z_predict))
print("Test R2:")
print(R2(bind_eng.z_test, z_predict))


# Printing scores for the article values
print("---------------- Article ----------------")
print("Optimal learning rate:")
print(opt_eta)
print("Optimal depth:")
print(opt_depth)
print("Optimal lambda:")
print(opt_lamb)
print("Scores:")
print("Article MSE:")
print(MSE(bind_eng.z_article, z_predict_article))
print("Article R2:")
print(R2(bind_eng.z_article, z_predict_article))


# Rescaling the data back
bind_eng.reset_test_and_train_back()
x_rescaled, y_rescaled, z_rescaled = bind_eng.data_rescaling_default(bind_eng.x_scaled, bind_eng.y_scaled, bind_eng.z_scaled)
x_rescaled, y_rescaled, z_rescaled_predict = bind_eng.data_rescaling_default(bind_eng.x_scaled, bind_eng.y_scaled, z_predict_total)

# Printing scores for the article values, sigma_{rms}:
print("---------------- RMSD ----------------")
print(bind_eng.article_scores(z_rescaled, z_rescaled_predict))
