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

The folowing program performs the linear regression analysis with three options - OLS, Ridge and LASSO
The user might selesct one of the options by varying the regression_method variable taking values "OLS", "Ridge", "LASSO"
and switch one of the resampling options on by choosing between "cv" and "no resampling" for the resampling_method variable.
The output of the program is two grid search files for the test and training MSE and R2.

"""

np.random.seed(2021)

lamb = 0.001
analysis_type = "NN" # "NN", "Linear"
regression_method = "ridge" # "OLS", "ridge", "LASSO"

# Setting up the dataset
bind_eng = Data()
bind_eng.set_binding_energies("mass16.txt")

# Tracing the article nuclei
bind_eng.find_indeces()

# Getting the total binding energies
bind_eng.z_flat = bind_eng.z_flat*bind_eng.A_numpy

# Scaling the data
bind_eng.data_scaling()

# Setting matrices for the grid search
poly_deg = 3

bind_eng.design_matrix(poly_deg)

# Picking AME16 data without article values
bind_eng.set_new_training_set()

# Picking article values only
bind_eng.set_new_test_set()

# Splitting into the test and training set
bind_eng.test_train_split(0.2)

	# -----------------------------------------------Analysis---------------------------------------------

if(analysis_type == "NN"):
	n_hidd_layers = 4
	#nodes_in_hidd_layers = [90, 90, 90, 80]
	nodes_in_hidd_layers = [90, 90, 90, 80]
	min_size = 100 	#size of each minibatch
	n_epochs = 1000	#number of epochs
	eta = 0.0001 	# learning rate
	activation_function_hidd = "sigmoid" # "tanh", "sigmoid"
	
	NN = NeuralNetwork(
			bind_eng.X_train, 
			bind_eng.z_train, 
			n_hidden_layers = n_hidd_layers, 
			n_hidden_neurons = nodes_in_hidd_layers, 
			epochs = n_epochs, 
			batch_size = min_size, 
			eta = eta, 
			lmbd=lamb, 
			hidden_act_func=activation_function_hidd)
	NN.train()
	
	z_predict = np.ravel(NN.predict(bind_eng.X_test))
	z_tilde = np.ravel(NN.predict(bind_eng.X_train))
	z_plot = np.ravel(NN.predict(bind_eng.X))
	z_predict_article = np.ravel(NN.predict(bind_eng.X_article))
	z_predict_total = np.ravel(NN.predict(bind_eng.X_copy))

elif(analysis_type == "linear"):
	fit = Fitting(bind_eng)
	if(regression_method == "OLS"):
		fit.OLS()
	elif(regression_method == "ridge"):
		fit.ridge(lamb)
	elif(regression_method == "LASSO"):
		fit.LASSO_SKL(lamb)
	z_predict = fit.z_predict
	z_tilde = fit.z_tilde
	z_plot = fit.z_plot
	z_predict_article = np.array(bind_eng.X_article).dot(fit.beta)
	z_predict_total = np.array(bind_eng.X_copy).dot(fit.beta)

	
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
print("Scores:")
print("Article MSE:")
print(MSE(bind_eng.z_article, z_predict_article))
print("Article R2:")
print(R2(bind_eng.z_article, z_predict_article))


# Rescaling the data back
bind_eng.reset_test_and_train_back()
x_rescaled, y_rescaled, z_rescaled = bind_eng.data_rescaling_default(bind_eng.x_scaled, bind_eng.y_scaled, bind_eng.z_scaled)
x_rescaled, y_rescaled, z_rescaled_predict = bind_eng.data_rescaling_default(bind_eng.x_scaled, bind_eng.y_scaled, z_predict_total)

# Printing scores for the article values
print("---------------- RMSD ----------------")
bind_eng.article_scores(z_rescaled, z_rescaled_predict)
