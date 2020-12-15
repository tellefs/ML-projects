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

The folowing program performs the simple analysis and the grid search for the FFNN and linear regression 
on the total binding energy dataset and extracts the scores for the nuclei presented in the article (see project text)
separately.

"""

np.random.seed(2021)

lamb = 0.001
task = "grid search" # "simple analysis", "grid search"
analysis_type = "linear" # "NN", "Linear"
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

# ----------------------------------------------- Simple analysis ---------------------------------------------
# Simple collection of the scores for the NN and the linear regression for the chosen parameters
if(task == "simple analysis"):

	# Setting matrices for the grid search
	poly_deg = 3
	bind_eng.design_matrix(poly_deg)

	# Picking AME16 data without article values
	bind_eng.set_new_training_set()

	# Picking article values only
	bind_eng.set_new_test_set()

	# Splitting into the test and training set
	bind_eng.test_train_split(0.2)

	if(analysis_type == "NN"):
		n_hidd_layers = 4
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
	x_rescaled, y_rescaled, z_rescaled = bind_eng.data_rescaling_default(
														bind_eng.x_scaled, 
														bind_eng.y_scaled, 
														bind_eng.z_scaled)
	x_rescaled, y_rescaled, z_rescaled_predict = bind_eng.data_rescaling_default(
														bind_eng.x_scaled, 
														bind_eng.y_scaled, 
														z_predict_total)
	
	# Printing scores for the article values
	print("---------------- RMSD ----------------")
	print(bind_eng.article_scores(z_rescaled, z_rescaled_predict))

# ----------------------------------------------- Grid search ---------------------------------------------
# Grid search analysis for the FFNN and linear regression (ridge)
elif(task == "grid search"):
	if(analysis_type == "linear"):
		# Setting matrices and arrays for the grid search
		num_poly = 23
		poly_deg_array = np.linspace(1, num_poly, num_poly, dtype = int)
		lambda_array = [ 0, 0.0000001, 0.000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
		num_lambda = len(lambda_array)
		min_el, min_pol, min_lamb, imin= 1000, 0, 0, 0
		R2_test, R2_val, R2_train = 0, 0, 0
		MSE_test, MSE_val, MSE_train = 0, 0, 0
	
		for poly_deg in poly_deg_array:
	
			for i in range(num_lambda):
	
				# Creating the design matrix
				bind_eng.design_matrix(poly_deg)
	
				# Picking AME16 data without article values
				bind_eng.set_new_training_set()
	
				# Picking article values only
				bind_eng.set_new_test_set()
	
				# Splitting into the test and training set
				bind_eng.test_train_split(0.2)
	
				fit = Fitting(bind_eng)
				fit.ridge(lambda_array[i])
				z_predict = fit.z_predict
				z_tilde = fit.z_tilde
				z_plot = fit.z_plot
				z_predict_article = np.array(bind_eng.X_article).dot(fit.beta)
				z_predict_total = np.array(bind_eng.X_copy).dot(fit.beta)
	
				# Rescaling the data back
				bind_eng.reset_test_and_train_back()
				x_rescaled, y_rescaled, z_rescaled = bind_eng.data_rescaling_default(
																			bind_eng.x_scaled, 
																			bind_eng.y_scaled, 
																			bind_eng.z_scaled)
				x_rescaled, y_rescaled, z_rescaled_predict = bind_eng.data_rescaling_default(
																			bind_eng.x_scaled, 
																			bind_eng.y_scaled, 
																			z_predict_total)
				score = bind_eng.article_scores(z_rescaled, z_rescaled_predict)
	
				# Tracking the optimal parameters
				if(score <= min_el):
					min_el = score
					min_pol = poly_deg
					min_lamb = lambda_array[i]
					imin = i
					R2_train = R2(bind_eng.z_train, z_tilde)
					R2_val = R2(bind_eng.z_test, z_predict)
					R2_test = R2(bind_eng.z_article, z_predict_article)
					MSE_train = MSE(bind_eng.z_train, z_tilde)
					MSE_val = MSE(bind_eng.z_test, z_predict)
					MSE_test = MSE(bind_eng.z_article, z_predict_article)
	
		# Printing the optimal values
		print("Optimal polynomial degree:")	
		print(min_pol)
		print("Optimal lambda:")	
		print(min_lamb)
		print("Minimum RMSD:")
		print(min_el)

		# Printing scores
		print("--------------------------------")
		print("Scores:")
		print("Training MSE:")
		print(MSE_train)
		print("Training R2:")
		print(R2_train)
		print("Test MSE:")
		print(MSE_val)
		print("Test R2:")
		print(R2_val)
	
		# Printing scores for the article values
		print("---------------- Article ----------------")
		print("Scores:")
		print("Article MSE:")
		print(MSE_test)
		print("Article R2:")
		print(R2_test)

	if(analysis_type == "NN"):
		# Setting matrices and arrays for the grid search
		eta_array = [0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0.00001, 0.000001]
		lambda_array = [ 0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
		num_lambda = len(lambda_array)
		num_eta = len(eta_array)
		min_el, min_pol, min_lamb, imin = 1000, 0, 0, 0
		R2_test, R2_val, R2_train = 0, 0, 0
		MSE_test, MSE_val, MSE_train = 0, 0, 0

		n_hidd_layers = 4
		nodes_in_hidd_layers = [90, 90, 90, 80]
		min_size = 100 	#size of each minibatch
		n_epochs = 1000	#number of epochs
		eta = 0.0001 	# learning rate
		activation_function_hidd = "sigmoid" # "tanh", "sigmoid"

		for i in range(num_eta):
			for j in range(num_lambda):

				# Setting matrices for the grid search
				poly_deg = 3

				bind_eng.design_matrix(poly_deg)

				# Picking AME16 data without article values
				bind_eng.set_new_training_set()

				# Picking article values only
				bind_eng.set_new_test_set()

				# Splitting into the test and training set
				bind_eng.test_train_split(0.2)

				NN = NeuralNetwork(
					bind_eng.X_train, 
					bind_eng.z_train, 
					n_hidden_layers = n_hidd_layers, 
					n_hidden_neurons = nodes_in_hidd_layers, 
					epochs = n_epochs, 
					batch_size = min_size, 
					eta = eta_array[i], 
					lmbd=lambda_array[j], 
					hidden_act_func=activation_function_hidd)
				NN.train()

				z_predict = np.ravel(NN.predict(bind_eng.X_test))
				z_tilde = np.ravel(NN.predict(bind_eng.X_train))
				z_plot = np.ravel(NN.predict(bind_eng.X))
				z_predict_article = np.ravel(NN.predict(bind_eng.X_article))
				z_predict_total = np.ravel(NN.predict(bind_eng.X_copy))

				# Rescaling the data back
				bind_eng.reset_test_and_train_back()
				x_rescaled, y_rescaled, z_rescaled = bind_eng.data_rescaling_default(
																			bind_eng.x_scaled, 
																			bind_eng.y_scaled, 
																			bind_eng.z_scaled)
				x_rescaled, y_rescaled, z_rescaled_predict = bind_eng.data_rescaling_default(
																			bind_eng.x_scaled, 
																			bind_eng.y_scaled, 
																			z_predict_total)		
				score = bind_eng.article_scores(z_rescaled, z_rescaled_predict)

				# Tracking the optimal parameters
				if(score <= min_el):
					min_el = score
					min_eta = eta_array[i]
					min_lamb = lambda_array[j]
					imin = i
					jmin = j
					R2_train = R2(bind_eng.z_train, z_tilde)
					R2_val = R2(bind_eng.z_test, z_predict)
					R2_test = R2(bind_eng.z_article, z_predict_article)
					MSE_train = MSE(bind_eng.z_train, z_tilde)
					MSE_val = MSE(bind_eng.z_test, z_predict)
					MSE_test = MSE(bind_eng.z_article, z_predict_article)

		# Printing the scores
		print("Optimal eta:")	
		print(min_eta)
		print("Optimal lambda:")	
		print(min_lamb)
		print("Minimum RMSD:")
		print(min_el)

		# Printing scores
		print("--------------------------------")
		print("Scores:")
		print("Training MSE:")
		print(MSE_train)
		print("Training R2:")
		print(R2_train)
		print("Test MSE:")
		print(MSE_val)
		print("Test R2:")
		print(R2_val)
	
		# Printing scores for the article values
		print("---------------- Article ----------------")
		print("Scores:")
		print("Article MSE:")
		print(MSE_test)
		print("Article R2:")
		print(R2_test)