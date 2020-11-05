import numpy as np
from random import random, seed

from statistical_functions import *
from data_processing import Data
from print_and_plot import *
from regression_methods import Fitting
from resampling_methods import Resampling

''' Task a

	The following file contains the code used to perform all studies 
	for the first task of the project. User defines whether k-fold
	cross validation is used or not by setting resampling_method to 
	"cv" or "no resampling".

	Depending on choice of study, either study of R2 and MSE is run for 
	different learning rates ("learning rate"), number of epochs ("epochs"),
	number of minibatches ("minibatches"), or ridge gridd search ("Ridge grid search")

	resampling_method: "cv", "no resampling "
	regression_method: "OLS", "Ridge"
	minimization_method: "matrix_inv", "SGD", "GD", "SGD_SKL"
	study: "learning rate", "epochs", "minibatches", "simple analysis", "Ridge grid search"
''' 

# Setting the Franke's function
poly_deg = 5
N = 30
seed = 2021
alpha = 0.001
lamb = 0.001

min_size = 5	#size of each minibatch
n_epochs = 1000		# number of epochs
t0, t1 = 1, 1000	# initial parameters for eta adjustable

eta = 0.001 	# constant eta
n_iterations = 100000	   # number of iterations for the GD

# user-defined settings
resampling_method = "cv" 	
regression_method ="Ridge" 	   
minimization_method = "SGD"    
study = "simple analysis"      

# Setting Data
franke_data = Data()
franke_data.set_grid_franke_function(N,N,False)
franke_data.set_franke_function()
franke_data.add_noise(alpha, seed)

# Scaling data
franke_data.data_scaling()

# Setting model
model = Resampling(franke_data)


if(study == "learning rate"): 
# -------------------------------------------------Learning rate OLS -----------------------------------------------
# Simple study of R2 and MSE dependence on the learning rate

	# Setting files
	filename_1 = "Files/Ridge_SGD_R2_0_1.txt"
	f_1 = open(filename_1, "w")
	f_1.write("eta   R2train  R2test\n")
	f_1.close()
	
	filename_2 = "Files/Ridge_SGD_MSE_0_1.txt"
	f_2 = open(filename_2, "w")
	f_2.write("eta   MSEtrain  MSEtest\n")
	f_2.close()

	t1_array = [
		500, 1000, 2000, 3000, 
		4000, 5000, 10000, 20000, 
		30000, 40000, 50000, 100000, 
		200000, 300000, 400000, 500000, 1000000
		]

	f_1 = open(filename_1, "a")
	f_2 = open(filename_2, "a")

	for t1 in t1_array:
		eta = t0/t1
		if(resampling_method == "cv"):
			model.cross_validation(
				5, seed, minimization_method, 
				regression_method, poly_deg, n_epochs=n_epochs, 
				t0=t0, t1=t1, eta=eta, 
				lamb=lamb, min_size=min_size, n_iterations=n_iterations)	

		elif(resampling_method == "no resampling"):
			model.no_resampling(
				seed, minimization_method, regression_method, 
				poly_deg, n_epochs=n_epochs, t0=t0,
				t1=t1, eta=eta, lamb=lamb, 
				min_size=min_size, n_iterations=n_iterations)

		f_1.write("{0} {1} {2}\n".format(eta, model.R2_train, model.R2_test))
		f_2.write("{0} {1} {2}\n".format(eta, model.MSE_train, model.MSE_test))
	f_1.close()
	f_2.close()

elif(study == "minibatches"): 
# ------------------------------------------------ Minibatch size -------------------------------------------------
# Study of R2 and MSE dependence on the minibatch size

	# Setting files
	filename_1 = "Files/Ridge_SGD_R2_minibatch_0_1.txt"
	f_1 = open(filename_1, "w")
	f_1.write("eta   R2train  R2test\n")
	f_1.close()
	
	filename_2 = "Files/Ridge_SGD_MSE_minibatch_0_1.txt"
	f_2 = open(filename_2, "w")
	f_2.write("eta   MSEtrain  MSEtest\n")
	f_2.close()
	
	minibatches = [1,5,10,15,20,30,45,60,90]
	
	f_1 = open(filename_1, "a")
	f_2 = open(filename_2, "a")

	for min_size in minibatches:

		if(resampling_method == "cv"):
			model.cross_validation(
				5, seed, minimization_method, 
				regression_method, poly_deg, n_epochs=n_epochs,
				t0=t0, t1=t1, eta=eta, 
				lamb=lamb, min_size=min_size, n_iterations=n_iterations)

		elif(resampling_method == "no resampling"):
			model.no_resampling(
				seed, minimization_method, regression_method, 
				poly_deg, n_epochs=n_epochs, t0=t0, 
				t1=t1, eta=eta, lamb=lamb, 
				min_size=min_size, n_iterations=n_iterations)

		f_1.write("{0} {1} {2}\n".format(min_size, model.R2_train, model.R2_test))
		f_2.write("{0} {1} {2}\n".format(min_size, model.MSE_train, model.MSE_test))
	f_1.close()
	f_2.close()

elif(study == "epochs"): 
# ------------------------------------------------ Number of epochs -------------------------------------------------
# Study of R2 and MSE dependence on the number of epochs

	# Setting files
	filename_1 = "Files/Ridge_SGD_R2_epoch_0_1.txt"
	f_1 = open(filename_1, "w")
	f_1.write("eta   R2train  R2test\n")
	f_1.close()
	
	filename_2 = "Files/Ridge_SGD_MSE_epoch_0_1.txt"
	f_2 = open(filename_2, "w")
	f_2.write("eta   MSEtrain  MSEtest\n")
	f_2.close()
	
	epochs_array = [10,100,1000,10000,100000]
	
	f_1 = open(filename_1, "a")
	f_2 = open(filename_2, "a")

	for n_epochs in epochs_array:

		if(resampling_method == "cv"):
			model.cross_validation(
				5, seed, minimization_method, regression_method, 
				poly_deg, n_epochs=n_epochs, t0=t0, 
				t1=t1, eta=eta, lamb=lamb, 
				min_size=min_size, n_iterations=n_iterations)

		elif(resampling_method == "no resampling"):
			model.no_resampling(
				seed, minimization_method, regression_method, 
				poly_deg, n_epochs=n_epochs, t0=t0, 
				t1=t1, eta=eta, lamb=lamb, 
				min_size=min_size, n_iterations=n_iterations)

		f_1.write("{0} {1} {2}\n".format(n_epochs, model.R2_train, model.R2_test))
		f_2.write("{0} {1} {2}\n".format(n_epochs, model.MSE_train, model.MSE_test))
		print(n_epochs)
	f_1.close()
	f_2.close()

elif(study == "Ridge grid search"): 
# ------------------------------------------------ Ridge grid search -------------------------------------------------
# Grid search for Ridge

	lambdas = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
	t1_array = [1/0.008, 1/0.006,1/0.004,1/0.002, 1/0.001, 1/0.0001] # for the fine grid search

	# Setting files
	filename_1 = "Files/Ridge_test_MSE_fine.txt"
	filename_2 = "Files/Ridge_test_R2_fine.txt"
	filename_3 = "Files/Ridge_train_MSE_fine.txt"
	filename_4 = "Files/Ridge_train_R2_fine.txt"
	
	f_1 = open(filename_1, "w")
	f_1.write("lambda   eta  MSEtest\n")
	f_1.close()
	f_2 = open(filename_2, "w")
	f_2.write("lambda   eta  R2test\n")
	f_2.close()
	f_3 = open(filename_3, "w")
	f_3.write("lambda   eta  MSEtrain\n")
	f_3.close()
	f_4 = open(filename_4, "w")
	f_4.write("lambda   eta  R2train\n")
	f_4.close()
	
	f_1 = open(filename_1, "a")
	f_2 = open(filename_2, "a")
	f_3 = open(filename_3, "a")
	f_4 = open(filename_4, "a")

	for lamb in lambdas:
		for t1 in t1_array:
			if(resampling_method == "cv"):
				model.cross_validation(
					5, seed, minimization_method, regression_method, 
					poly_deg, n_epochs=n_epochs, t0=t0, 
					t1=t1, eta=eta, lamb=lamb, 
					min_size=min_size, n_iterations=n_iterations)

			elif(resampling_method == "no resampling"):
				model.no_resampling(
					seed, minimization_method, regression_method, 
					poly_deg, n_epochs=n_epochs, t0=t0, 
					t1=t1, eta=eta, lamb=lamb, 
					min_size=min_size, n_iterations=n_iterations)

			f_1.write("{0} {1} {2}\n".format(lamb, 1.0/t1, model.MSE_test))
			f_2.write("{0} {1} {2}\n".format(lamb, 1.0/t1, model.R2_test))
			f_3.write("{0} {1} {2}\n".format(lamb, 1.0/t1, model.MSE_train))
			f_4.write("{0} {1} {2}\n".format(lamb, 1.0/t1, model.R2_train))
	f_1.close()
	f_2.close()
	f_3.close()
	f_4.close()

# ------------------------------------------------ Simple analysis -------------------------------------------------
# MSE and R2 for the initially chosen parameters
	
elif(study == "simple analysis"):


	if(resampling_method == "cv"):
		model.cross_validation(
			5, seed, minimization_method, regression_method, 
			poly_deg, n_epochs=n_epochs, t0=t0, 
			t1=t1, eta=eta, lamb=lamb, 
			min_size=min_size, n_iterations=n_iterations)

	elif(resampling_method == "no resampling"):
		model.no_resampling(
			seed, minimization_method, regression_method, 
			poly_deg, n_epochs=n_epochs, t0=t0, 
			t1=t1, eta=eta, lamb=lamb, 
			min_size=min_size, n_iterations=n_iterations)

	franke_data.z_scaled = model.z_plot # This is for plotting all data points at once

	# Rescale data
	franke_data.data_rescaling()

	# Printing all scores
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
	
	#Plotting the surface
	surface_plot(
		franke_data.x_rescaled, franke_data.y_rescaled, 
		franke_data.z_mesh, franke_data.z_rescaled)


