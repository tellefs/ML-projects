# This program contains all resampling methods

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed

from sklearn.model_selection import train_test_split
import scipy.linalg as scl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
import sklearn.linear_model as skl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from .statistical_functions import *
from .data_processing import *
from .print_and_plot import *
from .regression_methods import *

seed = 2021

class Resampling():
	def __init__(self, inst):
		''' Resampling class containing k-fold cross-validation or no-resampling option'''
		self.inst = inst

	def cross_validation(
		self, nrk, seed, minimization_method,
		regression_method, poly_deg, n_epochs=1000, t0=1,
		t1=1000, eta=0.001, lamb=0.001, min_size=5,
		n_iterations=10000):
		''' k-fold cross-validation
		minimization_method takes values "matrix_inv", "SGD", "GD",  "SGD_SKL"
		regression_method takes values "OLS" and "Ridge" '''
		np.random.seed(seed)

		inst = self.inst
		inst.design_matrix(poly_deg)
		inst.split_minibatch(False, nrk)

		MSError_test_CV = np.zeros(nrk)
		MSError_train_CV = np.zeros(nrk)
		R2_test_CV = np.zeros(nrk)
		R2_train_CV = np.zeros(nrk)

		z_plot_stored = np.zeros(len(inst.z_scaled))
		z_tilde_stored = np.zeros(int(len(inst.z_scaled)*(nrk-1)/nrk))
		z_predict_stored = np.zeros(int(len(inst.z_scaled)/nrk))

		fit = Fitting(inst)

		# loop over folds
		for k in range(nrk):

			inst.X_test = inst.split_matrix[k]
			inst.z_test = inst.split_z[k]

			X_train = inst.split_matrix
			X_train = np.delete(X_train,k,0)
			inst.X_train = np.concatenate(X_train)

			z_train = inst.split_z
			z_train = np.delete(z_train,k,0)
			inst.z_train = np.ravel(z_train)

			# refression
			if(minimization_method == "matrix_inv"):
				if(regression_method == "Ridge"):
					fit.ridge(lamb)
				elif(regression_method == "OLS"):
					fit.OLS()
			elif(minimization_method == "SGD"):
				n_minibatch = int(len(inst.z_train)/min_size)
				inst.split_minibatch(True, n_minibatch)
				fit.SGD(n_epochs, t0, t1, seed, regression_method, lamb)
			elif(minimization_method == "SGD_SKL"):
				fit.SGD_SKL(eta, seed, regression_method, lamb)
			elif(minimization_method == "GD"):
				fit.GD(n_iterations, eta, seed, regression_method, lamb)

			z_plot_stored = z_plot_stored + fit.z_plot
			z_tilde_stored = z_tilde_stored + fit.z_tilde
			z_predict_stored = z_predict_stored + fit.z_predict

			# Calculating scores
			R2_test_CV[k] = R2(inst.z_test,fit.z_predict)
			R2_train_CV[k] = R2(inst.z_train,fit.z_tilde)
			MSError_test_CV[k] = MSE(fit.z_predict,inst.z_test)
			MSError_train_CV[k] = MSE(fit.z_tilde,inst.z_train)

		self.MSE_test = np.mean(MSError_test_CV)
		self.MSE_train = np.mean(MSError_train_CV)
		self.R2_test = np.mean(R2_test_CV)
		self.R2_train = np.mean(R2_train_CV)
		self.z_predict = z_predict_stored/nrk
		self.z_tilde = z_tilde_stored/nrk
		self.z_plot = z_plot_stored/nrk



	def no_resampling(
		self, seed, minimization_method, regression_method,
		poly_deg, n_epochs=1000, t0=1, t1=1000,
		eta=0.001, lamb=0.001, min_size=5, n_iterations=10000):
		''' No resampling
		minimization_method takes values "matrix_inv", "SGD", "GD",  "SGD_SKL"
		regression_method takes values "OLS" and "Ridge" '''

		np.random.seed(seed)

		inst = self.inst
		inst.design_matrix(poly_deg)
		inst.test_train_split(0.2)
		fit = Fitting(inst)

		# regression
		if(minimization_method == "matrix_inv"):
			if(regression_method == "Ridge"):
				fit.ridge(lamb)
			elif(regression_method == "OLS"):
				fit.OLS()
		elif(minimization_method == "SGD"):
			n_minibatch = int(len(inst.z_train)/min_size)
			inst.split_minibatch(True, m_minibatch)
			fit.SGD(n_epochs, t0, t1, seed, regression_method, lamb)
		elif(minimization_method == "SGD_SKL"):
			fit.SGD_SKL(eta, seed, regression_method, lamb)
		elif(minimization_method == "GD"):
			fit.GD(n_iterations, eta, seed, regression_method, lamb)

		self.MSE_test = MSE(fit.z_predict,inst.z_test)
		self.MSE_train = MSE(fit.z_tilde,inst.z_train)
		self.R2_test = R2(inst.z_test,fit.z_predict)
		self.R2_train = R2(inst.z_train,fit.z_tilde)
		self.z_predict = fit.z_predict
		self.z_tilde = fit.z_tilde
		self.z_plot = fit.z_plot
