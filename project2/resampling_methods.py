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

from statistical_functions import *
from data_processing import *
from print_and_plot import *
from regression_methods import *

seed = 2021

class resampling():

	def __init__(self, inst):
		self.inst = inst

	def Cross_Validation(self, nrk, seed, minimization_method, regression_method, PolyDeg, n_epochs=1000, t0=1, t1=1000, eta=0.001, lamb=0.001, Min=5, Niterations=10000):

		np.random.seed(seed)

		inst = self.inst

		inst.DesignMatrix(PolyDeg)

		inst.SplitMinibatch(False, nrk)

		MSError_test_CV = np.zeros(nrk)
		MSError_train_CV = np.zeros(nrk)
		R2_test_CV = np.zeros(nrk)
		R2_train_CV = np.zeros(nrk)

		z_plot_stored = np.zeros(len(inst.z_scaled))
		z_tilde_stored = np.zeros(int(len(inst.z_scaled)*(nrk-1)/nrk))
		z_predict_stored = np.zeros(int(len(inst.z_scaled)/nrk))

		fit = fitting(inst)

		for k in range(nrk):

			inst.X_test = inst.split_matrix[k]
			inst.z_test = inst.split_z[k]

			X_train = inst.split_matrix
			X_train = np.delete(X_train,k,0)
			inst.X_train = np.concatenate(X_train)

			z_train = inst.split_z
			z_train = np.delete(z_train,k,0)
			inst.z_train = np.ravel(z_train)

			if(minimization_method == "matrix_inv"):
				if(regression_method == "Ridge"):
					fit.Ridge(lamb)
				elif(regression_method == "OLS"):
					fit.OLS()
			elif(minimization_method == "SGD"):
				NumMin = int(len(inst.z_train)/Min)
				inst.SplitMinibatch(True, NumMin)
				fit.SGD(n_epochs, t0, t1, seed, regression_method, lamb)
			elif(minimization_method == "SGD_SKL"):
				fit.SGD_SKL(eta, seed, regression_method, lamb)
			elif(minimization_method == "GD"):
				fit.GD(Niterations, eta, seed, regression_method, lamb)

			z_plot_stored = z_plot_stored + fit.z_plot
			z_tilde_stored = z_tilde_stored + fit.z_tilde
			z_predict_stored = z_predict_stored + fit.z_predict

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



	def NoResampling(self, seed, minimization_method, regression_method, PolyDeg, n_epochs=1000, t0=1, t1=1000, eta=0.001, lamb=0.001, Min=5, Niterations=10000):

		np.random.seed(seed)

		inst = self.inst
		inst.DesignMatrix(PolyDeg)
		inst.TestTrainSplit(0.2)
		fit = fitting(inst)

		if(minimization_method == "matrix_inv"):
			if(regression_method == "Ridge"):
				fit.Ridge(lamb)
			elif(regression_method == "OLS"):
				fit.OLS()
		elif(minimization_method == "SGD"):
			NumMin = int(len(inst.z_train)/Min)
			inst.SplitMinibatch(True, NumMin)
			fit.SGD(n_epochs, t0, t1, seed, regression_method, lamb)
		elif(minimization_method == "SGD_SKL"):
			fit.SGD_SKL(eta, seed, regression_method, lamb)
		elif(minimization_method == "GD"):
			fit.GD(Niterations, eta, seed, regression_method, lamb)

		self.MSE_test = MSE(fit.z_predict,inst.z_test)
		self.MSE_train = MSE(fit.z_tilde,inst.z_train)
		self.R2_test = R2(inst.z_test,fit.z_predict)
		self.R2_train = R2(inst.z_train,fit.z_tilde)
		self.z_predict = fit.z_predict
		self.z_tilde = fit.z_tilde
		self.z_plot = fit.z_plot
