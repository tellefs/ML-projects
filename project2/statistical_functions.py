import numpy as np
import scipy.linalg as scl

''' File contains all used metrics'''

def R2(z_data, z_model):
	'''R2 score'''
	return 1 - np.sum((z_data - z_model) ** 2)/np.sum((z_data - np.mean(z_data)) ** 2)

def MSE(z_data,z_model):
	''' MSE'''
	n = np.size(z_model)
	return np.sum((z_data-z_model)**2)/n

def bias(z_data,z_model):
	'''Bias'''
	n = np.size(z_model)
	return np.sum((z_data-np.sum(z_model)/n)**2)/n

def variance(z):
	''' Variance'''
	n = np.size(z)
	return np.sum((z-np.sum(z)/n)**2)/n

def accuracy_score(Y_test, Y_pred):
	'''Accuracy for classification'''
	return np.sum(Y_test == Y_pred) / len(Y_test)+0.0