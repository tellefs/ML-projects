# All statistical functions used in all other files

import numpy as np
import scipy.linalg as scl

def R2(z_data, z_model):
	return 1 - np.sum((z_data - z_model) ** 2)/np.sum((z_data - np.mean(z_data)) ** 2)

def MSE(z_data,z_model):
	n = np.size(z_model)
	return np.sum((z_data-z_model)**2)/n

def Bias(z_data,z_model):
	n = np.size(z_model)
	return np.sum((z_data-np.sum(z_model)/n)**2)/n

def Variance(z):
	n = np.size(z)
	return np.sum((z-np.sum(z)/n)**2)/n

#SVD matrix inversion beta
def  ols_svd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    u, s, v = scl.svd(x)
    return v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ y

def BetaVar(X, alpha, sigma):
    return  sigma**2*np.diag(alpha**2*np.linalg.pinv(X.T.dot(X)))
