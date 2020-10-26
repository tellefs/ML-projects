import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import sklearn.linear_model as skl
from statistical_functions import *
from sklearn.linear_model import SGDRegressor

class fitting():

	def __init__(self, inst):
		self.inst = inst

	def OLS(self):

		inst = self.inst
		beta = np.linalg.pinv(inst.X_train.T.dot(inst.X_train)).dot(inst.X_train.T.dot(inst.z_train))
		self.z_predict = inst.X_test.dot(beta)
		self.z_tilde = inst.X_train.dot(beta)
		self.z_plot = inst.X.dot(beta)
		self.beta = beta

	def Ridge(self, lamb):

		inst = self.inst
		I = np.identity(len(inst.X_train[0,:]))
		beta = np.linalg.pinv(inst.X_train.T.dot(inst.X_train)+lamb*I).dot(inst.X_train.T.dot(inst.z_train))
		self.z_predict = inst.X_test.dot(beta)
		self.z_tilde = inst.X_train.dot(beta)
		self.z_plot = inst.X.dot(beta)
		self.beta = beta

	def SGD(self, n_epochs, t0, t1, seed, regression_method, lamb):

		np.random.seed(seed)

		inst = self.inst
		len_beta = len(inst.X_train[0,:])
		beta = np.ones(len_beta)

		for epoch in range(n_epochs):
			for i in range(inst.NumMin):
				k = np.random.randint(inst.NumMin) #Pick the k-th minibatch at random
				Xk = inst.split_matrix_train[k]
				zk = inst.split_z_train[k]
				if(regression_method =="OLS"):
					gradients = 2/inst.Min * Xk.T @ ((Xk @ beta)-zk)
				elif(regression_method == "Ridge"):
					gradients = 2/inst.Min * Xk.T @ ((Xk @ beta)-zk)+2*lamb*beta

				#eta = t0/(epoch*inst.NumMin+i+t1)
				eta = t0/(t1)
				beta = beta - eta*gradients

		self.z_tilde  = inst.X_train.dot(beta)
		self.z_predict = inst.X_test.dot(beta)
		self.z_plot = inst.X.dot(beta)
		self.beta = beta

	def SGD_SKL(self, eta, seed, regression_method, lamb):

		np.random.seed(seed)

		inst = self.inst
		if(regression_method == "OLS"):
			sgdreg = SGDRegressor(max_iter = 1000, penalty=None, eta0=eta, fit_intercept = True)
		elif(regression_method == "Ridge"):
			sgdreg = SGDRegressor(max_iter = 1000, penalty='l2', alpha=lamb, eta0=eta, fit_intercept = True)

		sgdreg.fit(inst.X_train, inst.z_train.ravel())

		self.z_tilde = inst.X_train.dot(sgdreg.coef_)+sgdreg.intercept_
		self.z_predict = inst.X_test.dot(sgdreg.coef_)+sgdreg.intercept_
		self.z_plot = inst.X.dot(sgdreg.coef_)+sgdreg.intercept_
		self.beta = sgdreg.coef_

	def GD(self, Niterations, eta, seed, regression_method, lamb):

		np.random.seed(seed)

		inst = self.inst
		len_beta = len(inst.X_train[0,:])
		beta = np.ones(len_beta)
		len_z_train = len(inst.z_train)
		y = inst.z_train[:,np.newaxis]

		for i in range(Niterations):
			if(regression_method == "OLS"):
				gradients = 2.0/len_z_train*inst.X_train.T @ ((inst.X_train @ beta)-inst.z_train)
				#gradients = 2.0/len_z_train*inst.X_train.T.dot(inst.X_train.dot(beta)-inst.z_train)
			elif(regression_method == "Ridge"):
				gradients = 2.0/len_z_train*inst.X_train.T @ ((inst.X_train @ beta)-inst.z_train)+2*lamb*beta
				#gradients = 2.0/len_z_train*inst.X_train.T.dot(inst.X_train.dot(beta)-inst.z_train)+2*lamb*beta
			beta = beta - eta*gradients

		self.z_tilde  = inst.X_train.dot(beta)
		self.z_predict = inst.X_test.dot(beta)
		self.z_plot = inst.X.dot(beta)
		self.beta = beta

""" Uncomment if we need something else

def OLS_SVD(X_train, z_train, X_test):
	beta = ols_svd(X_train, z_train)
	z_predict = X_test.dot(beta)
	return z_predict

def OLS_SVD_beta(X_train, z_train):
	return ols_svd(X_train, z_train)

def Ridge(X_train, z_train, X_test, lamb, pol_deg):
	I = np.identity(int((pol_deg+2)*(pol_deg+1)/2))
	beta = np.linalg.pinv(X_train.T.dot(X_train)+lamb*I).dot(X_train.T.dot(z_train))
	z_predict = X_test.dot(beta)
	return z_predict

def Ridge_beta(X_train, z_train, lamb, pol_deg):
	I = np.identity(int((pol_deg+2)*(pol_deg+1)/2))
	return np.linalg.pinv(X_train.T.dot(X_train)+lamb*I).dot(X_train.T.dot(z_train))


#def Ridge_SKL_beta():


def LASSO_SKL(X_train, z_train, X_test, lamb):
	lasso_fit = skl.Lasso(alpha=lamb, max_iter=10e5, tol=1e-6, normalize=True, fit_intercept=False).fit(X_train,z_train)
	z_predict = lasso_fit.predict(X_test)
	beta = lasso_fit.coef_
	return z_predict

def LASSO_SKL_beta(X_train, z_train, lamb):
	lasso_fit = skl.Lasso(alpha=lamb, max_iter=10e5, tol=1e-6, normalize=True, fit_intercept=False).fit(X_train,z_train)
	return lasso_fit.coef_
"""