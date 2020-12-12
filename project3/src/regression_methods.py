import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import sklearn.linear_model as skl
from sklearn.linear_model import SGDRegressor
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from .statistical_functions import *


class Fitting():
	def __init__(self, inst):
		''' Fitting class containing OLS and ridge with matrix inversion,
		SGD, GD, SKL SGD, XGB and DT'''
		self.inst = inst

	def OLS(self):
		''' OLS with matrix inversion'''
		inst = self.inst
		beta = np.linalg.pinv(inst.X_train.T.dot(inst.X_train)).dot(inst.X_train.T.dot(inst.z_train))
		self.z_predict = inst.X_test.dot(beta)
		self.z_tilde = inst.X_train.dot(beta)
		self.z_plot = inst.X.dot(beta)
		self.beta = beta

	def ridge(self, lamb):
		''' ridge with matrix inversion'''
		inst = self.inst
		I = np.identity(len(inst.X_train[0,:]))
		beta = np.linalg.pinv(inst.X_train.T.dot(inst.X_train)+lamb*I).dot(inst.X_train.T.dot(inst.z_train))
		self.z_predict = inst.X_test.dot(beta)
		self.z_tilde = inst.X_train.dot(beta)
		self.z_plot = inst.X.dot(beta)
		self.beta = beta

	def LASSO_SKL(self, lamb):
		inst = self.inst
		lasso_fit = Lasso(alpha = lamb, max_iter = 10e5, tol = 0.01, normalize = True, fit_intercept = False)
		#lasso_fit = skl.Lasso(alpha=lamb, max_iter=10e6, tol=1e-3, normalize=True, fit_intercept=False).fit(inst.X_train,inst.z_train)
		lasso_fit.fit(inst.X_train,inst.z_train)
		self.z_predict = lasso_fit.predict(inst.X_test)
		self.z_tilde = lasso_fit.predict(inst.X_train)
		self.z_plot = lasso_fit.predict(inst.X)

	def SGD(self, n_epochs, t0, t1, seed, regression_method, lamb):
		''' Stochastic gradient descent for a given number of epochs and eta=t0/t1'''
		np.random.seed(seed)

		inst = self.inst
		len_beta = len(inst.X_train[0,:])
		beta = np.ones(len_beta)

		# Looping over number of epochs
		for epoch in range(n_epochs):
			for i in range(inst.n_minibatch):
				k = np.random.randint(inst.n_minibatch) #Pick the k-th minibatch at random
				Xk = inst.split_matrix_train[k]
				zk = inst.split_z_train[k]
				if(regression_method =="OLS"):
					gradients = 2/inst.min_size * Xk.T @ ((Xk @ beta)-zk)
				elif(regression_method == "Ridge"):
					gradients = 2/inst.min_size * Xk.T @ ((Xk @ beta)-zk)+2*lamb*beta

				#eta = t0/(epoch*inst.NumMin+i+t1)
				eta = t0/(t1)
				beta = beta - eta*gradients

		self.z_tilde  = inst.X_train.dot(beta)
		self.z_predict = inst.X_test.dot(beta)
		self.z_plot = inst.X.dot(beta)
		self.beta = beta

	def SGD_SKL(self, eta, seed, regression_method, lamb):
		''' Scikit-learn based stochastic gradient descent'''
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
		''' Standard gradient descent for a given number of epochs and eta'''
		np.random.seed(seed)

		inst = self.inst
		len_beta = len(inst.X_train[0,:])
		beta = np.ones(len_beta)
		len_z_train = len(inst.z_train)
		y = inst.z_train[:,np.newaxis]

		for i in range(Niterations):
			if(regression_method == "OLS"):
				gradients = 2.0/len_z_train*inst.X_train.T @ ((inst.X_train @ beta)-inst.z_train)
			elif(regression_method == "Ridge"):
				gradients = 2.0/len_z_train*inst.X_train.T @ ((inst.X_train @ beta)-inst.z_train)+2*lamb*beta
			beta = beta - eta*gradients

		self.z_tilde  = inst.X_train.dot(beta)
		self.z_predict = inst.X_test.dot(beta)
		self.z_plot = inst.X.dot(beta)
		self.beta = beta

	def logistic_regression(
		self, X, X_test,
		y, Niterations = 100000, eta = 0.001,
		option = "GD", epochs = 100, lamb = 0.001):
		''' Logistic gradient descent taking two options: "SGD" and "GD"'''
		x1 = X.shape[0]
		x2 = X.shape[1]

		y1 = y.shape[0]
		y2 = y.shape[1]

		beta = np.ones((x2,y2))

		if(option == "GD"):
			''' GD-based logistic regression'''
			for i in range(Niterations):
				y_curr = X @ beta
				probability = softmax(y_curr)
				inv_probability = 1 - softmax(y_curr)
				gradients =  - X.T @ (y - probability) + 2*lamb*beta
				beta -= eta*gradients * 2./(y1*y2)

			y_tilde = softmax(X @ beta)
			y_pred = softmax(X_test @ beta)
			return np.argmax(y_pred, axis=1), np.argmax(y_tilde, axis=1)

		if(option == "SGD"):
			''' SGD-based logistic regression'''
			n_inputs = X.shape[0]
			batch_size = 50

			iterations = n_inputs // batch_size

			data_indices = np.arange(n_inputs)

			for i in range(epochs):
				for j in range(iterations):
					chosen_datapoints = np.random.choice(data_indices, batch_size, replace=False)
					X_iter = X[chosen_datapoints]
					y_iter = y[chosen_datapoints]
					y_curr = X_iter @ beta
					probability = softmax(y_curr)
					inv_probability = 1 - softmax(y_curr)
					gradients =  - X_iter.T @ (y_iter - probability) + 2*lamb*beta
					beta -= eta*gradients * 2./(batch_size*y2)
			y_tilde = softmax(X @ beta)
			y_pred = softmax(X_test @ beta)
			return np.argmax(y_pred, axis=1), np.argmax(y_tilde, axis=1)

	def XGB(self, max_depth=6, learning_rate=0.3, reg_lambda=1):
		"""
		eXtra Gradient Boost method
		Keywords:
		max_depth (int): max depth of decision tree (default = 6)
		lamb: L2 regularization term (default = 1)
		learning_rate: step size shrinkage (default = 0.3)
		"""
		inst = self.inst
		self.xgb_regression = xgb.XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, reg_lambda=reg_lambda)
		self.xgb_regression.fit(inst.X_train, inst.z_train)
		self.z_tilde = self.xgb_regression.predict(inst.X_train)
		self.z_predict = self.xgb_regression.predict(inst.X_test)
		self.z_plot = self.xgb_regression.predict(inst.X)


	def decision_tree(self, random_state=2020, depth=7, lamb=0.0):
		'''
		Decision tree for regression using SKL.
		Keywords:
		depth - takes value 7 as default, but is altered to change the max depth.
		lamb - defaults 0.0. Is used by SKL for the ccp_alpha parameter,
			   i.e. the pruning parameter lambda in our code and report.
		'''
		inst = self.inst
		regr=DecisionTreeRegressor(criterion='mse', max_depth=int(depth), ccp_alpha=lamb)
		regr.fit(inst.X_train, inst.z_train)
		self.z_tilde = regr.predict(inst.X_train)
		self.z_predict = regr.predict(inst.X_test)
		self.z_plot = regr.predict(inst.X)
