import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Data():
	def __init__(self):
		''' Data class, used for both Franke's function and MNIST dataset'''
		self.x = {}
		self.y = {}

	def set_grid_franke_function(self, N, M, random):
		''' Setting grid with dimensions N and M'''
		self.N = N
		self.M = M

		if random:
			self.x = np.sort(np.random.uniform(0,1,N))
			self.y = np.sort(np.random.uniform(0,1,M))
		else:
			self.x = np.arange(0, 1, 1.0/N)
			self.y = np.arange(0, 1, 1.0/M)

		self.x_mesh, self.y_mesh = np.meshgrid(self.x,self.y)
		self.x_flat = np.ravel(self.x_mesh)
		self.y_flat = np.ravel(self.y_mesh)

	def set_franke_function(self):
		''' Setting the Franke's function based on the grid'''
		term1 = 0.75*np.exp(-(0.25*(9*self.x_mesh-2)**2) - 0.25*((9*self.y_mesh-2)**2))
		term2 = 0.75*np.exp(-((9*self.x_mesh+1)**2)/49.0 - 0.1*(9*self.y_mesh+1))
		term3 = 0.5*np.exp(-(9*self.x_mesh-7)**2/4.0 - 0.25*((9*self.y_mesh-3)**2))
		term4 = -0.2*np.exp(-(9*self.x_mesh-4)**2 - (9*self.y_mesh-7)**2)
		self.z_mesh = term1 + term2 + term3 + term4
		self.z_flat = np.ravel(self.z_mesh)

	def design_matrix(self, poly_deg):
		''' Setting design matrix for a given polynomial degree'''
		num_elem = int((poly_deg+2)*(poly_deg+1)/2)
		X = np.ones((len(self.x_scaled),num_elem))
		for i in range(1,poly_deg+1):
			j = int((i)*(i+1)/2)
			for k in range(i+1):
				X[:,j+k] = self.x_scaled**(i-k)*self.y_scaled**k
		self.X = X

	def add_noise(self, alpha, seed):
		''' Adding noise to Franke's function'''
		np.random.seed(seed)
		noise = alpha*np.random.randn(self.N, self.M)
		self.z_mesh = self.z_mesh+noise
		self.z_flat = np.ravel(self.z_mesh)

	def data_scaling(self):
		''' Scaling of the data'''
		# x and y must be raveled
		dataset = np.stack((self.x_flat, self.y_flat, self.z_flat)).T
		scaler = StandardScaler()
		scaler.fit(dataset)
		self.scaler = scaler
		[self.x_scaled, self.y_scaled, self.z_scaled] = scaler.transform(dataset).T

	def data_rescaling(self):
		''' Rescaling data back'''
		dataset_scaled = np.stack((self.x_scaled, self.y_scaled, self.z_scaled))
		dataset_rescaled = self.scaler.inverse_transform(dataset_scaled.T)
		self.x_rescaled = np.zeros((self.N,self.M))
		self.x_rescaled = np.reshape(dataset_rescaled[:,0], (self.N,self.M))
		self.y_rescaled = np.zeros((self.N,self.M))
		self.y_rescaled = np.reshape(dataset_rescaled[:,1], (self.N,self.M))
		self.z_rescaled = np.zeros((self.N,self.M))
		self.z_rescaled = np.reshape(dataset_rescaled[:,2], (self.N,self.M))

	def test_train_split(self, test_train):
		''' Splitting into the test and train datasets'''
		self.X_train, self.X_test, self.z_train, self.z_test = train_test_split(self.X, self.z_scaled, test_size=test_train)

	def split_minibatch(self, training, n_minibatch):
		''' Splitting of the dataset into the minibatches
		If training = True, then the methoid deals with the training set. 
		If False - with the whole dataset (for CV)''' 
		if training:
			z = self.z_train
			X = self.X_train
		else:
			z = self.z_scaled
			X = self.X

		# Shuffling the dataset
		shuffled_indices = np.arange(len(z))
		np.random.shuffle(shuffled_indices)

		shuffled_matrix = np.zeros(X.shape)
		shuffled_z = np.zeros(len(z))

		for i in range(len(z)):
			shuffled_matrix[i] = X[shuffled_indices[i]]
			shuffled_z[i] = z[shuffled_indices[i]]

		# Splitting into minibatches
		self.n_minibatch = n_minibatch # number of minibatches in the test set
		self.min_size = int(len(z)/n_minibatch) # size of a minibatch

		if training:
			self.split_matrix_train = np.split(shuffled_matrix,n_minibatch)
			self.split_z_train = np.split(shuffled_z,n_minibatch)
		else:
			self.split_matrix = np.split(shuffled_matrix,n_minibatch)
			self.split_z = np.split(shuffled_z,n_minibatch)

	def to_categorical_numpy(self, integer_vector):
		''' Transforming MNIST raw data into one-hot vectors'''
		n_inputs = len(integer_vector)
		n_categories = np.max(integer_vector) + 1
		onehot_vector = np.zeros((n_inputs, n_categories))
		onehot_vector[range(n_inputs), integer_vector] = 1
    
		return onehot_vector
