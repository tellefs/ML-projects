import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class data():

	def __init__(self):

		self.x = {}
		self.y = {}

	def SetGridFrankeFunction(self, N, M, random):

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

	def SetFrankeFunction(self):
		term1 = 0.75*np.exp(-(0.25*(9*self.x_mesh-2)**2) - 0.25*((9*self.y_mesh-2)**2))
		term2 = 0.75*np.exp(-((9*self.x_mesh+1)**2)/49.0 - 0.1*(9*self.y_mesh+1))
		term3 = 0.5*np.exp(-(9*self.x_mesh-7)**2/4.0 - 0.25*((9*self.y_mesh-3)**2))
		term4 = -0.2*np.exp(-(9*self.x_mesh-4)**2 - (9*self.y_mesh-7)**2)
		self.z_mesh = term1 + term2 + term3 + term4
		self.z_flat = np.ravel(self.z_mesh)

	def DesignMatrix(self, PolyDeg):

		NumElem = int((PolyDeg+2)*(PolyDeg+1)/2)
		X = np.ones((len(self.x_scaled),NumElem))
		for i in range(1,PolyDeg+1):
			j = int((i)*(i+1)/2)
			for k in range(i+1):
				X[:,j+k] = self.x_scaled**(i-k)*self.y_scaled**k
		self.X = X

	def AddNoise(self,alpha, seed):

		np.random.seed(seed)
		noise = alpha*np.random.randn(self.N, self.M)
		self.z_mesh = self.z_mesh+noise
		self.z_flat = np.ravel(self.z_mesh)

	def DataScaling(self):

		# x and y must be raveled
		dataset = np.stack((self.x_flat, self.y_flat, self.z_flat)).T
		scaler = StandardScaler()
		scaler.fit(dataset)
		self.scaler = scaler
		[self.x_scaled, self.y_scaled, self.z_scaled] = scaler.transform(dataset).T

	def DataRescaling(self):

		dataset_scaled = np.stack((self.x_scaled, self.y_scaled, self.z_scaled))
		dataset_rescaled = self.scaler.inverse_transform(dataset_scaled.T)
		self.x_rescaled = np.zeros((self.N,self.M))
		self.x_rescaled = np.reshape(dataset_rescaled[:,0], (self.N,self.M))
		self.y_rescaled = np.zeros((self.N,self.M))
		self.y_rescaled = np.reshape(dataset_rescaled[:,1], (self.N,self.M))
		self.z_rescaled = np.zeros((self.N,self.M))
		self.z_rescaled = np.reshape(dataset_rescaled[:,2], (self.N,self.M))

	def TestTrainSplit(self, test_train):

		self.X_train, self.X_test, self.z_train, self.z_test = train_test_split(self.X, self.z_scaled, test_size=test_train)

	def SplitMinibatch(self, training, NumMin):
		# If training = True, then the methoid deals with the training set. If False - with the whole dataset (for CV)

		if training:
			z = self.z_train
			X = self.X_train
		else:
			z = self.z_scaled
			X = self.X

		shuffled_indices = np.arange(len(z))
		np.random.shuffle(shuffled_indices)

		shuffled_matrix = np.zeros(X.shape)
		shuffled_z = np.zeros(len(z))

		for i in range(len(z)):
			shuffled_matrix[i] = X[shuffled_indices[i]]
			shuffled_z[i] = z[shuffled_indices[i]]

		self.NumMin = NumMin #number of minibatches in the test set
		self.Min = int(len(z)/NumMin) # Size of a minibatch

		if training:
			self.split_matrix_train = np.split(shuffled_matrix,NumMin)
			self.split_z_train = np.split(shuffled_z,NumMin)
		else:
			self.split_matrix = np.split(shuffled_matrix,NumMin)
			self.split_z = np.split(shuffled_z,NumMin)

