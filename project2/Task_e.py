# The main program is used to collecting results for the Franke's function
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from imageio import imread

from sklearn.model_selection import train_test_split
import scipy.linalg as scl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
import sklearn.linear_model as skl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression

from statistical_functions import *
from data_processing import data
from print_and_plot import *
from regression_methods import fitting
from resampling_methods import resampling
from neuralnetwork import NeuralNetwork
from activation_functions import *

from sklearn.neural_network import MLPClassifier #For Classification
from sklearn.neural_network import MLPRegressor #For Regression
from keras_NN import Create_NeuralNetwork_Keras

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
from tensorflow.keras.models import Model
from tensorflow import keras
from keras.regularizers import l2

from sklearn import datasets # MNIST dataset

# -----------------------------------------------Logistic regression---------------------------------------------

from sklearn import datasets


# Ensure the same random numbers appear every time
np.random.seed(2020)

plt.rcParams['figure.figsize'] = (5,5)

# Download MNIST dataset
digits = datasets.load_digits()

# Define inputs and labels
inputs = digits.images
labels = digits.target

# Flatten the image
# The value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)

# Choose some random images to display
indices = np.arange(n_inputs)
random_indices = np.random.choice(indices, size=5)

# Check images
for i, image in enumerate(digits.images[random_indices]):
    plt.subplot(1, 5, i+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Label: %d" % digits.target[random_indices[i]])
#plt.show()

# Splitting into the test and train datasets
X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=0.8,test_size=0.2)

n_inputs, n_features = X_train.shape

# Converting into the one-hot representation
digits = data()
Y_train_onehot, Y_test_onehot = digits.to_categorical_numpy(Y_train), digits.to_categorical_numpy(Y_test)

# For SGD
epochs = 100
batch_size = 50
eta = 0.01
lamb = 0.0
n_categories = 10

#For the GD
Niterations = 10000

option = "SGD" #"SGD", "GD", "SKL Logistic"

model = fitting(digits)

if(option == "GD"):

	y_pred, y_tilde = model.Logistic_Regression(X_train, X_test, Y_train_onehot, Niterations = 100000, eta = 0.001, option = "GD", lamb = lamb)
	print("Training accuracy: {:.10f}".format(accuracy_score(Y_train, y_tilde)))
	print("Test accuracy: {:.10f}".format(accuracy_score(Y_test, y_pred)))

elif(option == "SGD"):

	y_pred, y_tilde = model.Logistic_Regression(X_train, X_test, Y_train_onehot, epochs = 100, eta = 0.01, option = "SGD", lamb = lamb)
	print("Training accuracy: {:.10f}".format(accuracy_score(Y_train, y_tilde)))
	print("Test accuracy: {:.10f}".format(accuracy_score(Y_test, y_pred)))

elif(option == "SKL Logistic"):

	if(lamb>0):
		logreg = LogisticRegression(penalty='l2', solver='lbfgs')
	else:
		logreg = LogisticRegression(penalty='none', solver='lbfgs')
	logreg.fit(X_train, Y_train)
	print("Test set accuracy with SKL Logistic Regression: {:.4f}".format(logreg.score(X_train,Y_train)))
	print("Test set accuracy with SKL Logistic Regression: {:.4f}".format(logreg.score(X_test,Y_test)))

