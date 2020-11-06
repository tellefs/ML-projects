import numpy as np
import matplotlib.pyplot as plt
from random import random, seed

from src.statistical_functions import *
from src.data_processing import Data
from src.print_and_plot import *
from src.regression_methods import Fitting
from src.resampling_methods import Resampling
from src.neuralnetwork import NeuralNetwork
from src.activation_functions import *

from sklearn.linear_model import LogisticRegression
from sklearn import datasets # MNIST dataset

''' Task e

	The following file contains the code used to perform all studies
	for the task e of the project. User defines which option for the logistic
	regression to run (SGD-based, GD-based of scikit-learn.

	option: "SGD", "GD", "SKL Logistic"
'''

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
digits = Data()
Y_train_onehot, Y_test_onehot = digits.to_categorical_numpy(Y_train), digits.to_categorical_numpy(Y_test)

# For SGD
epochs = 100
batch_size = 50
eta = 0.01
lamb = 0.0
n_categories = 10

#For the GD
n_iterations = 100000

# user defined option
option = "SGD"

# setting the model
model = Fitting(digits)

# --------------------------------------- GD based Logistic regression--------------------------------------
if(option == "GD"):

	y_pred, y_tilde = model.logistic_regression(
		X_train, X_test,
		Y_train_onehot,
		n_iterations = n_iterations,
		eta = 0.001,
		option = "GD",
		lamb = lamb)
	print("Training accuracy: {:.10f}".format(accuracy_score(Y_train, y_tilde)))
	print("Test accuracy: {:.10f}".format(accuracy_score(Y_test, y_pred)))

# --------------------------------------- SGD based Logistic regression--------------------------------------
elif(option == "SGD"):

	y_pred, y_tilde = model.logistic_regression(
		X_train,
		X_test,
		Y_train_onehot,
		epochs = epochs,
		eta = eta,
		option = "SGD",
		lamb = lamb)
	print("Training accuracy: {:.10f}".format(accuracy_score(Y_train, y_tilde)))
	print("Test accuracy: {:.10f}".format(accuracy_score(Y_test, y_pred)))

# --------------------------------------- SKL Logistic regression--------------------------------------
elif(option == "SKL Logistic"):

	if(lamb>0):
		logreg = LogisticRegression(penalty='l2', solver='lbfgs')
	else:
		logreg = LogisticRegression(penalty='none', solver='lbfgs')
	logreg.fit(X_train, Y_train)
	print("Test set accuracy with SKL Logistic Regression: {:.4f}".format(logreg.score(X_train,Y_train)))
	print("Test set accuracy with SKL Logistic Regression: {:.4f}".format(logreg.score(X_test,Y_test)))
