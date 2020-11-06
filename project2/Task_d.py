import matplotlib.pyplot as plt
import numpy as np
from random import random, seed

from src.statistical_functions import *
from src.data_processing import Data
from src.print_and_plot import *
from src.regression_methods import Fitting
from src.resampling_methods import Resampling
from src.neuralnetwork import NeuralNetwork
from src.activation_functions import *

from sklearn.neural_network import MLPClassifier #For Classification

from sklearn import datasets # MNIST dataset

''' Task d

	The following file contains the code used to perform all studies
	for the task d of the project. User defines which type of NN
	will be used (self made NN, SKL).

	Depending on choice of study, either study of accuracy is run for
	different number of hidden layers ("hidden layers"), number of epochs ("epochs"),
	number of neurons ("neurons"), or gridd search ("grid search")

	study: "simple analysis", "epochs", "hidden layers", "neurons", "grid search"
	activation_function: "sigmoid", "relu", "leaky relu", "tanh"
	NN_type = "self made", "skl"
'''

# -----------------------------------------------NN---------------------------------------------

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

# user defined settings
epochs = 100
batch_size = 50
eta = 0.01
lamb = 0.01
n_categories = 10

study = "simple analysis"
activation_function = "sigmoid"
NN_type = "self made"

# ---------------------------------------------- Simple analysis ---------------------------------------------
# Simple study of accuracy for different NN
if(study == "simple analysis"):
	if(NN_type == "self made"):

		# Setting number of hidden layers and neurons in each layer
		num_hidd_layers = 3
		nodes_in_hidd_layers = [50, 50, 50]

		NN = NeuralNetwork(
			X_train, Y_train_onehot,
			n_hidden_layers = num_hidd_layers, n_hidden_neurons = nodes_in_hidd_layers,
			n_categories=n_categories, epochs=epochs,
			batch_size=batch_size, eta=eta,
			out_act_func = "softmax", hidden_act_func=activation_function,
			lmbd=lamb)

		NN.train()
		Y_predict = NN.predict_class(X_test)
		Y_tilde = NN.predict_class(X_train)

	elif(NN_type == "skl"):

		NN = MLPClassifier(
			hidden_layer_sizes=(50,50,50), activation='logistic',
			alpha=lamb, learning_rate_init=eta,
			max_iter=1000, solver='sgd',
			batch_size = batch_size)

		NN.fit(X_train, Y_train)
		Y_predict = NN.predict(X_test)
		Y_tilde = NN.predict(X_train)

	print("Accuracy score on test set: ", accuracy_score(Y_test, Y_predict))
	print("%.4f" % accuracy_score(Y_test, Y_predict))
	print("Accuracy score on training set: ", accuracy_score(Y_train, Y_tilde))
	print("%.4f" % accuracy_score(Y_train, Y_tilde))

# ---------------------------------------------- Hidden layers ---------------------------------------------
# Study of accuracy dependence on the number of hidden layers
if(study == "hidden layers"):

	# Setting number of hidden layers and neurons in each layer
	num_hidd_layers = 1
	nodes_in_hidd_layers = [50]

	# Setting files
	filename = "Files/Class_hidd_layers.txt"
	f = open(filename, "w")
	f.write("num_hidd_layers   Acctrain  Acctest\n")
	f.close()

	f = open(filename, "a")

	for i in range(12):
		NN = NeuralNetwork(
			X_train, Y_train_onehot,
			n_hidden_layers = num_hidd_layers, n_hidden_neurons = nodes_in_hidd_layers,
			n_categories=n_categories, epochs=epochs,
			batch_size=batch_size, eta=eta,
			out_act_func = "softmax",hidden_act_func=activation_function,
			lmbd=lamb)

		NN.train()
		Y_predict = NN.predict_class(X_test)
		Y_tilde = NN.predict_class(X_train)

		f.write("{0} {1} {2}\n".format(num_hidd_layers, accuracy_score(Y_train, Y_tilde), accuracy_score(Y_test, Y_predict)))
		num_hidd_layers=num_hidd_layers+1
		nodes_in_hidd_layers.append(50)

	f.close()

# ---------------------------------------------- Number of neurons ---------------------------------------------
# Study of R2 and MSE dependence on the number of neurons
if(study == "neurons"):

	# Setting number of hidden layers and neurons in each layer
	num_hidd_layers = 1
	nodes_in_hidd_layers = [1]
	neurons = 1

	# Setting files
	filename = "Files/Class_hidd_neurons.txt"
	f = open(filename, "w")
	f.write("NumNeurons   Acctrain  Acctest\n")
	f.close()

	f = open(filename, "a")

	for i in range(50):
		NN = NeuralNetwork(
			X_train, Y_train_onehot,
			n_hidden_layers = num_hidd_layers, n_hidden_neurons = nodes_in_hidd_layers,
			n_categories=n_categories, epochs=epochs,
			batch_size=batch_size, eta=eta,
			out_act_func = "softmax",hidden_act_func=activation_function,
			lmbd=lamb)

		NN.train()
		Y_predict = NN.predict_class(X_test)
		Y_tilde = NN.predict_class(X_train)

		f.write("{0} {1} {2}\n".format(neurons, accuracy_score(Y_train, Y_tilde), accuracy_score(Y_test, Y_predict)))
		neurons=neurons+1
		nodes_in_hidd_layers[0]=neurons

	f.close()
if(study == "epochs"):
# ---------------------------------------------- Number of epochs ---------------------------------------------
# Study of accuracy dependence on the number of epochs

	# Setting number of hidden layers and neurons in each layer
	num_hidd_layers = 3
	nodes_in_hidd_layers = [50,50,50]

	# Setting files
	filename = "Files/Class_epochs.txt"
	f = open(filename, "w")
	f.write("epochs   Acctrain  Acctest\n")
	f.close()

	epochs = np.arange(20,1020,20)

	f = open(filename, "a")

	for ep in epochs:
		NN = NeuralNetwork(
			X_train, Y_train_onehot,
			n_hidden_layers = num_hidd_layers,
			n_hidden_neurons = nodes_in_hidd_layers,
			n_categories=n_categories, epochs=ep,
			batch_size=batch_size, eta=eta,
			out_act_func = "softmax",hidden_act_func=activation_function,
			lmbd=lamb)

		NN.train()
		Y_predict = NN.predict_class(X_test)
		Y_tilde = NN.predict_class(X_train)

		f.write("{0} {1} {2}\n".format(ep, accuracy_score(Y_train, Y_tilde), accuracy_score(Y_test, Y_predict)))

	f.close()

if(study == "grid search"):
# ---------------------------------------------- Ridge grid search ---------------------------------------------

	lambdas = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
	etas = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

	# Setting number of hidden layers and neurons in each layer
	num_hidd_layers = 3
	nodes_in_hidd_layers = [50,50,50]

	# Setting files
	filename = "Files/Class_Ridge_grid_search_3_layers.txt"
	f = open(filename, "w")
	f.write("lambda   eta   Acctrain  Acctest\n")
	f.close()

	f = open(filename, "a")

	for lamb in lambdas:
		for eta in etas:
			NN = NeuralNetwork(
				X_train, Y_train_onehot,
				n_hidden_layers = num_hidd_layers, n_hidden_neurons = nodes_in_hidd_layers,
				n_categories=n_categories, epochs=epochs,
				batch_size=batch_size, eta=eta,
				out_act_func = "softmax",hidden_act_func=activation_function,
				lmbd=lamb)

			NN.train()
			Y_predict = NN.predict_class(X_test)
			Y_tilde = NN.predict_class(X_train)

			f.write("{0} {1} {2} {3}\n".format(
				lamb, eta, accuracy_score(Y_train, Y_tilde), accuracy_score(Y_test, Y_predict)))

	f.close()
