import numpy as np
from activation_functions import *

class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_layers = 1,
            n_hidden_neurons=[50],
            n_categories=1,
            epochs=1000,
            batch_size=5,
            eta=0.001,
            HiddenActFunc = "sigmoid",
            OutActFunc= "linear",
            lmbd=0.0):
        # Leaky ReLU, tanh and ReLU eta=0.00001, epochs=10000 (100000 for better results), sigmoid - 0.001, 1000
        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories
        self.hidden_layers = []
        self.HiddenActFunc = HiddenActFunc
        self.OutActFunc = OutActFunc

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        for i in range(self.n_hidden_layers):
            if(i==0):
                layer = HiddenLayer(self.n_hidden_neurons[i], self.n_features, self.HiddenActFunc)
            else:
                layer = HiddenLayer(self.n_hidden_neurons[i], self.n_hidden_neurons[i-1], self.HiddenActFunc)
            layer.create_biases_and_weights()
            self.hidden_layers.append(layer)

        self.create_output_biases_and_weights()


    def create_output_biases_and_weights(self):
        
        self.output_weights = np.random.randn(self.hidden_layers[self.n_hidden_layers-1].n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01


    def feed_forward(self):
        # feed-forward for training
        # Running over all hidden layers
        for i, layer in enumerate(self.hidden_layers):

            if(i == 0):
                a_in = self.X_data
            else:
                a_in = self.hidden_layers[i-1].a_h

            layer.z_h = np.matmul(a_in, layer.hidden_weights) + layer.hidden_bias
            layer.a_h = layer.HiddActFunction(layer.z_h)

        # Output layer
        self.z_o = np.matmul(self.hidden_layers[self.n_hidden_layers-1].a_h, self.output_weights) + self.output_bias
        self.a_o = ActivationFunction(self.z_o, self.OutActFunc)


    def backpropagation(self):

        # This line will be different for classification
        if(self.OutActFunc=="linear"):
            error_output = self.a_o - self.Y_data[:,np.newaxis]
        else:
            error_output = self.a_o - self.Y_data
            
        self.error_output = error_output

        # Calculate gradients for the output layer
        self.output_weights_gradient = np.matmul(self.hidden_layers[self.n_hidden_layers-1].a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights

        # Update weights and biases in the output layer
        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient

        for i, layer in reversed(list(enumerate(self.hidden_layers))):
            if(i == (self.n_hidden_layers-1)):
                forward_error = error_output
                forward_weights = self.output_weights
                
            else:
                forward_error = self.hidden_layers[i+1].error
                forward_weights = self.hidden_layers[i+1].hidden_weights

            layer.error = np.matmul(forward_error, forward_weights.T) * layer.HiddActFunctionDeriv(layer.z_h)

            if(i == 0):
                backward_a = self.X_data
            else:
                backward_a = self.hidden_layers[i-1].a_h

            layer.hidden_weights_gradient = np.matmul(backward_a.T, layer.error)
            layer.hidden_bias_gradient = np.sum(layer.error, axis=0)

            if self.lmbd > 0.0:
                layer.hidden_weights_gradient += self.lmbd * layer.hidden_weights

            layer.hidden_weights -= self.eta * layer.hidden_weights_gradient
            layer.hidden_bias -= self.eta * layer.hidden_bias_gradient




    def train(self):

        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints without replacement
                chosen_datapoints = np.random.choice(data_indices, size=self.batch_size, replace=False)

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()
                

    def predict(self, X):
        
        self.X_data = X
        self.feed_forward()
        return self.a_o

    def predict_class(self, X):

        self.X_data = X
        self.feed_forward()
        return np.argmax(self.a_o, axis=1)



class HiddenLayer:
    def __init__(self, n_neurons, n_features, activation_function):
        self.n_hidden_neurons = n_neurons
        self.n_features = n_features
        self.activation_function = activation_function

    def HiddActFunction(self, x):
        return(ActivationFunction(x, self.activation_function))

    def HiddActFunctionDeriv(self, x):
        return(ActivationFunctionDeriv(x, self.activation_function))

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

