import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import datasets

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function

from sklearn.model_selection import train_test_split


def create_neural_network_Keras(n_neurons_layer1, n_neurons_layer2, n_neurons_layer3,
                                n_categories, eta ,lmbd, activation,
                                activation_layers):

    model = Sequential()
    model.add(Dense(n_neurons_layer1, activation=activation_layers,
                    kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(n_neurons_layer2, activation=activation_layers,
                    kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(n_neurons_layer3, activation=activation_layers,
                    kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(n_categories, activation=activation))

    sgd = optimizers.SGD(lr=eta)
    #adam = optimizers.Adam(lr=eta)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                    metrics=['accuracy'])

    return model
