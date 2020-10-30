# This program contains all plotting functions

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from data_processing import *
import seaborn as sns

from statistical_functions import R2, MSE, ols_svd, BetaVar

def SurfacePlot(x,y,z_original,z_predicted):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x, y, z_original, cmap=cm.ocean,linewidth=0, antialiased=False)
    surf = ax.plot_surface(x, y, z_predicted, cmap=cm.Pastel1,linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()



def PrintErrors(z_train,z_tilde,z_test,z_predict):
	print("Training R2")
	print(R2(z_train,z_tilde))
	print("Training MSE")
	print(MSE(z_train,z_tilde))
	print("Test R2")
	print(R2(z_test,z_predict))
	print("Test MSE")
	print(MSE(z_test,z_predict))

def Plot_R2():

    file = np.loadtxt("Files/OLS_SGD_R2.txt", skiprows=1)
    eta  = file[:,0]
    trainR2 = file[:,1]
    testR2 = file[:,2]

    plt.plot(eta, trainR2, label="Train",color="navy")
    plt.plot(eta, testR2, label="Test",color="orangered")

    plt.xlabel("$\eta$",fontsize=12)
    plt.ylabel("$R^2$",fontsize=12)
    plt.title("OLS, $R^2$ for the test and train datasets vs $\eta$", fontsize=12)
    plt.grid(True)
    plt.xscale("log")
    plt.legend(fontsize=12)
    plt.show()

def Plot_MSE():

    file = np.loadtxt("Files/OLS_SGD_MSE.txt", skiprows=1)
    eta  = file[:,0]
    trainMSE = file[:,1]
    testMSE = file[:,2]

    plt.plot(eta, trainMSE, label="Train",color="navy")
    plt.plot(eta, testMSE, label="Test",color="orangered")

    plt.xlabel("$\eta$",fontsize=12)
    plt.ylabel("MSE",fontsize=12)
    plt.title("OLS, MSE for the test and train datasets vs $\eta$", fontsize=12)
    plt.grid(True)
    plt.xscale("log")
    plt.legend(fontsize=12)
    plt.show()

def Plot_R2_minibatch():

    file = np.loadtxt("Files/OLS_SGD_R2_minibatch.txt", skiprows=1)
    mini  = file[:,0]
    trainR2 = file[:,1]
    testR2 = file[:,2]

    plt.plot(mini, trainR2, label="Train",color="navy")
    plt.plot(mini, testR2, label="Test",color="orangered")

    plt.xlabel("Minibatch size",fontsize=12)
    plt.ylabel("$R^2$",fontsize=12)
    plt.title("OLS, $R^2$ for the test and train datasets vs minibatch size", fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

def Plot_MSE_minibatch():

    file = np.loadtxt("Files/OLS_SGD_MSE_minibatch.txt", skiprows=1)
    mini  = file[:,0]
    trainMSE = file[:,1]
    testMSE = file[:,2]

    plt.plot(mini, trainMSE, label="Train",color="navy")
    plt.plot(mini, testMSE, label="Test",color="orangered")

    plt.xlabel("Minibatch size",fontsize=12)
    plt.ylabel("MSE",fontsize=12)
    plt.title("OLS, MSE for the test and train datasets vs minibatch size", fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

def Plot_MSE_epochs():

    file = np.loadtxt("Files/OLS_SGD_MSE_epoch.txt", skiprows=1)
    epochs  = file[:,0]
    trainMSE = file[:,1]
    testMSE = file[:,2]

    plt.plot(epochs, trainMSE, label="Train",color="navy")
    plt.plot(epochs, testMSE, label="Test",color="orangered")

    plt.xlabel("Number of epochs",fontsize=12)
    plt.ylabel("MSE",fontsize=12)
    plt.title("OLS, MSE for the test and train datasets vs number of epochs", fontsize=12)
    plt.grid(True)
    plt.xscale("log")
    plt.legend(fontsize=12)
    plt.show()

def Plot_R2_epochs():

    file = np.loadtxt("Files/OLS_SGD_R2_epoch.txt", skiprows=1)
    epochs  = file[:,0]
    trainR2 = file[:,1]
    testR2 = file[:,2]

    plt.plot(epochs, trainR2, label="Train",color="navy")
    plt.plot(epochs, testR2, label="Test",color="orangered")

    plt.xlabel("Number of epochs",fontsize=12)
    plt.ylabel("$R^2$",fontsize=12)
    plt.title("OLS, $R^2$ for the test and train datasets vs number of epochs", fontsize=12)
    plt.grid(True)
    plt.xscale("log")
    plt.legend(fontsize=12)
    plt.show()

def Plot_heatmaps_coarse():
    file = np.loadtxt("Files/Ridge_train_R2.txt", skiprows=1)
    r21  = file[:,2]
    r21 = r21[:,np.newaxis]
    r21=np.reshape(r21,(8,5))

    file = np.loadtxt("Files/Ridge_test_R2.txt", skiprows=1)
    r22  = file[:,2]
    r22 = r22[:,np.newaxis]
    r22=np.reshape(r22,(8,5))

    lambdas = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    t1_array = [100, 1000, 10000, 100000, 1000000]
    for i in range(len(t1_array)):
        t1_array[i]=1.0/t1_array[i]
    x = np.log10(t1_array)
    y = lambdas
    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (6.5, 11))
    axs = axes.ravel()
    ax=sns.heatmap(r21,  xticklabels=x, yticklabels=y, annot=True, ax=axs[0], cmap="viridis")
    axs[0].set_title("Ridge, $R^2$ for the training set")
    axs[0].set_xlabel("$\log_{10}\eta$")
    axs[0].set_ylabel("$\lambda$")

    ax=sns.heatmap(r22,  xticklabels=x, yticklabels=y, annot=True, ax=axs[1], cmap="viridis")
    axs[1].set_title("Ridge, $R^2$ for the test set")
    axs[1].set_xlabel("$\log_{10}\eta$")
    axs[1].set_ylabel("$\lambda$")    
    plt.savefig('Figures/Ridge_R2.pdf')
    plt.show()

    file = np.loadtxt("Files/Ridge_train_MSE.txt", skiprows=1)
    mse1  = file[:,2]
    mse1 = mse1[:,np.newaxis]
    mse1=np.reshape(mse1,(8,5))

    file = np.loadtxt("Files/Ridge_test_MSE.txt", skiprows=1)
    mse2  = file[:,2]
    mse2 = mse2[:,np.newaxis]
    mse2=np.reshape(mse2,(8,5))

    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (6.5, 11))
    axs = axes.ravel()
    ax=sns.heatmap(mse1, vmin = 0, vmax =2.5, xticklabels=x, yticklabels=y, annot=True, ax=axs[0], cmap="viridis")
    axs[0].set_title("Ridge, MSE for the training set")
    axs[0].set_xlabel("$\log_{10}\eta$")
    axs[0].set_ylabel("$\lambda$")

    ax=sns.heatmap(mse2, vmin = 0, vmax =2.5, xticklabels=x, yticklabels=y, annot=True, ax=axs[1], cmap="viridis")
    axs[1].set_title("Ridge, MSE for the test set")
    axs[1].set_xlabel("$\log_{10}\eta$")
    axs[1].set_ylabel("$\lambda$")    
    plt.savefig('Figures/Ridge_MSE.pdf')
    plt.show()

def Plot_heatmaps_fine():
    file = np.loadtxt("Files/Ridge_train_R2_fine.txt", skiprows=1)
    r21  = file[:,2]
    r21 = r21[:,np.newaxis]
    r21=np.reshape(r21,(8,6))

    file = np.loadtxt("Files/Ridge_test_R2_fine.txt", skiprows=1)
    r22  = file[:,2]
    r22 = r22[:,np.newaxis]
    r22=np.reshape(r22,(8,6))

    lambdas = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    t1_array = [1/0.001,1/0.0005,1/0.0001]
    for i in range(len(t1_array)):
        t1_array[i]=1.0/t1_array[i]
    x = [0.005,0.004,0.003,0.002,0.001, 0.0001]
    y = lambdas
    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (6.5, 11))
    axs = axes.ravel()
    ax=sns.heatmap(r21,  xticklabels=x, yticklabels=y, annot=True, ax=axs[0], cmap="viridis")
    axs[0].set_title("Ridge, $R^2$ for the training set")
    axs[0].set_xlabel("$\log_{10}\eta$")
    axs[0].set_ylabel("$\lambda$")

    ax=sns.heatmap(r22,  xticklabels=x, yticklabels=y, annot=True, ax=axs[1], cmap="viridis")
    axs[1].set_title("Ridge, $R^2$ for the test set")
    axs[1].set_xlabel("$\log_{10}\eta$")
    axs[1].set_ylabel("$\lambda$")    
    plt.savefig('Figures/Ridge_R2_fine.pdf')
    plt.show()

    file = np.loadtxt("Files/Ridge_train_MSE_fine.txt", skiprows=1)
    mse1  = file[:,2]
    mse1 = mse1[:,np.newaxis]
    mse1=np.reshape(mse1,(8,6))

    file = np.loadtxt("Files/Ridge_test_MSE_fine.txt", skiprows=1)
    mse2  = file[:,2]
    mse2 = mse2[:,np.newaxis]
    mse2=np.reshape(mse2,(8,6))

    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (6.5, 11))
    axs = axes.ravel()
    ax=sns.heatmap(mse1, vmin = 0, vmax =0.45, xticklabels=x, yticklabels=y, annot=True, ax=axs[0], cmap="viridis")
    axs[0].set_title("Ridge, MSE for the training set")
    axs[0].set_xlabel("$\log_{10}\eta$")
    axs[0].set_ylabel("$\lambda$")

    ax=sns.heatmap(mse2, vmin = 0, vmax =0.45, xticklabels=x, yticklabels=y, annot=True, ax=axs[1], cmap="viridis")
    axs[1].set_title("Ridge, MSE for the test set")
    axs[1].set_xlabel("$\log_{10}\eta$")
    axs[1].set_ylabel("$\lambda$")    
    plt.savefig('Figures/Ridge_MSE_fine.pdf')
    plt.show()

def NN_epochs():

    file = np.loadtxt("Files/NN_MSE_epochs.txt", skiprows=1)
    epochs  = file[:,0]   
    mse_train  = file[:,1]  
    mse_test  = file[:,2]  

    file = np.loadtxt("Files/NN_R2_epochs.txt", skiprows=1) 
    r2_train  = file[:,1]  
    r2_test  = file[:,2]  

    plt.xlabel("Number of epochs",fontsize=12)
    plt.ylabel("MSE",fontsize=12)
    plt.title("MSE for the test and train datasets vs number of epochs", fontsize=12)
    plt.plot(epochs, mse_train, label="Train",color="navy")
    plt.plot(epochs, mse_test, label="Test",color="orangered")
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

    plt.xlabel("Number of epochs",fontsize=12)
    plt.ylabel("$R^2$",fontsize=12)
    plt.title("$R^2$ for the test and train datasets vs number of epochs", fontsize=12)
    plt.plot(epochs, r2_train, label="Train",color="navy")
    plt.plot(epochs, r2_test, label="Test",color="orangered")
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

def NN_layers():

    file = np.loadtxt("Files/NN_MSE_layers.txt", skiprows=1)
    epochs  = file[:,0]   
    mse_train  = file[:,1]  
    mse_test  = file[:,2]  

    file = np.loadtxt("Files/NN_R2_layers.txt", skiprows=1) 
    r2_train  = file[:,1]  
    r2_test  = file[:,2]  

    plt.xlabel("Number of hidden layers",fontsize=12)
    plt.ylabel("MSE",fontsize=12)
    plt.title("MSE for the test and train datasets vs number of hidden layers", fontsize=12)
    plt.plot(epochs, mse_train, label="Train",color="navy")
    plt.plot(epochs, mse_test, label="Test",color="orangered")
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

    plt.xlabel("Number of hidden layers",fontsize=12)
    plt.ylabel("$R^2$",fontsize=12)
    plt.title("$R^2$ for the test and train datasets vs number of hidden layers", fontsize=12)
    plt.plot(epochs, r2_train, label="Train",color="navy")
    plt.plot(epochs, r2_test, label="Test",color="orangered")
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

def Plot_heatmaps_NN():
    file = np.loadtxt("Files/NN_Ridge_MSE.txt", skiprows=1)
    mse_train  = file[:,2]
    mse_test  = file[:,3]
    mse_train = mse_train[:,np.newaxis]
    mse_test = mse_test[:,np.newaxis]
    mse_train=np.reshape(mse_train,(8,6))
    mse_test=np.reshape(mse_test,(8,6))

    file = np.loadtxt("Files/NN_Ridge_R2.txt", skiprows=1)
    r2_train  = file[:,2]
    r2_test  = file[:,3]
    r2_train = r2_train[:,np.newaxis]
    r2_train=np.reshape(r2_train,(8,6))
    r2_test = r2_test[:,np.newaxis]
    r2_test=np.reshape(r2_test,(8,6))

    lambdas = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    etas = [0.1, 0.01, 0.001,0.0001,0.00001,0.000001]

    x = np.log10(etas)
    y = lambdas
    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (6.5, 11))
    axs = axes.ravel()
    ax=sns.heatmap(r2_train,  xticklabels=x, yticklabels=y, annot=True, ax=axs[0], cmap="viridis",fmt=".3")
    axs[0].set_title("Ridge, $R^2$ for the training set")
    axs[0].set_xlabel("$\log_{10}\eta$")
    axs[0].set_ylabel("$\lambda$")

    ax=sns.heatmap(r2_test,  xticklabels=x, yticklabels=y, annot=True, ax=axs[1], cmap="viridis",fmt=".3")
    axs[1].set_title("Ridge, $R^2$ for the test set")
    axs[1].set_xlabel("$\log_{10}\eta$")
    axs[1].set_ylabel("$\lambda$")    
    plt.savefig('Figures/NN_Ridge_R2.pdf')
    plt.show()


    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (6.5, 11))
    axs = axes.ravel()
    ax=sns.heatmap(mse_train, vmin = 0, vmax =2.5, xticklabels=x, yticklabels=y, annot=True, ax=axs[0], cmap="viridis")
    axs[0].set_title("Ridge, MSE for the training set")
    axs[0].set_xlabel("$\log_{10}\eta$")
    axs[0].set_ylabel("$\lambda$")

    ax=sns.heatmap(mse_test, vmin = 0, vmax =2.5, xticklabels=x, yticklabels=y, annot=True, ax=axs[1], cmap="viridis")
    axs[1].set_title("Ridge, MSE for the test set")
    axs[1].set_xlabel("$\log_{10}\eta$")
    axs[1].set_ylabel("$\lambda$")    
    plt.savefig('Figures/NN_Ridge_MSE.pdf')
    plt.show()

#Plot_R2()
#Plot_MSE()
#Plot_R2_minibatch()
#Plot_MSE_minibatch()
#Plot_R2_epochs()
#Plot_MSE_epochs()
#Plot_heatmaps_coarse()
#Plot_heatmaps_fine()
#NN_epochs()
#NN_layers()
#Plot_heatmaps_NN()