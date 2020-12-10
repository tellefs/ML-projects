import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import seaborn as sns

''' This program contains all plotting functions'''

def surface_plot(x,y,z_original,z_predicted):
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    """
    # Plot the surface.
    surf = ax.plot_surface(x, y, z_original, cmap=cm.magma, linewidth=0, antialiased=False, vmin=0.0, vmax=15.0)
    #surf = ax.plot_surface(x, y, z_predicted, cmap=cm.Pastel1,linewidth=0, antialiased=False)
    ax.set_zlim(0.0, 9.00)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
     #fig.colorbar(surf, shrink=0.5, aspect=5)
    """
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter(np.ravel(x), np.ravel(y), np.ravel(z_original), c=np.ravel(z_original),cmap=cm.inferno,s=5, vmin=0.0, vmax=10.0)
    cbar=plt.colorbar(scat)
    cbar.set_label("Values (units)")
    plt.show()


def plot_heatmaps_lin_reg():
    file = np.loadtxt("../Files/Grid_search_Ridge_MSE.txt", skiprows=1)
    mse_train  = file[:,2]
    mse_test  = file[:,3]
    mse_train = mse_train[:,np.newaxis]
    mse_test = mse_test[:,np.newaxis]
    mse_train=np.reshape(mse_train,(6,8))
    mse_test=np.reshape(mse_test,(6,8))

    file = np.loadtxt("../Files/Grid_search_Ridge_R2.txt", skiprows=1)
    r2_train  = file[:,2]
    r2_test  = file[:,3]
    r2_train = r2_train[:,np.newaxis]
    r2_test = r2_test[:,np.newaxis]
    r2_train=np.reshape(r2_train,(6,8))
    r2_test=np.reshape(r2_test,(6,8))

    poly_deg_array = [14,15,16,17,18,19]
    lambda_array = [ 0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]

    x = lambda_array
    y = poly_deg_array
    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (7, 10))
    axs = axes.ravel()
    sns.set(font_scale=1)
    ax=sns.heatmap(r2_train,  xticklabels=x, yticklabels=y, annot=True, ax=axs[0], cmap="inferno",fmt=".3", vmin = 0.8, vmax = 1.0)
    axs[0].set_title("$R^2$ for the training set, Ridge", fontsize=14)
    axs[0].set_xlabel("$\lambda$", fontsize=14)
    axs[0].set_ylabel("Polynomial degree", fontsize=14)
    axs[0].tick_params(axis="x", labelsize=12)
    axs[0].tick_params(axis="y", labelsize=12,rotation=0)

    ax=sns.heatmap(r2_test,  xticklabels=x, yticklabels=y, annot=True, ax=axs[1],cmap="inferno", fmt=".3", vmin = 0.8, vmax = 1.0)
    axs[1].set_title("$R^2$ for the test set, Ridge", fontsize=14)
    axs[1].set_xlabel("$\lambda$", fontsize=14)
    axs[1].set_ylabel("Polynomial degree", fontsize=14)
    axs[1].tick_params(axis="x", labelsize=12)
    axs[1].tick_params(axis="y", labelsize=12,rotation=0)

    plt.subplots_adjust(hspace=0.25, top=0.96, bottom=0.08, right=0.99,left=0.14)
    plt.savefig('../Figures/Ridge_R2.pdf')
    plt.show()


    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (7, 10))
    axs = axes.ravel()
    sns.set(font_scale=1)
    ax=sns.heatmap(mse_train, xticklabels=x, yticklabels=y, annot=True, ax=axs[0], cmap="inferno", fmt=".1", vmin = 0.0, vmax = 0.08)
    axs[0].set_title("MSE for the training set, Ridge", fontsize=14)
    axs[0].set_xlabel("$\lambda$", fontsize=14)
    axs[0].set_ylabel("Polynomial degree", fontsize=14)
    axs[0].tick_params(axis="x", labelsize=12)
    axs[0].tick_params(axis="y", labelsize=12,rotation=0)

    ax=sns.heatmap(mse_test, xticklabels=x, yticklabels=y, annot=True, ax=axs[1], cmap="inferno", fmt=".2", vmin = 0.0, vmax = 0.08)
    axs[1].set_title("MSE for the test set, Ridge", fontsize=14)
    axs[1].set_xlabel("$\lambda$", fontsize=14)
    axs[1].set_ylabel("Polynomial degree", fontsize=14)
    axs[1].tick_params(axis="x", labelsize=12)
    axs[1].tick_params(axis="y", labelsize=12,rotation=0)

    plt.subplots_adjust(hspace=0.25, top=0.96, bottom=0.08, right=0.99,left=0.14)
    plt.savefig('../Figures/Ridge_MSE.pdf')
    plt.show()


def plot_heatmaps_NN():
    file = np.loadtxt("../Files/Grid_search_NN_tanh_MSE_tot.txt", skiprows=1)
    mse_train  = file[:,2]
    mse_test  = file[:,3]
    mse_train = mse_train[:,np.newaxis]
    mse_test = mse_test[:,np.newaxis]
    mse_train=np.reshape(mse_train,(7,8))
    mse_test=np.reshape(mse_test,(7,8))

    file = np.loadtxt("../Files/Grid_search_NN_tanh_R2_tot.txt", skiprows=1)
    r2_train  = file[:,2]
    r2_test  = file[:,3]
    r2_train = r2_train[:,np.newaxis]
    r2_test = r2_test[:,np.newaxis]
    r2_train=np.reshape(r2_train,(7,8))
    r2_test=np.reshape(r2_test,(7,8))

    eta_array = [0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0.00001, 0.000001]
    lambda_array = [ 0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]

    x = lambda_array
    y = eta_array

    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (7, 10))
    axs = axes.ravel()
    sns.set(font_scale=1)
    ax=sns.heatmap(r2_train,  xticklabels=x, yticklabels=y, annot=True, ax=axs[0], cmap="inferno",fmt=".3", vmin = 0.4, vmax = 1.0)
    axs[0].set_title("$R^2$ for the training set, FFNN, tanh", fontsize=14)
    axs[0].set_xlabel("$\lambda$", fontsize=14)
    axs[0].set_ylabel("$\eta$", fontsize=14)
    axs[0].tick_params(axis="x", labelsize=12)
    axs[0].tick_params(axis="y", labelsize=12,rotation=0)

    ax=sns.heatmap(r2_test,  xticklabels=x, yticklabels=y, annot=True, ax=axs[1],cmap="inferno", fmt=".3", vmin = 0.4, vmax = 1.0)
    axs[1].set_title("$R^2$ for the test set, FFNN, tanh", fontsize=14)
    axs[1].set_xlabel("$\lambda$", fontsize=14)
    axs[1].set_ylabel("$\eta$", fontsize=14)
    axs[1].tick_params(axis="x", labelsize=12)
    axs[1].tick_params(axis="y", labelsize=12,rotation=0)

    plt.subplots_adjust(hspace=0.25, top=0.96, bottom=0.08, right=0.99,left=0.14)
    plt.savefig('../Figures/NN_tanh_R2.pdf')
    plt.show()


    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (7, 10))
    axs = axes.ravel()
    sns.set(font_scale=1)
    ax=sns.heatmap(mse_train, xticklabels=x, yticklabels=y, annot=True, ax=axs[0], cmap="inferno", vmin = 0.0, vmax = 0.55)
    axs[0].set_title("MSE for the training set, FFNN, tanh", fontsize=14)
    axs[0].set_xlabel("$\lambda$", fontsize=14)
    axs[0].set_ylabel("$\eta$", fontsize=14)
    axs[0].tick_params(axis="x", labelsize=12)
    axs[0].tick_params(axis="y", labelsize=12,rotation=0)

    ax=sns.heatmap(mse_test, xticklabels=x, yticklabels=y, annot=True, ax=axs[1], cmap="inferno", vmin = 0.0, vmax = 0.55)
    axs[1].set_title("MSE for the test set, FFNN, tanh", fontsize=14)
    axs[1].set_xlabel("$\lambda$", fontsize=14)
    axs[1].set_ylabel("$\eta$", fontsize=14)
    axs[1].tick_params(axis="x", labelsize=12)
    axs[1].tick_params(axis="y", labelsize=12,rotation=0)

    plt.subplots_adjust(hspace=0.25, top=0.96, bottom=0.08, right=0.99,left=0.14)
    plt.savefig('../Figures/NN_tanh_MSE.pdf')

def gridsearch_decisiontree():
    file = np.loadtxt('../Files/DecisionTree_train_R2.txt',skiprows=1)

    r21  = file[:,2]
    r21 = r21[:,np.newaxis]
    r21=np.reshape(r21,(10,8))

    file = np.loadtxt("../Files/DecisionTree_test_R2.txt", skiprows=1)
    r22  = file[:,2]
    r22 = r22[:,np.newaxis]
    r22=np.reshape(r22,(10,8))

    lambdas = lambda_values = np.hstack((np.array([0.0]), np.logspace(-7,-1,7)))
    y = np.linspace(1,10,10)
    x = lambdas
    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (7, 13))
    axs = axes.ravel()
    sns.set(font_scale=1)
    ax=sns.heatmap(r21,  xticklabels=x, yticklabels=y, annot=True, ax=axs[0], cmap="inferno")
    axs[0].set_title("$R^2$ for the training set", fontsize=14)
    axs[0].set_xlabel(r"$\lambda$", fontsize=14)
    axs[0].set_ylabel("Depth", fontsize=14)
    axs[0].tick_params(axis="x", labelsize=12)
    axs[0].tick_params(axis="y", labelsize=12,rotation=0)

    ax=sns.heatmap(r22,  xticklabels=x, yticklabels=y, annot=True, ax=axs[1], cmap="inferno")
    axs[1].set_title("$R^2$ for the test set", fontsize=14)
    axs[1].set_xlabel(r"$\lambda$", fontsize=14)
    axs[1].set_ylabel("Depth", fontsize=14)
    axs[1].tick_params(axis="x", labelsize=12)
    axs[1].tick_params(axis="y", labelsize=12,rotation=0)
    plt.subplots_adjust(hspace=0.2, top=0.96, bottom=0.08, right=0.99,left=0.14)
    plt.savefig('../Figures/DecisionTree_R2.pdf')
    plt.show()

    file = np.loadtxt("../Files/DecisionTree_train_MSE.txt", skiprows=1)
    mse1  = file[:,2]
    mse1 = mse1[:,np.newaxis]
    mse1=np.reshape(mse1,(10,8))

    file = np.loadtxt("../Files/DecisionTree_test_MSE.txt", skiprows=1)
    mse2  = file[:,2]
    mse2 = mse2[:,np.newaxis]
    mse2=np.reshape(mse2,(10,8))

    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (7, 13))
    axs = axes.ravel()
    sns.set(font_scale=1)
    ax=sns.heatmap(mse1, vmin = 0, vmax =0.45, xticklabels=x, yticklabels=y, annot=True, ax=axs[0], cmap="inferno")
    axs[0].set_title("MSE for the training set", fontsize=14)
    axs[0].set_xlabel(r"$\lambda$", fontsize=14)
    axs[0].set_ylabel("Depth", fontsize=14)
    axs[0].tick_params(axis="x", labelsize=12)
    axs[0].tick_params(axis="y", labelsize=12,rotation=0)

    ax=sns.heatmap(mse2, vmin = 0, vmax =0.45, xticklabels=x, yticklabels=y, annot=True, ax=axs[1], cmap="inferno")
    axs[1].set_title("MSE for the test set", fontsize=14)
    axs[1].set_xlabel(r"$\lambda$", fontsize=14)
    axs[1].set_ylabel("Depth", fontsize=14)
    axs[1].tick_params(axis="x", labelsize=12)
    axs[1].tick_params(axis="y", labelsize=12,rotation=0)
    plt.subplots_adjust(hspace=0.2, top=0.96, bottom=0.08, right=0.99,left=0.14)
    plt.savefig('../Figures/DecisionTree_MSE.pdf')
    plt.show()

def gridsearch_XGB():
    file = np.loadtxt('../Files/XGB_train_R2.txt',skiprows=1)

    r21  = file[:,2]
    r21 = r21[:,np.newaxis]
    r21=np.reshape(r21,(7,8))

    file = np.loadtxt("../Files/XGB_test_R2.txt", skiprows=1)
    r22  = file[:,2]
    r22 = r22[:,np.newaxis]
    r22=np.reshape(r22,(7,8))

    lambdas = lambda_values = np.hstack((np.array([0.0]), np.logspace(-4,2,7)))
    y = np.linspace(1,7,7)
    x = lambdas
    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (7, 13))
    axs = axes.ravel()
    sns.set(font_scale=1)
    ax=sns.heatmap(r21, xticklabels=x, yticklabels=y, annot=True, ax=axs[0], cmap="inferno", fmt='.2g')
    axs[0].set_title("$R^2$ for the training set", fontsize=14)
    axs[0].set_xlabel(r"$\lambda$", fontsize=14)
    axs[0].set_ylabel("Depth", fontsize=14)
    axs[0].tick_params(axis="x", labelsize=12)
    axs[0].tick_params(axis="y", labelsize=12,rotation=0)

    ax=sns.heatmap(r22, vmin=0.69, vmax=1, xticklabels=x, yticklabels=y, annot=True, ax=axs[1], cmap="inferno", fmt='.2g')
    axs[1].set_title("$R^2$ for the test set", fontsize=14)
    axs[1].set_xlabel(r"$\lambda$", fontsize=14)
    axs[1].set_ylabel("Depth", fontsize=14)
    axs[1].tick_params(axis="x", labelsize=12)
    axs[1].tick_params(axis="y", labelsize=12,rotation=0)
    plt.subplots_adjust(hspace=0.2, top=0.96, bottom=0.08, right=0.99,left=0.14)
    plt.savefig('../Figures/XGB_R2.pdf')
    plt.show()

    file = np.loadtxt("../Files/XGB_train_MSE.txt", skiprows=1)
    mse1  = file[:,2]
    mse1 = mse1[:,np.newaxis]
    mse1=np.reshape(mse1,(8,7))

    file = np.loadtxt("../Files/XGB_test_MSE.txt", skiprows=1)
    mse2  = file[:,2]
    mse2 = mse2[:,np.newaxis]
    mse2=np.reshape(mse2,(8,7))

    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (7, 13))
    axs = axes.ravel()
    sns.set(font_scale=1)
    ax=sns.heatmap(mse1, vmin = 0, vmax =0.1, xticklabels=x, yticklabels=y, annot=True, ax=axs[0], cmap="inferno", fmt='.2g')
    axs[0].set_title("MSE for the training set", fontsize=14)
    axs[0].set_xlabel(r"$\lambda$", fontsize=14)
    axs[0].set_ylabel("Depth", fontsize=14)
    axs[0].tick_params(axis="x", labelsize=12)
    axs[0].tick_params(axis="y", labelsize=12,rotation=0)

    ax=sns.heatmap(mse2, vmin = 0, vmax =0.3, xticklabels=x, yticklabels=y, annot=True, ax=axs[1], cmap="inferno", fmt='.2g')
    axs[1].set_title("MSE for the test set", fontsize=14)
    axs[1].set_xlabel(r"$\lambda$", fontsize=14)
    axs[1].set_ylabel("Depth", fontsize=14)
    axs[1].tick_params(axis="x", labelsize=12)
    axs[1].tick_params(axis="y", labelsize=12,rotation=0)
    plt.subplots_adjust(hspace=0.2, top=0.96, bottom=0.08, right=0.99,left=0.14)
    plt.savefig('../Figures/XGB_MSE.pdf')
    plt.show()


# Uncomment the following lines to get required plot:
#plot_heatmaps_lin_reg()
#plot_heatmaps_NN()
#gridsearch_decisiontree()
#gridsearch_XGB()
