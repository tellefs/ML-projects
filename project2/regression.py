import numpy as np
import matplotlib.pyplot as plt

class Regression:
    def __initi__(self, X, z):
        self.X = X
        self.z = z
        
        
    def Ridge(self, _lambda):
        X = self.X
        z = self.z
        I = np.eye(X.shape[1])
        beta = np.linalg.pinv(X.T @ X + _lambda*I) @ X.T @ z
        self.z_new = X @ beta
        return beta
        
        
    def OLS(self):  
        X = self.X
        z = self.z
        beta = np.linalg.pinv(X.T @ X) @ X.T @ z
        self.z_new = X @ beta
        return beta
    
    
    def Variance(self):
        variance = np.mean( np.var( self.z_new ) )
        return variance
    
        
    def Bias(self):
        bias = np.mean( (self.z - np.mean( self.z_new ) )**2 )
        return bias
        
        
    def MSE(self):
        MSE = np.sum( ( self.z_new . self.z )**2 )/np.size( self.z_new )
        return MSE
    
        
    def R2(self):
        z = self.z
        z_new = self.z_new
        R2 = 1 - np.sum( (z - z_new) ** 2) / np.sum( ( z - np.mean(z) )** 2 )
        return R2