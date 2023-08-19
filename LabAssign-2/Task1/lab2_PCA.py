#%%
# -*- coding: utf-8 -*-
"""
Created on Jan 3 20:11:55 2021

@author: Christian Wiskott

PCA from scratch.
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from numpy import linalg
from sklearn.decomposition import PCA

class PCA_custom:
    """
    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Training data.

    k : int, default=2 
        Number of selected features for analysis of the principal 
        components.
    
    Attributes
    ----------
    Z : array, shape (n_samples, n_components)
        Projected points onto principal components.

    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected components.

    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.
    """

    def __init__(self, X, k=2):
        self.X = X
        self.k = k
        self.Z = 0
        self.explained_variance_ratio_ = 0
        self.explained_variance_ = 0
        self.main_calculation()
        
    def main_calculation(self): 
                
        n = self.X.shape[0] # sample size
        m = self.X.shape[1] # number of features

        # Centering the data
        self.X -= np.mean(self.X, axis=0).T

        # Computing covariance matrix
        cov = (1/(n-1)) * self.X.T @ self.X

        # Compute ordered eigenvalues and eigenvectors
        evals, evectors = linalg.eig(cov)

        # Pick k<=m eigenvectors with largest eigenvalues
        W = evectors[:,:self.k]

        # Project the data. Equivalent to z_i = W.T @ x_i 
        self.Z = self.X @ W

        #compute percentage of explained variance
        self.explained_variance_ratio_ = evals[:self.k] / np.sum(np.diag(cov))
        self.explained_variance_ = evals[:self.k]

""" Comparing the custom and the sklearn PCA """

# Loading the data
X = genfromtxt("seeds.csv", skip_header=1, delimiter=',')

# Initializing the custom PCA class
custom = PCA_custom(X, k=2)

print("Custom PCA: Explained variance: ", 
      custom.explained_variance_)
print("Custom PCA: Explained variance ratio: "
      " {}\n".format(custom.explained_variance_ratio_))

# Initialiting and fitting the sklear PCA class
pca_sklearn = PCA(n_components=2) 
pca_sklearn.fit(X)

print("Sklearn PCA: Explained variance: ", 
      pca_sklearn.explained_variance_)
print("Sklearn PCA: Explained variance ratio: "
      "{}\n".format(pca_sklearn.explained_variance_ratio_))

""" Plotting the principal components """

sns.set_style('darkgrid')
sns.set_context('paper')

# Principal components
comp_custom = custom.Z
comp_sklean = pca_sklearn.fit_transform(X)

def color(X):
    # Manual colors
    cs = [('blue' if i == -0.9949748743718594 else 
           ('red' if i == 1.0050251256281406
             else 'green')) for i in X]
    return cs

plt.title('Principal Component Analysis', fontweight="bold", 
          fontsize=14)
plt.scatter(comp_custom[:,0], comp_custom[:,1], c=color(X[:,-1]), cmap=plt.cm.Set1, edgecolor='k',
            linewidths=1, marker='o', label="Custom PCA",
            alpha=0.5)
#plt.scatter(comp_sklean[:, 0], comp_sklean[:, 1], 
 #           c=color(X[:,-1]), label="Sklearn PCA", linewidths=1, marker='x',
  #          alpha=0.5)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
#plt.legend()
plt.savefig("PCA.png", dpi=1200)
plt.show()
# %%