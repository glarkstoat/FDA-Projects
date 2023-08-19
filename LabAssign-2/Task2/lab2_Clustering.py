#%%
# -*- coding: utf-8 -*-

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from numpy import linalg
from sklearn import cluster 
from sklearn import mixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

files = ["dataset1_noCluster7.csv", "dataset2_noCluster6.csv", "dataset3_noCluster2.csv",
         "dataset4_noCluster2.csv", "dataset5_noCluster2.csv"]

scores_DBSCAN, scores_Km, scores_Gauss, scores_Agg = [], [], [], []

for file in files:

      # Loading and preprocessing the data
      data = genfromtxt(file, delimiter=",", skip_header=1)
      training_set = data[:,:2]; targets = data[:,2]
      training_set = StandardScaler().fit_transform(training_set)

      # Plot of the original data
      plt.figure()
      plt.title('{}'.format(file), fontweight="bold", 
            fontsize=14)
      plt.scatter(training_set[:,0], training_set[:,1], c=targets,
                  s=50, cmap=plt.cm.Set1, edgecolor='k', alpha=0.8)
      plt.tight_layout()
      plt.savefig("{}.png".format(file), dpi=1200)
      plt.show()

      # 2x2 Subplot containing predictions by the four methods
      fig = plt.figure()

      # DBSCAN
      plt.subplot(2, 2, 1)
      plt.title('DBSCAN', fontweight="bold", 
            fontsize=10)

      clustering = cluster.DBSCAN(eps=0.3, min_samples=30).fit(training_set)
      labels = clustering.labels_
      scores_DBSCAN.append(normalized_mutual_info_score(targets, labels))
      scores_DBSCAN.append(adjusted_rand_score(targets, labels))

      plt.scatter(training_set[:,0], training_set[:,1], 
                  c=labels, label="Predicted", s=9, alpha=0.9,
                   linewidths=1, cmap=plt.cm.Set1, edgecolor='k')
      plt.xticks(fontsize=5); plt.yticks(fontsize=5)
      plt.legend(prop={'size': 6})
      
      # KMeans
      plt.subplot(2, 2, 2)
      plt.title('KMeans', fontweight="bold", 
            fontsize=10)

      clustering = cluster.KMeans(n_clusters=int(file[-5])).fit(training_set)
      labels = clustering.labels_
      scores_Km.append(normalized_mutual_info_score(targets, labels))
      scores_Km.append(adjusted_rand_score(targets, labels))

      plt.scatter(training_set[:,0], training_set[:,1], 
                  c=labels, label="Predicted", s=9, alpha=0.9,
                   cmap=plt.cm.Set1, edgecolor='k')
      plt.xticks(fontsize=5);     plt.yticks(fontsize=5)
      plt.legend(prop={'size': 6})

      # GaussianMixture
      plt.subplot(2, 2, 3)
      plt.title('GaussianMixture', fontweight="bold", 
            fontsize=10)

      clustering = mixture.GaussianMixture(n_components=int(file[-5]))
      labels = clustering.fit_predict(training_set)
      scores_Gauss.append(normalized_mutual_info_score(targets, labels))
      scores_Gauss.append(adjusted_rand_score(targets, labels))

      plt.scatter(training_set[:,0], training_set[:,1], 
                  c=labels, label="Predicted", s=9, alpha=0.9,
                   cmap=plt.cm.Set1, edgecolor='k')
      plt.xticks(fontsize=5);     plt.yticks(fontsize=5)
      plt.legend(prop={'size': 6})

      # AgglomerativeClustering      
      plt.subplot(2, 2, 4)
      plt.title('AgglomerativeClustering', fontweight="bold", 
            fontsize=10)

      clustering = cluster.AgglomerativeClustering(n_clusters=int(file[-5])).fit(training_set)
      labels = clustering.labels_
      scores_Agg.append(normalized_mutual_info_score(targets, labels))
      scores_Agg.append(adjusted_rand_score(targets, labels))

      plt.scatter(training_set[:,0], training_set[:,1], cmap=plt.cm.Set1, edgecolor='k',
                  c=labels, label="Predicted", s=9, alpha=0.9)
      plt.xticks(fontsize=5);     plt.yticks(fontsize=5)
      plt.legend(prop={'size': 6})

      fig.tight_layout()
      fig.savefig("{}_methods.png".format(file), dpi=1200)

      plt.show()

# %%

file = "dataset4_noCluster2.csv"

# Loading and preprocessing the data
data = genfromtxt(file, delimiter=",", skip_header=1)
training_set = data[:,:2]; targets = data[:,2]
training_set = StandardScaler().fit_transform(training_set)

# KMeans
plt.title('KMeans', fontweight="bold", 
      fontsize=10)

clustering = cluster.KMeans(n_clusters=int(file[-5])).fit(training_set)
labels = clustering.labels_
scores_Km.append(normalized_mutual_info_score(targets, labels))
scores_Km.append(adjusted_rand_score(targets, labels))

plt.scatter(training_set[:,0], training_set[:,1], 
            c=labels, label="Predicted", s=9, alpha=0.7,
                  cmap=plt.cm.Set1, edgecolor='k')
plt.xticks(fontsize=5);     plt.yticks(fontsize=5)
plt.legend(prop={'size': 6})
# %%

targets

# %%
