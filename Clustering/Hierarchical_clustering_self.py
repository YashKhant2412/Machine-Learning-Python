# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:37:59 2020

@author: Yash
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Mall_Customers.csv')

X = df.iloc[:,[3,4]].values

#Using Dendogram to find the optimal num of cluster
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method='ward',))
plt.title('Dendogram')
plt.xlabel('Customer')
plt.ylabel("Ecluid distance")
plt.savefig('dendogram.pdf')
plt.show()

#Fitting HC to data
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc =hc.fit_predict(X)

#visulization
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s = 100, c= 'purple', label='c1' )
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s = 100, c= 'blue', label='c2' )
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s = 100, c= 'green', label='c3' )
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s = 100, c= 'purple', label='c4' )
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s = 100, c= 'cyan', label='c5' )
plt.title('Hierarchical Clustering')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.legend()
plt.savefig('abc.pdf')
plt.show()




