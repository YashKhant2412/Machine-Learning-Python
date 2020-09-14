# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 09:38:29 2020

@author: Yash
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv('Mall_Customers.csv')

X = df.iloc[:,[3,4]].values

#Using Elbow method to optimal num of cluster
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow cluster')
plt.xlabel('Num of Cluster')
plt.ylabel('WCSS')
plt.show()

#Applying K-means to the dataset
kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(X)


#visulization
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s = 100, c= 'purple', label='c1' )
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s = 100, c= 'blue', label='c2' )
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s = 100, c= 'green', label='c3' )
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s = 100, c= 'purple', label='c4' )
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s = 100, c= 'cyan', label='c5' )
plt.scatter(kmeans.cluster_centers_[:,0],  kmeans.cluster_centers_[:,1], s=300, c='red' , label='center')
plt.title('K means Clustering')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.legend()
plt.savefig('abc.pdf')
plt.show()

