# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:04:45 2020

@author: Yash
"""


#SVR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load data
dataset = pd.read_csv("Position_Salaries.csv")

X= dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

#Features Scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
x = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(y)


#Regression
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,Y)


# Visulize SVR
plt.scatter(X,y,color = 'red')
plt.plot(X,sc_Y.inverse_transform(regressor.predict(x)),color='blue')
plt.title('Linear Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()




