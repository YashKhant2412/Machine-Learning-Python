# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 22:52:24 2020

@author: Yash
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load data
dataset = pd.read_csv("Position_Salaries.csv")

X= dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#linear regresson
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)


#polynomial regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly,y)

# Visulize Linear regression
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Linear Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visulize Polynomial Regression
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg_poly.predict(X_poly),color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()



