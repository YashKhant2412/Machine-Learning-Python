# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:06:08 2020

@author: Yash
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load data
dataset = pd.read_csv("Salary_Data.csv")

X= dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 ,random_state=41)

#fit model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#plot training data
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train,regressor.predict(X_train), color="blue")
plt.title("Salary Vs. Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.plot()

#plot testing data
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train,regressor.predict(X_train), color="blue")
plt.title("Salary Vs. Experience on Test")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.plot()

