# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 09:28:16 2020

@author: Yash
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load data
dataset = pd.read_csv("50_Startups.csv")

X= dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X=np.array(columnTransformer.fit_transform(X))

#avoiding the dummy variable trap
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 ,random_state=26)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

import statsmodels.api as sm
X = np.append(np.ones((50,1)).astype(int), X , axis =1)
X_opt = np.array(X[:,[0,1,2,3,4,5]], dtype = float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = np.array(X[:,[0,1,3,4,5]], dtype = float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = np.array(X[:,[0,3,4,5]], dtype = float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = np.array(X[:,[0,3,5]], dtype = float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()



