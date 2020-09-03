# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 18:56:52 2020

@author: Yash
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load data
dataset = pd.read_csv("Data.csv")

X= dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#fill missing data
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#encoding data
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
        #labelencoder_X = LabelEncoder()                  #for erlier version sklearn
        #X[:,0]=labelencoder_X.fit_transform(X[:,0])
        #onehotencoder = OneHotEncoder(categoriacal_features=[0])
        #X=onehotencoder.fit_transform(X).toarray()
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X=np.array(columnTransformer.fit_transform(X))
labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)

#splitting into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=42)

#Sacling the data....     Most of ML library no need to use Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
















