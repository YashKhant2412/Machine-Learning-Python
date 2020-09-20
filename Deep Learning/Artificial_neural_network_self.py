# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 15:58:27 2020

@author: Yash
"""

#Part 1 - Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Churn_Modelling.csv")
X= dataset.iloc[:,3:-1]
y= dataset.iloc[:,-1]

#Encoding data
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1,2])], remainder='passthrough')
X=np.array(columnTransformer.fit_transform(X))
labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)
X = X[:,[1,2,4,5,6,7,8,9,10,11,12]]

#Spliting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0, test_size = 0.20)

#Features Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Part2 Lets make ANN
#import Keras libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initilization of ANN
classifier = Sequential()
#Adding the input layer and first Hidden Layer
classifier.add(Dense(units = 6,activation='relu', input_dim=11, kernel_initializer='uniform'))

#adding second layer
classifier.add(Dense(units = 6,activation='relu', kernel_initializer='uniform'))

classifier.add(Dense(units = 6,activation='relu', kernel_initializer='uniform'))

#adding output layer
classifier.add(Dense(units = 1, activation='sigmoid', kernel_initializer='uniform'))

#Compile the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting ANN to Training set
classifier.fit(X_train,y_train, batch_size= 5, epochs=100)



#predict on test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

(1483+229)/2000