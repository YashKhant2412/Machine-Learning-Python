# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:15:45 2020

@author: Yash
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Social_Network_Ads.csv")
X= dataset.iloc[:,2:4].values
y= dataset.iloc[:,-1].values

#Spliting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0, test_size = 0.20)

#Features Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting Classifier
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0,C=10,gamma=0.5)
classifier.fit(X_train,y_train)

#predict on test set
y_pred = classifier.predict(X_test)

#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Applying k-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train,cv=10, n_jobs = -1)
accuracies.std()

#Applying Grid Search to find best parameter and best model
from sklearn.model_selection import GridSearchCV
para = [{'C':[1,10,100,1000],'kernel':['linear']},
        {'C':[1,10,100,1000],'kernel':['rbf'], 'gamma': [0.5,0.1,0.01,.001,0.0001]}]
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=para,
                           scoring='accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train,y_train)
best_acc = grid_search.best_score_
best_para = grid_search.best_params_

#Visulization on Training result
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01),
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))
plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('purple','blue')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c = ListedColormap(('orange','yellow'))(i), label = j)
plt.title("Kernel SVM (TRAIN SET)")
plt.xlabel("Age")
plt.ylabel('Salary')
plt.legend()
plt.show()

#visulizing test data

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.01),
                     np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.01))
plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('purple','blue')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c = ListedColormap(('orange','yellow'))(i), label = j)
plt.title("Kernel SVM (TEST SET)")
plt.xlabel("Age")
plt.ylabel('Salary')
plt.legend()
plt.savefig('abc.svg')
plt.show()