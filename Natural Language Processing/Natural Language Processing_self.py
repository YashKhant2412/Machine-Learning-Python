# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 21:18:16 2020

@author: Yash
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
corpus = []
#Data Cleaning
for i in range(len(data)):
    review = re.sub('[^a-zA-Z0-9]', ' ',data['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)

#Creation Of Bag of Words model
CV = CountVectorizer(max_features=750)
X = CV.fit_transform(corpus).toarray()
y = data.iloc[:,1].values

#classification
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0, test_size = 0.10)



#Fitting Classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#predict on test set
y_pred = classifier.predict(X_test)

#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)




