# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 10:57:54 2021

@author: emreb
"""

import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np


#%% Data Preparation

data = pd.read_csv("heart.csv")
y = data.target.values

x = data.drop(["target"],axis = 1)
x = (x - np.min(x)) / (np.max(x) - np.min(x))

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2 , random_state = 1)

accuracy = list()

#%% Decision Tree Classification

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)
accuracy.append(dt.score(x_test,y_test))
print("print accuracy of decision tree algo:" , dt.score(x_test,y_test))


#%% Naive Bayes Classification

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)
accuracy.append(nb.score(x_test,y_test))
print("print accuracy of naive bayes algo:" , nb.score(x_test,y_test))

#%% SVM(Support Vector Machine) Classification
from sklearn.svm import SVC

svm = SVC(random_state = 1)
svm.fit(x_train,y_train)

print("print accuracy of svm algo:" , svm.score(x_test,y_test))
accuracy.append(svm.score(x_test,y_test))

#%%  Random Forest Classification
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100,random_state = 1)
rf.fit(x_train,y_train)
print("random forest algo result: ",rf.score(x_test,y_test))
accuracy.append(rf.score(x_test,y_test))

#Confusion_Matrix
y_pred = rf.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)


print("Max accuracy:",max(accuracy))





