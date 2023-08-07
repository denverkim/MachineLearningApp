# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 02:05:02 2023

@author: USER
"""

# Read original dataset
from sklearn import datasets
import pandas as pd
import os
os.chdir('D:/2205 ALGOLINK/2307 한동대/LAB/')

iris = datasets.load_iris()

iris_df = pd.DataFrame(iris.data, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
iris_df['Species'] = iris.target

# selecting features and target data
x = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df[['Species']]

# split data into train and test sets
# 70% training and 30% test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) #stratify=y

# create an instance of the random forest classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)

# train the classifier on the training data
rf.fit(x_train, y_train)

# predict on the test set
y_pred = rf.predict(x_test)

# calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}") # Accuracy: 0.91

import joblib
joblib.dump(rf, 'rf_model.sav')