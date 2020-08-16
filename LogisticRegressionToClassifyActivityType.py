# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 21:11:22 2020

@author: Niranch
"""

#==============================================================================
# Code to predict classification type using "Logistic Regression" for
# multiclass problem. The input data contains acceleration data from 
# smartphone on three axis. A second file provided contains labels to classify the 
# given acceleartion to it's activity type. The code uses this data to train 
# the prediction model to generate prediction of activity type for new 
# acceleration input
#==============================================================================

# Load libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load training data
train_data = pd.read_csv("\\HarvardPythonFinalProject\\train_time_series.csv")
train_labels = pd.read_csv("\\HarvardPythonFinalProject\\train_labels.csv")
#==============================================================================
# print(train_data.shape)
# print(train_labels.shape)
#==============================================================================
# Input data (Acceleration & Label was given in two different files and had to be joined)
data_to_train = pd.merge(train_data,train_labels,how = "inner", on = "timestamp")
#==============================================================================
# print(data_to_train.shape)
#==============================================================================
#Preparing my predictors and training data to generate the model
X=data_to_train[["x","y","z"]]
y=data_to_train[["label"]]
print(y.shape)
# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
# Create one-vs-rest logistic regression object 
clf = LogisticRegression(random_state=0, multi_class='ovr')

# Train model
model = clf.fit(X_std, y["label"])

# Create test array to predict thier class types
test_data =  pd.read_csv("\\HarvardPythonFinalProject\\test_time_series.csv")
#Use the model trained above to predict the given test data
y_hats = model.predict(test_data[["x","y","z"]])
test_data["label"]=y_hats
#output the test data with prediction results 
test_data.to_csv("\\HarvardPythonFinalProject\\test_time_series_output.csv")

