#!/usr/bin/env python3
# -*- coding: utf-8 -*-  
"""
Created on Wed Apr 12 20:09:16 2017

@author: deeptichevvuri
"""    
import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

trainDataSet = pandas.read_csv("lab4-train.csv")
testDataSet = pandas.read_csv("lab4-test.csv")

trainDataArray = trainDataSet.values
testDataArray = testDataSet.values
xTrain = trainDataArray[:,0:4]
yTrain = trainDataArray[:,4]
xTest = testDataArray[:,0:4]
yTest = testDataArray[:,4]

print("Tast 1")
print("Random Forest")
kfold = model_selection.KFold(n_splits=200, random_state=5)#seed = 5 nsplits=200, randomstate=5
rfModel = RandomForestClassifier(n_estimators=50, max_features=2)#nestimators=numtrees=50, maxfeatures=maxfeatures=2   nestimators=50, maxfeatures=2
rfModel = rfModel.fit(xTrain,yTrain)
rfResults = model_selection.cross_val_score(rfModel, xTest, yTest, cv=kfold)
yhat = rfModel.predict(xTest)
confusion_matrix(yTest,yhat)
print("Accuracy= {} ".format(rfResults.mean()))
print("Confusion Matrix : \n{}".format(confusion_matrix(yTest,yhat)))

print("AdaBoost")
kfold = model_selection.KFold(n_splits=300, random_state=1)#seed = 5
dtstump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dtstump.fit(xTrain, yTrain)
dtstump_err = 1.0 - dtstump.score(xTest, yTest)
ada = AdaBoostClassifier( base_estimator=dtstump,learning_rate=1,n_estimators=500,random_state=1,algorithm="SAMME")
ada = ada.fit(xTrain, yTrain)
yhat = ada.predict(xTest)
ada_err = 1.0 - ada.score(xTest, yTest)
results = model_selection.cross_val_score(ada, xTrain, yTrain, cv=kfold)
confusion_matrix(yTest,yhat)
print(results.mean())
print(confusion_matrix(yTest,yhat))

print("Task 2: Voting CLassifier")
print("Nueral Network(NN) CLassifier")
nnClassifier = MLPClassifier(random_state=1)
nnClassifier=nnClassifier.fit(xTrain,yTrain)
scores = cross_val_score(nnClassifier, xTest, yTest, cv=10, scoring='accuracy')
yhat = nnClassifier.predict(xTest)
print("Accuracy: {}".format(scores.mean()))
print("Confusion Matrix : \n{}".format(confusion_matrix(yTest,yhat)))

print("KNN CLassifier")
knnClassifier= KNeighborsClassifier(n_neighbors=20)
knnClassifier=knnClassifier.fit(xTrain,yTrain)
scores = cross_val_score(knnClassifier, xTest, yTest, cv=10, scoring='accuracy')
yhat = knnClassifier.predict(xTest)
print("Accuracy: {}".format(scores.mean()))
print("Confusion Matrix : \n{}\n".format(confusion_matrix(yTest,yhat)))

print("Logistic Regression(LR) CLassifier")
lrClassifier=LogisticRegression(random_state=1)
lrClassifier=lrClassifier.fit(xTrain,yTrain)
scores = cross_val_score(lrClassifier, xTest, yTest, cv=10, scoring='accuracy')
yhat = lrClassifier.predict(xTest)
print("Accuracy: {}".format(scores.mean()))
print("Confusion Matrix : \n{}\n".format(confusion_matrix(yTest,yhat)))

print("Naive Bayes(NB) CLassifier")
nbClassifier=  GaussianNB() 
nbClassifier=nbClassifier.fit(xTrain,yTrain)
scores = cross_val_score(nbClassifier, xTest, yTest, cv=10, scoring='accuracy')
yhat = nbClassifier.predict(xTest)
print("Accuracy: {}".format(scores.mean()))
print("Confusion Matrix : \n{}\n".format(confusion_matrix(yTest,yhat)))

print("Decision Tree(DT) CLassifier")
dtClassifier=DecisionTreeClassifier(random_state=10)
dtClassifier=dtClassifier.fit(xTrain,yTrain)
scores = cross_val_score(dtClassifier, xTest, yTest, cv=60, scoring='accuracy')
yhat = dtClassifier.predict(xTest)
print("Accuracy: {}".format(scores.mean()))
print("Confusion Matrix : \n{}\n".format(confusion_matrix(yTest,yhat)))

print("Unweigthed Majority Vote Ensemble")
eclfunweighted = VotingClassifier(estimators=[('nn', nnClassifier), ('knn', knnClassifier), ('lr', lrClassifier), ('gnb', nbClassifier), ('dt', dtClassifier)], voting='hard')
eclfunweighted=eclfunweighted.fit(xTrain,yTrain)
scores = cross_val_score(eclfunweighted, xTest, yTest, cv=5, scoring='accuracy')
print("Ensemble Unweigthed Accuracy: {}\n".format(scores.mean()))

print("Weigthed Majority Vote Ensemble")
eclfweighted = VotingClassifier(estimators=[('nn', nnClassifier), ('knn', knnClassifier), ('lr', lrClassifier), ('gnb', nbClassifier), ('dt', dtClassifier)], voting='soft',weights=[1.01,1.01,1.02,1.01,1])
eclfweighted=eclfweighted.fit(xTrain,yTrain)
scores = cross_val_score(eclfweighted, xTest, yTest, cv=5, scoring='accuracy')
print("Ensemble Weighted Accuracy: {}\n".format(scores.mean()))

print("Task 3")
#Include Random Forest and AdaBoost
print("unweigthed Majority Vote Ensemble")
eclfweighted = VotingClassifier(estimators=[('rf', rfModel), ('ada', ada), ('nn', nnClassifier), ('knn', knnClassifier), ('lr', lrClassifier), ('gnb', nbClassifier), ('dt', dtClassifier)], voting='hard')
eclfweighted=eclfweighted.fit(xTrain,yTrain)
scores = cross_val_score(eclfweighted, xTest, yTest, cv=5, scoring='accuracy')
print("Ensemble Weighted Accuracy: {}\n".format(scores.mean()))

print("weigthed Majority Vote Ensemble")
eclfweighted = VotingClassifier(estimators=[('rf', rfModel), ('ada', ada), ('nn', nnClassifier), ('knn', knnClassifier), ('lr', lrClassifier), ('gnb', nbClassifier), ('dt', dtClassifier)], voting='soft',weights=[1.08,1.04,1.01,1.01,1.02,1.01,1])
eclfweighted=eclfweighted.fit(xTrain,yTrain)
scores = cross_val_score(eclfweighted, xTest, yTest, cv=5, scoring='accuracy')
print("Ensemble Weighted Accuracy: {}\n".format(scores.mean()))

