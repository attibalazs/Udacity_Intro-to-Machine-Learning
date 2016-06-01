#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np

data_dict = pickle.load(open("../my_final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### your code goes here

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
     features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)
print("accuracy is ", acc)

#print labels_test.count(0.0)*1.0/len(pred)
print labels_test
print pred
print precision_score(labels_test, pred)
print recall_score(labels_test, pred)


predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]


acc = accuracy_score(true_labels, predictions)
print("accuracy is ", acc)
print "precision", precision_score(true_labels, predictions)
print "recall", recall_score(true_labels, predictions)
