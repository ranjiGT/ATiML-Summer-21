# -*- coding: utf-8 -*-
"""
Created on Sat May 22 23:43:45 2021

@author: Ranji Raj
"""


import numpy as np
import pandas as pd
path = r'C:/Users/User/Downloads/Telegram Desktop/SSL-S3VM'
color_layout_features  = pd.read_pickle(path + "/color_layout_descriptor.pkl")
bow_surf  = pd.read_pickle(path + "/bow_surf.pkl")
color_hist_features  = pd.read_pickle(path + "/hist.pkl")
labels  = pd.read_pickle(path +"/labels.pkl")

features = np.hstack([color_layout_features, color_hist_features, bow_surf])
print(features.shape)


X,Y = features,labels


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, Y)


from sklearn.semi_supervised import LabelSpreading


import os
import numpy as np


# select a mask of 20% of the train dataset
y_mask = np.random.rand(len(y_train)) < 0.2


print(len(y_mask))

print(len(X_test))


y_train[~y_mask] = -1


# evaluate label spreading on the semi-supervised learning dataset
from numpy import concatenate
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelSpreading
#from sklearn.semi_supervised import LabelPropogation


# define dataset
X, Y = features,labels
# split into train and test
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50, random_state=1, stratify=Y)
# split train into labeled and unlabeled
X_train_lab, X_train_unlab, y_train_lab, y_train_unlab = train_test_split(X, Y, test_size=0.40, random_state=1)
# create the training dataset input
X_train_mixed = concatenate((X_train_lab, X_train_unlab))
# create "no label" for unlabeled data
nolabel = [-1 for _ in range(len(y_train_unlab))]
# recombine training dataset labels
y_train_mixed = concatenate((y_train_lab, nolabel))


#x = FunctionTransformer(lambda x: X.todense())
model = LabelSpreading(max_iter=1000)


model.fit(X_train_mixed, y_train_mixed)


# make predictions on hold out test set
yhat = model.predict(X_train_unlab)
# calculate score for test set
score = accuracy_score(y_train_unlab, yhat)
# summarize score
print('Accuracy: %.3f' % (score*100))


# make predictions on hold out test set
yhat = model.predict(X_train_lab)
# calculate score for test set
score = accuracy_score(y_train_lab, yhat)
# summarize score
print('Accuracy: %.3f' % (score*100))


import os
import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
ls_pipeline = Pipeline([
    ('clf', LabelSpreading()),
])


def eval_and_print_metrics(clf, X_train, y_train, X_test, y_test):
    print("Number of training samples:", len(X_train))
    print("Unlabeled samples in training set:",
          sum(1 for x in y_train if x == -1))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Micro-averaged F1 score on test set: "
          "%0.3f" % f1_score(y_test, y_pred, average='micro'))
    print("-" * 10)
    print()


eval_and_print_metrics(ls_pipeline, X_train_mixed, y_train_mixed, X_test, y_test)