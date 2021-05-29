import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from numpy import concatenate
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import decomposition
import warnings
warnings.filterwarnings("ignore")


color_layout_features = pd.read_pickle("color_layout_descriptor.pkl")
bow_surf  = pd.read_pickle("bow_surf.pkl")
color_hist_features  = pd.read_pickle("hist.pkl")
labels  = pd.read_pickle("labels.pkl")

# Feat. Scaling
def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

# normalization
def normalize(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))

color_layout_features_scaled = scale(color_layout_features, 0, 1)
color_hist_features_scaled = scale(color_hist_features, 0, 1)
bow_surf_scaled = scale(bow_surf, 0, 1)


features = np.hstack([color_layout_features_scaled, color_hist_features_scaled, bow_surf_scaled])


# define dataset
X, Y = features, labels
X = normalize(X)
pca = decomposition.PCA(n_components=100)
pca.fit(X)
X = pca.transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1, stratify=Y)
# split train into labeled and unlabeled
X_train_lab, X_test_unlab, y_train_lab, y_test_unlab = train_test_split(X_train, y_train, test_size=0.30, random_state=1, stratify=y_train)
# create the training dataset input
X_train_mixed = concatenate((X_train_lab, X_test_unlab))
# create "no label" for unlabeled data
nolabel = [-1 for _ in range(len(y_test_unlab))]
# recombine training dataset labels
y_train_mixed = concatenate((y_train_lab, nolabel))


from semisupervised.TSVM import S3VM
from safeu.classification.TSVM import TSVM

model = S3VM(kernel="rbf", C = 1e-5, gamma = 0.5, lamU = 1.3, probability=True)
#model = TSVM()
#model.fit(X_train_mixed, _train_mixed)
model.fit(np.vstack((X_train_lab, X_test_unlab)), np.append(y_train_lab, nolabel))
#model.fit(np.vstack((label_X_train, unlabel_X_train)), np.append(label_y_train, unlabel_y))

# predict
predict = model.predict(X_test)
acc = metrics.accuracy_score(y_test, predict)
# metric
print("accuracy", acc*100)
