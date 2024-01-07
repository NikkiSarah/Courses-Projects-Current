# train a random forest, extra trees and SVM classifier on MNIST
import numpy as np
from sklearn.datasets import fetch_openml
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

n_cpu = os.cpu_count()
print("Number of CPUs in the system:", n_cpu)

mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data, mnist.target

X = X / 255.0
y = y.astype(np.uint8)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, stratify=y, train_size=60000)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, stratify=y_train_val, train_size=50000)

rf_clf = RandomForestClassifier(random_state=42)
et_clf = ExtraTreesClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42)
mlp_clf = MLPClassifier(random_state=42)

classifiers = [rf_clf, et_clf, svm_clf, mlp_clf]
for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(accuracy)
# of the individual classifiers, it's the SVM (0.9796), MLP (0.9769), ET (0.9731) and finally the RF (0.9691)

# train a voting classifier that performs better than any of the individual classifiers
named_classifiers = [
    ("random_forest_clf", rf_clf),
    ("extra_trees_clf", et_clf),
    ("svm_clf", svm_clf),
    ("mlp_clf", mlp_clf),
]
for vtype in ['hard', 'soft']:
    vt_clf = VotingClassifier(named_classifiers, voting=vtype, weights=None, n_jobs=n_cpu-2, verbose=True)
    vt_clf.fit(X_train, y_train)
    y_pred = vt_clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(accuracy)
# the soft classifier performs better than the hard classifier (0.9812 vs 0.9785)

# recall the accuracies from the individual classifiers were:
print([(clf.score(X_val, y_val), clf) for clf in vt_clf.estimators_])
# on the validation set, the soft classifier outperforms every individual classifier, and the hard classifier
# outperforms every individual classifier except for the SVM

# compare the individual classifiers with the best voting classifier on the test set
print([clf.score(X_test, y_test) for clf in vt_clf.estimators_])
# vt_clf = VotingClassifier(named_classifiers, voting='soft', n_jobs=n_cpu-2, verbose=True)
# vt_clf.fit(X_train, y_train)
print(vt_clf.score(X_test, y_test))
# on the test set, the soft classifier again outperforms every individual classifier

# create a new training set with the predictions from the individual classifiers
val_preds = np.empty((len(X_val), len(classifiers)), dtype=np.float32)
for idx, clf in enumerate(classifiers):
    val_preds[:, idx] = clf.predict(X_val)

# train a classifier on this new training set
blender = RandomForestClassifier(oob_score=True, random_state=42)
blender.fit(val_preds, y_val)
print(blender.oob_score_)

# evaluate the ensemble on the test set - make predictions with all the classifiers and feed them to the blender to
# get the ensemble predictions
test_preds = np.empty((len(X_test), len(classifiers)), dtype=np.float32)
for idx, clf in enumerate(classifiers):
    test_preds[:, idx] = clf.predict(X_test)
y_pred = blender.predict(test_preds)
print(accuracy_score(y_test, y_pred))
# this stacking ensemble doesn't perform quite as well as the voting ensemble (0.9775 vs 0.9838)
