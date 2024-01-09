import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from scipy.ndimage.interpolation import shift

# build a MNIST classifier that achieves over 97% accuracy
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data, mnist.target

y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
clf_accuracy = accuracy_score(y_test, predictions)
print(clf_accuracy)

param_grid = [
    {'n_neighbors': [1, 3, 5, 7, 9],
     'weights': ['uniform', 'distance'],
     'metric': ['cityblock', 'cosine', 'euclidean', 'haversine', 'minkowski']}
]

t0 = time.time()
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, n_jobs=-1, cv=3, verbose=1)
grid_search.fit(X_train, y_train)
run_time = time.time() - t0
print(run_time) # about 25 minutes

best_model_grid = grid_search.best_estimator_
print(best_model_grid)

best_score_grid = np.sqrt(-grid_search.best_score_)
print(best_score_grid)

best_params_grid = grid_search.best_params_
print(best_params_grid)

results_grid = pd.DataFrame(grid_search.cv_results_)
results_grid.sort_values(by='rank_test_score', inplace=True)


# create a function that can shift an MNIST image in any direction, create 4 shifted copies of each image and assess
# model performance on this augmented dataset
def shift_image(image, vertical_shift, horizontal_shift):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [vertical_shift, horizontal_shift], cval=0, mode='constant')
    shifted_image = shifted_image.reshape([-1])
    return shifted_image


X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for vertical_shift, horizontal_shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, vertical_shift, horizontal_shift))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

clf2 = KNeighborsClassifier(**grid_search.best_params_)
clf2.fit(X_train_augmented, y_train_augmented)
predictions = clf2.predict(X_test)
clf_accuracy = accuracy_score(y_test, predictions)
print(clf_accuracy)
