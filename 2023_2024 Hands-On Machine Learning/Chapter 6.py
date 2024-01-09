# train and fine-tune a decision tree for the moons dataset
import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import optuna
from sklearn.base import clone
from scipy.stats import mode

n_cpu = os.cpu_count()
print("Number of CPUs in the system:", n_cpu)

# use make_moons to generate a moons dataset
X, y = make_moons(n_samples=10000, noise=0.4)

# split the dataset into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


# use grid search with cross-validation to find good hyperparameter values
def objective(trial):
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    splitter = trial.suggest_categorical("splitter", ["best", "random"])
    max_depth = trial.suggest_int("max_depth", 1, 1000)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 40)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_features = trial.suggest_categorical("max_features", [None, "sqrt", "log2"])
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 1, 100)
    ccp_alpha = trial.suggest_float("ccp_alpha", 0, 1)

    dt_clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                    max_features=max_features, max_leaf_nodes=max_leaf_nodes, ccp_alpha=ccp_alpha,
                                    random_state=42)
    dt_clf.fit(X_train, y_train)

    y_pred = dt_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


study = optuna.create_study(storage="sqlite:///db.sqlite3", study_name="Ch6_moons", direction="maximize")
study.optimize(objective, n_trials=100, n_jobs=n_cpu-2, show_progress_bar=True)

print("Best trial:", study.best_trial.number)
print("Best accuracy:", study.best_trial.value)
print("Best hyperparameters:", study.best_params)

# train it on the full training set using the best hyperparameters and measure performance on the test set
best_params = {'criterion': 'log_loss', 'splitter': 'best', 'max_depth': 394, 'min_samples_split': 5,
               'min_samples_leaf': 14, 'max_features': None, 'max_leaf_nodes': 44, 'ccp_alpha': 0.0643725803432289}
dt_clf_best = DecisionTreeClassifier(criterion=best_params['criterion'], splitter=best_params['splitter'],
                                     max_depth=best_params['max_depth'],
                                     min_samples_split=best_params['min_samples_split'],
                                     min_samples_leaf=best_params['min_samples_leaf'],
                                     max_features=best_params['max_features'],
                                     max_leaf_nodes=best_params['max_leaf_nodes'], ccp_alpha=best_params['ccp_alpha'],
                                     random_state=42)
dt_clf_best.fit(X_train, y_train)
y_pred = dt_clf_best.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(test_accuracy)

# grow a forest
# generate 1,000 subsets of the training set, each containing 100 instances selected randomly
num_trees = 1000
num_samples = 100
sss = StratifiedShuffleSplit(n_splits=num_trees, train_size=num_samples, random_state=42)
mini_train = []
for i, (train_idx, val_idx) in enumerate(sss.split(X_train, y_train)):
    mini_X_train = X_train[train_idx]
    mini_y_train = y_train[train_idx]
    mini_train.append((i, train_idx, mini_X_train, mini_y_train))

# train a single decision tree on each subset, using the best hyperparameters found previously. Evaluate these 1,000
# trees on the test set
forest = clone([dt_clf_best for _ in range(num_trees)])

accuracy_scores = []
for tree, (_, _, mini_X_train, mini_y_train) in zip(forest, mini_train):
    tree.fit(mini_X_train, mini_y_train)

    y_pred = tree.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(test_accuracy)
print(np.mean(accuracy_scores))

# for each test set instance, generate the predictions of the 1,000 trees and keep only the most frequent prediction
Y_pred = np.empty([num_trees, len(X_test)], dtype=np.uint8)

for idx, tree in enumerate(forest):
    Y_pred[idx] = tree.predict(X_test)
majority_vote_preds = np.array(mode(Y_pred, axis=0))[0]

# evaluate these predictions on the test set
forest_accuracy = accuracy_score(y_test, majority_vote_preds)
print(forest_accuracy)
