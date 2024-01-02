# train an SVM classifier on the MNIST dataset
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import os

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import optuna

n_cpu = os.cpu_count()
print("Number of CPUs in the system:", n_cpu)

mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data, mnist.target

data = pd.concat([X, y], axis=1)
data = data.groupby("class", observed=True).sample(n=1500, replace=False, random_state=42)

X = data.drop('class', axis=1)
y = data['class']

X = X / 255.0
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)


def objective(trial):
    # define model and hyperparameters
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    c = trial.suggest_int("C", 1, 2e6)
    # gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    gamma = trial.suggest_int("gamma", 0.0001, 1)
    degree = trial.suggest_int("degree", 2, 10)

    # train and evaluate model
    svm_clf = SVC(kernel=kernel, C=c, gamma=gamma, degree=degree)
    svm_clf.fit(X_train, y_train)

    # return the evaluation metric
    y_pred = svm_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


study = optuna.create_study(storage="sqlite:///db.sqlite3", study_name="Ch5_mnist", direction="maximize")
study.optimize(objective, n_trials=30, n_jobs=n_cpu-2, show_progress_bar=True)

print("Best trial:", study.best_trial.number)
print("Best accuracy:", study.best_trial.value)
print("Best hyperparameters:", study.best_params)

# train an SVM regressor on the housing dataset
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import optuna

n_cpu = os.cpu_count()
print("Number of CPUs in the system:", n_cpu)

housing = pd.read_csv('./datasets/housing.csv')

housing['income_cat'] = pd.cut(housing.median_income,
                               bins=[0., 1.5, 3., 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(housing, housing.income_cat):
    train = housing.loc[train_idx]
    test = housing.loc[test_idx]

for set_ in (train, test):
    set_.drop("income_cat", axis=1, inplace=True)

# prepare the data
train_labels = train.median_house_value.copy()
train = train.drop('median_house_value', axis=1)
train_num = train.drop('ocean_proximity', axis=1)

test_labels = test.median_house_value.copy()
test = test.drop('median_house_value', axis=1)
test_num = test.drop('ocean_proximity', axis=1)


rooms_idx, bedrooms_idx, population_idx, households_idx = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_idx] / X[:, households_idx]
        population_per_household = X[:, population_idx] / X[:, households_idx]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_idx] / X[:, rooms_idx]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

num_attribs = list(train_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])

training_data = full_pipeline.fit_transform(train)
testing_data = full_pipeline.fit_transform(test)


def objective(trial):
    # define model and hyperparameters
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    c = trial.suggest_int("C", 1, 2e6)
    # gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    gamma = trial.suggest_int("gamma", 0.0001, 1)
    degree = trial.suggest_int("degree", 2, 10)
    epsilon = trial.suggest_float("epsilon", 0.0001, 1)

    # train and evaluate model
    svm_reg = SVR(kernel=kernel, C=c, gamma=gamma, degree=degree, epsilon=epsilon)
    svm_reg.fit(training_data, train_labels)

    # return the evaluation metric
    test_preds = svm_reg.predict(testing_data)
    mse = mean_squared_error(test_labels, test_preds)

    return mse


study = optuna.create_study(storage="sqlite:///db.sqlite3", study_name="Ch5_housing2", direction="minimize")
study.optimize(objective, n_trials=30, n_jobs=n_cpu-2, show_progress_bar=True)

print("Best trial:", study.best_trial.number)
print("Best RMSE:", np.sqrt(study.best_trial.value))
print("Best hyperparameters:", study.best_params)
