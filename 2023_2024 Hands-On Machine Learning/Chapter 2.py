import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
import time

housing = pd.read_csv('./datasets/housing.csv')

housing.info()
desc = housing.describe()

housing['income_cat'] = pd.cut(housing.median_income,
                               bins=[0., 1.5, 3., 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(housing, housing.income_cat):
    train = housing.loc[train_idx]
    test = housing.loc[test_idx]

for set_ in (train, test):
    set_.drop("income_cat", axis=1, inplace=True)

# prepare the training data
train_labels = train.median_house_value.copy()
train = train.drop('median_house_value', axis=1)

train_num = train.drop('ocean_proximity', axis=1)

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

# train an SVM model on the training data
svm_reg = SVR()
svm_reg.fit(training_data, train_labels)

predictions = svm_reg.predict(training_data)
svm_rmse = np.sqrt(mean_squared_error(train_labels, predictions))
print(svm_rmse)

scores = cross_val_score(svm_reg, training_data, train_labels, scoring='neg_mean_squared_error', cv=10)
svm_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation", scores.std())


display_scores(svm_rmse_scores)

# experiment with different values of 'C' and 'gamma'
param_grid = [
    {'kernel': ['linear'], 'C': [2e5, 2.5e5, 5e5, 1e6, 1.5e6, 2e6]},
    {'kernel': ['rbf'], 'C': [2e5, 2.5e5, 5e5, 1e6, 1.5e6, 2e6], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]}
    ]

t0 = time.time()
grid_search = GridSearchCV(SVR(), param_grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=3, verbose=1)
grid_search.fit(training_data, train_labels)
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

# use a randomised instead of grid search
t0 = time.time()
random_search = RandomizedSearchCV(SVR(), param_grid, n_iter=50, scoring='neg_mean_squared_error',
                                   n_jobs=-1, cv=3, verbose=1)
random_search.fit(training_data, train_labels)
run_time = time.time() - t0
print(run_time) # about 40 minutes

best_model_random = random_search.best_estimator_
print(best_model_random)

best_score_random = np.sqrt(-random_search.best_score_)
print(best_score_random)

best_params_random = random_search.best_params_
print(best_params_random)

results_random = pd.DataFrame(random_search.cv_results_)
results_random.sort_values(by='rank_test_score', inplace=True)

# add a transformer in the preparation pipeline to select only the most important attributes
feature_selection_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', RFECV(RandomForestRegressor(), verbose=1, cv=3, scoring='neg_mean_squared_error', n_jobs=-1))
])

training_data2 = feature_selection_pipeline.fit_transform(train, train_labels)

selected_features = feature_selection_pipeline['feature_selection'].get_support()
cat_features = feature_selection_pipeline['preparation'].named_transformers_['cat'].get_feature_names_out()
num_features = np.append(train.columns[:-1], ['rooms_per_household', 'population_per_household', 'bedrooms_per_room'])
input_features = np.append(num_features, cat_features)
output_features = input_features[selected_features]

# create a single pipeline that does the full data preparation plus the final prediction
final_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', RFECV(RandomForestRegressor(), verbose=1, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)),
    ('model', SVR(**random_search.best_params_))
])

final_pipeline.fit(train, train_labels)
train_preds = final_pipeline.predict(train)
svm_rmse_train = np.sqrt(mean_squared_error(train_labels, train_preds))
print(svm_rmse_train)

test_labels = test.median_house_value.copy()
test = test.drop('median_house_value', axis=1)

predictions = final_pipeline.predict(test, test_labels)
svm_rmse_test = np.sqrt(mean_squared_error(train_labels, train_preds))
print(svm_rmse_test)
