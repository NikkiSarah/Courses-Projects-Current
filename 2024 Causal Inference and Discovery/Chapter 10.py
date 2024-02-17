import pandas as pd
from dowhy import CausalModel
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
import json

import matplotlib as mpl
mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass

#%% doubly-robust learners
earnings_train = pd.read_csv('./data/ml_earnings_interaction_train.csv')
earnings_test = pd.read_csv('./data/ml_earnings_interaction_test.csv')

print(earnings_train.shape, earnings_test.shape)
print(earnings_train.head())
print(earnings_test.head())

nodes = ['took_a_course', 'earnings', 'age', 'python_proficiency']
edges = [('took_a_course', 'earnings'),
         ('age', 'took_a_course'),
         ('age', 'earnings'),
         ('python_proficiency', 'earnings')]
gml_string = 'graph [directed 1\n'
for node in nodes:
    gml_string += f'\tnode [id "{node}" label "{node}"]\n'
for edge in edges:
    gml_string += f'\tedge [source "{edge[0]}" target "{edge[1]}"]\n'
gml_string += ']'

model = CausalModel(data=earnings_train, treatment='took_a_course', outcome='earnings',
                    effect_modifiers='python_proficiency', graph=gml_string)
model.view_model()

estimand = model.identify_effect()
print(estimand)

est = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dr.LinearDRLearner',
                            target_units='ate',
                            method_params={
                                'init_params': {
                                    'model_propensity': LogisticRegression(),
                                    'model_regression': LGBMRegressor(n_estimators=1000, max_depth=10)
                                },
                                'fit_params': {}
                            })

earnings_test2 = earnings_test.drop(['true_effect', 'took_a_course'], axis=1)
est_test = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dr.LinearDRLearner',
                                 target_units=earnings_test2, fit_estimator=False, method_params={})

effect_pred = est_test.cate_estimates.flatten()
effect_true = earnings_test.true_effect.values
print(mean_absolute_percentage_error(effect_true, effect_pred))


def PlotEffect(true_effect, predicted_effect):
    plt.scatter(true_effect, predicted_effect, color='#00B0F0')
    plt.plot(np.sort(true_effect), np.sort(true_effect), color='#FF0000', alpha=0.7, label='Perfect model')
    plt.xlabel('True effect', alpha=0.5)
    plt.ylabel('Predicted effect', alpha=0.5)
    plt.legend()


PlotEffect(effect_true, effect_pred)

# repeat with a more complicated model
est = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dr.DRLearner',
                            target_units='ate',
                            method_params={
                                'init_params': {
                                    'model_propensity': LogisticRegression(),
                                    'model_regression': LGBMRegressor(n_estimators=1000, max_depth=10),
                                    'model_final': LGBMRegressor(n_estimators=500, max_depth=10)
                                },
                                'fit_params': {}
                            })

est_test = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dr.DRLearner',
                                 target_units=earnings_test2, fit_estimator=False, method_params={})

effect_pred = est_test.cate_estimates.flatten()
print(mean_absolute_percentage_error(effect_true, effect_pred))

PlotEffect(effect_true, effect_pred)


#%% doubly-robust ML
est = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dml.LinearDML',
                            target_units='ate',
                            method_params={
                                'init_params': {
                                    'model_y': LGBMRegressor(n_estimators=500, max_depth=10),
                                    'model_t': LogisticRegression(),
                                    'discrete_treatment': True
                                },
                                'fit_params': {}
                            })

est_test = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dml.LinearDML',
                                 target_units=earnings_test2, fit_estimator=False, method_params={})

effect_pred = est_test.cate_estimates.flatten()
print(mean_absolute_percentage_error(effect_true, effect_pred))

PlotEffect(effect_true, effect_pred)

# reduce the complexity of the outcome model and increase the number of cross-fitting folds
est = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dml.LinearDML',
                            target_units='ate',
                            method_params={
                                'init_params': {
                                    'model_y': LGBMRegressor(n_estimators=50, max_depth=10),
                                    'model_t': LogisticRegression(),
                                    'discrete_treatment': True,
                                    'cv': 4
                                },
                                'fit_params': {}
                            })

est_test = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dml.LinearDML',
                                 target_units=earnings_test2, fit_estimator=False, method_params={})

effect_pred = est_test.cate_estimates.flatten()
print(mean_absolute_percentage_error(effect_true, effect_pred))

PlotEffect(effect_true, effect_pred)

# tune hyperparameters
model_y = GridSearchCV(
    estimator=LGBMRegressor(),
    param_grid={
        'max_depth': [3, 10, 20, 100],
        'n_estimators': [10, 50, 100]
    },
    cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
)

model_t = GridSearchCV(
    estimator=LGBMClassifier(),
    param_grid={
        'max_depth': [3, 10, 20, 100],
        'n_estimators': [10, 50, 100]
    },
    cv=10, n_jobs=-1, scoring='accuracy'
)

est = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dml.LinearDML',
                            target_units='ate',
                            method_params={
                                'init_params': {
                                    'model_y': model_y,
                                    'model_t': model_t,
                                    'discrete_treatment': True,
                                    'cv': 4
                                },
                                'fit_params': {}
                            })

est_test = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dml.LinearDML',
                                 target_units=earnings_test2, fit_estimator=False, method_params={})

effect_pred = est_test.cate_estimates.flatten()
print(mean_absolute_percentage_error(effect_true, effect_pred))

PlotEffect(effect_true, effect_pred)


#%% causal forests
est = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dml.CausalForestDML',
                            target_units='ate',
                            method_params={
                                'init_params': {
                                    'model_y': LGBMRegressor(n_estimators=50, max_depth=10),
                                    'model_t': LGBMClassifier(n_estimators=50, max_depth=10),
                                    'discrete_treatment': True,
                                    'cv': 4
                                },
                                'fit_params': {}
                            })

est_test = model.estimate_effect(identified_estimand=estimand, method_name='backdoor.econml.dml.CausalForestDML',
                                 target_units=earnings_test2, fit_estimator=False, method_params={})

effect_pred = est_test.cate_estimates.flatten()
print(mean_absolute_percentage_error(effect_true, effect_pred))

PlotEffect(effect_true, effect_pred)


#%% heterogeneous treatment effects with experimental data
hillstrom_clean = pd.read_csv('./data/hillstrom_clean.csv')

with open('./data/hillstrom_clean_label_mapping.json', 'r') as f:
    hillstrom_labels_mapping = json.load(f)

hillstrom_clean = hillstrom_clean.drop(['zip_code__urban', 'channel__web'], axis=1)