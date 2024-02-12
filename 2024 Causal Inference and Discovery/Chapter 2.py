#%% observational data and linear regression
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# linear regression
np.random.seed(42)
num_samples = 5000
alpha = 1.12
beta = 0.93
error = np.random.randn(num_samples)

X = np.random.randn(num_samples)
y = alpha + beta * X + 0.5 * error

X = sm.add_constant(X)

print(X[:5, :])

model = sm.OLS(y, X)
fitted_model = model.fit()
print(fitted_model.summary())

# reversing the order
X = np.random.randn(num_samples)
y = alpha + beta * X + 0.5 * error

y = sm.add_constant(y)

print(y[:5, :])

model = sm.OLS(X, y)
fitted_model = model.fit()
print(fitted_model.summary())

# if you don't know where you're going, you might end up somewhere else



