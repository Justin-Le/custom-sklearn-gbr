# Author: Justin Le <lejustin.lv@gmail.com>
#
# License: BSD 3 Clause

import numpy as np
import matplotlib.pyplot as plt

import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import scale
import pdb

SEED = 13
X, y = datasets.load_boston(return_X_y=True)
X = scale(X) # standardize
cv = StratifiedKFold(n_splits=10, random_state=SEED)

# Reproduce DecisionTreeRegressor() results using new gradient_boosting.py
########################################

os.system("bash restore.sh")

clf = DecisionTreeRegressor(criterion='mse', random_state=SEED)
cv_scores_original = cross_val_score(clf, X, y, cv=cv, scoring='neg_mean_squared_error') 

os.system("bash run.sh")

clf = DecisionTreeRegressor(criterion='mse', random_state=SEED)
cv_scores_new = cross_val_score(clf, X, y, cv=cv, scoring='neg_mean_squared_error') 

print("Does the new gradient_boosting.py produce the same ten-fold cross-validation scores?:")
print(cv_scores_original==cv_scores_new)

# Reproduce cross-validation profile for tree GBR using new gradient_boosting.py
########################################

ensemble_sizes = range(50, 550, 50)
score_means = np.zeros_like(ensemble_sizes, dtype=np.float)
score_mins = np.zeros_like(ensemble_sizes, dtype=np.float)

os.system("bash restore.sh")

for num_est, score_idx in zip(ensemble_sizes, range(len(score_means))):
    params = {'criterion': 'mse', 'random_state': SEED, 'n_estimators': num_est, 
              'learning_rate': 0.1, 'loss': 'ls'}

    clf = ensemble.GradientBoostingRegressor(**params)

    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='neg_mean_squared_error') 
    cv_scores = -1*cv_scores # reverse the negative in neg_mean_squared_error

    score_means[score_idx] = np.mean(cv_scores)
    score_mins[score_idx] = min(cv_scores)

    # print("MSE average: %.4f" % np.mean(cv_scores))
    # print("MSE sigma: %.4f" % np.std(cv_scores))
    # print("\n")

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(ensemble_sizes, score_means, 'b-', alpha=0.8)
plt.plot(ensemble_sizes, score_mins, 'g-', alpha=0.8)
plt.xlabel('Boosting Iterations')
plt.ylabel('CV mean (blue) & min (green)')

score_means = np.zeros_like(ensemble_sizes, dtype=np.float)
score_mins = np.zeros_like(ensemble_sizes, dtype=np.float)

os.system("bash run.sh")

for num_est, score_idx in zip(ensemble_sizes, range(len(score_means))):
    params = {'base_estimator': DecisionTreeRegressor(), 'criterion': 'mse', 'random_state': SEED, 'n_estimators': num_est, 
              'learning_rate': 0.1, 'loss': 'ls'}

    clf = ensemble.GradientBoostingRegressor(**params)

    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='neg_mean_squared_error') 
    cv_scores = -1*cv_scores # reverse the negative in neg_mean_squared_error

    score_means[score_idx] = np.mean(cv_scores)
    score_mins[score_idx] = min(cv_scores)

    # print("MSE average: %.4f" % np.mean(cv_scores))
    # print("MSE sigma: %.4f" % np.std(cv_scores))
    # print("\n")

plt.subplot(1, 2, 2)
plt.plot(ensemble_sizes, score_means, 'b-', alpha=0.8)
plt.plot(ensemble_sizes, score_mins, 'g-', alpha=0.8)
plt.xlabel('Boosting Iterations')
plt.ylabel('CV mean (blue) & min (green)')
plt.show()

# Produce a linear visual when fitting/predicting with linear-base GBR on a single variable
# Repeat for 4 different variables
########################################

params = {'base_estimator': LinearRegression(), 'n_estimators': 100, 
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

plt.figure()

for subplot_idx in range(5, 9):
    clf.fit(X[:, subplot_idx-1].reshape(-1, 1), y)
    plt.subplot(2, 2, subplot_idx-4)
    plt.plot(X[:, subplot_idx-1], y, 'ro', alpha=0.6)
    plt.plot(X[:, subplot_idx-1], clf.predict(X[:, subplot_idx-1].reshape(-1, 1)), 'b-')

plt.show()
