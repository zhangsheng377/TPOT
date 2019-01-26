import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler, PolynomialFeatures, RobustScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import OneHotEncoder, StackingEstimator
from xgboost import XGBRegressor
from sklearn.metrics.regression import r2_score, mean_squared_error
import pylab as pl

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('airdata.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('pm25', axis=1).values
training_features, testing_features, training_target, testing_target = train_test_split(features, tpot_data['pm25'].values, random_state=None)

# Average CV score on the training set was:-4.128845406107942
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            StackingEstimator(estimator=RidgeCV()),
            StackingEstimator(estimator=RidgeCV()),
            StackingEstimator(estimator=DecisionTreeRegressor(max_depth=9, min_samples_leaf=7, min_samples_split=12)),
            StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="quantile", max_depth=1, max_features=0.8, min_samples_leaf=3, min_samples_split=19, n_estimators=100, subsample=0.9500000000000001)),
            SelectPercentile(score_func=f_regression, percentile=27),
            MaxAbsScaler()
        ),
        make_pipeline(
            StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=9, min_child_weight=7, n_estimators=100, nthread=1, subsample=0.6500000000000001)),
            MaxAbsScaler()
        )
    ),
    VarianceThreshold(threshold=0.005),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    VarianceThreshold(threshold=0.001),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.01, max_depth=5, min_child_weight=7, n_estimators=100, nthread=1, subsample=0.9500000000000001)),
    StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=29, p=1, weights="uniform")),
    OneHotEncoder(minimum_fraction=0.1, sparse=False, threshold=10),
    RobustScaler(),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.85, learning_rate=0.5, loss="huber", max_depth=8, max_features=0.55, min_samples_leaf=1, min_samples_split=18, n_estimators=100, subsample=0.9000000000000001)),
    StackingEstimator(estimator=LinearSVR(C=0.5, dual=False, epsilon=0.1, loss="squared_epsilon_insensitive", tol=1e-05)),
    XGBRegressor(learning_rate=0.1, max_depth=6, min_child_weight=3, n_estimators=100, nthread=1, subsample=1.0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print('Mean Absolute Error = %0.4f' % np.mean(abs(results - testing_target)))
print('R-squared:%0.4f MSE:%0.4f' % (r2_score(testing_target, results), mean_squared_error(testing_target, results)))

print(training_target.size)
print(testing_target.size)
x_range = range(0, training_target.size)
print(x_range)
pl.plot(x_range, training_target, '-b')
#pl.show()
x_range = range(training_target.size, training_target.size + testing_target.size)
print(x_range)
pl.plot(x_range, testing_target, '-g')
pl.plot(x_range, results, '--r')
pl.show()

