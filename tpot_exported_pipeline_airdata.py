import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoLarsCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('airdata.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('pm25', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['pm25'].values, random_state=None)

# Average CV score on the training set was:-4.420723876543176
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RidgeCV()),
    StackingEstimator(estimator=RidgeCV()),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=LinearSVR(C=5.0, dual=True, epsilon=0.001, loss="epsilon_insensitive", tol=0.1)),
    GradientBoostingRegressor(alpha=0.85, learning_rate=0.1, loss="huber", max_depth=6, max_features=0.9500000000000001, min_samples_leaf=9, min_samples_split=6, n_estimators=100, subsample=0.8)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print('Mean Absolute Error = %0.4f' % np.mean(abs(results - testing_target)))
