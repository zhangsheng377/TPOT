import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import autosklearn.regression
import sklearn.metrics
from joblib import dump, load

# NOTE: Make sure that the class is labeled 'target' in the data file
data = pd.read_csv('airdata.csv', sep=',', dtype=np.float64)
features = data.drop('pm25', axis=1).values
training_features, testing_features, training_target, testing_target = train_test_split(features, data['pm25'].values, random_state=None)

automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=60*60,
    per_run_time_limit=60*6,
)
print("fit begin")
automl.fit(training_features, training_target)
print("fit over")
print(automl.show_models())
predictions = automl.predict(testing_features)
print("predictions:", predictions)
print("R2 score:", sklearn.metrics.r2_score(testing_target, predictions))
dump(automl, 'autosklearn_airdata.joblib')
