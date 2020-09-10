import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline



data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, data['target'], random_state=None)


pipeline = make_pipeline( GradientBoostingClassifier(learning_rate=0.5, max_depth=10,
 max_features=0.7, min_samples_leaf=9, min_samples_split=16, n_estimators=100, subsample=0.85))
)

pipeline.fit(training_features, training_target)
results = pipeline.predict(testing_features)
