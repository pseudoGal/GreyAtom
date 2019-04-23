import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
df=pd.read_pickle('cleanedbank10%_mark_2.pkl')
X=df.drop(['y','duration'],1)
y=df['y']
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)
X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size=0.3, random_state=0)
import xgboost 
features = X_resampled
target=y_resampled
training_features, testing_features, training_target, testing_target = \
            train_test_split(features,target, random_state=42)


exported_pipeline = make_pipeline(
    PCA(),
    GradientBoostingClassifier(learning_rate=0.5, max_depth=7, max_features=0.3, min_samples_leaf=0.08, min_samples_split=0.1, n_estimators=350, subsample=0.6000000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

from sklearn.metrics import accuracy_score,f1_score
acc=accuracy_score(testing_target,results)

from sklearn.metrics import classification_report
print(classification_report(testing_target,results))