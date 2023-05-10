import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# 2
data=pd.read_csv('/kaggle/input/heart-failure-prediction/heart.csv')
gender_var={"Sex":{"M":1,"F":0}}
chest_pain_type={"ChestPainType":{"ASY":0,"NAP":1,"ATA":2,"TA":3}}
resting_ecg={"RestingECG":{"Normal":0,"LVH":1,"ST":2}}
exercise_agina={"ExerciseAngina":{"Y":0,"N":1}}
st_slope={"ST_Slope":{"Up":0,"Down":1,"Flat":2}}
data=data.replace(gender_var)
data=data.replace(chest_pain_type)
data=data.replace(resting_ecg)
data=data.replace(exercise_agina)
data=data.replace(st_slope)


data.info()


#4
#Get target data
y=data['HeartDisease']
X=data.drop(['HeartDisease'],axis=1)

#5
X.head()


#6
X.shape

#7
X.isnull().sum()

#8
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

import matplotlib.pyplot as plt

param_list={"penalty": ["l1", "l2"],
#            "max_depth": [3, None],
           "fit_intercept":[True,False]}
lg=LogisticRegression(max_iter=1000,solver='liblinear')
lg_cross_validation=GridSearchCV(lg, param_list, cv=12,scoring='accuracy')
lg_cross_validation.fit(X_train, y_train)
print("The best hyperparameters of logistic regression is: {}".format(lg_cross_validation.best_params_))
print("Best score is {}".format(lg_cross_validation.best_score_))

penalty=['l1', 'l2']
fit_intercept=[True,False]
scores_mean = lg_cross_validation.cv_results_['mean_test_score']
scores_mean = np.array(scores_mean).reshape(len(penalty), len(fit_intercept)).T

print('Best params = {}'.format(lg_cross_validation.best_params_))
print('Best score = {}'.format(scores_mean.max()))

_, ax = plt.subplots(1,1)

# Param1 is the X-axis, Param 2 is represented as a different curve (color line)
for idx, val in enumerate(penalty):
    ax.plot(fit_intercept, scores_mean[idx, :], '-o', label="penalty" + ': ' + str(val))

ax.tick_params(axis='x', rotation=0)
ax.set_title('Grid Search Result on Dev Set')
ax.set_xlabel('fit_intercept')
ax.set_ylabel('CV score (accuracy)')
ax.legend(loc='best')
ax.grid('on')

confusion_matrix(y_test,lg_cross_validation.predict(X_test))