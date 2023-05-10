# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

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

#Get target data
y=data['HeartDisease']
X=data.drop(['HeartDisease'],axis=1)

X.head()

X.shape

X.isnull().sum()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

param_list={"min_samples_leaf": [1,2,3,4,5],
#            "max_depth": [3, None],
           "criterion": ["gini", "entropy"]}
tree=DecisionTreeClassifier()
tree_cross_validation=GridSearchCV(tree, param_list, cv=12,scoring='accuracy')
tree_cross_validation.fit(X_train, y_train)
print("The best hyperparameters of decision tree is: {}".format(tree_cross_validation.best_params_))
print("Best score is {}".format(tree_cross_validation.best_score_))

min_leaf=[1,2,3,4,5]
criterion=['gini','entropy']
scores_mean = tree_cross_validation.cv_results_['mean_test_score']
scores_mean = np.array(scores_mean).reshape(len(criterion), len(min_leaf)).T

print('Best params = {}'.format(tree_cross_validation.best_params_))
print('Best score = {}'.format(scores_mean.max()))

_, ax = plt.subplots(1,1)

# Param1 is the X-axis, Param 2 is represented as a different curve (color line)
for idx, val in enumerate(min_leaf):
    ax.plot(criterion, scores_mean[idx, :], '-o', label="min leaf" + ': ' + str(val))

ax.tick_params(axis='x', rotation=0)
ax.set_title('Grid Search Result on Dev Set')
ax.set_xlabel('Criterion (gini/entropy)')
ax.set_ylabel('CV score (accuracy)')
ax.legend(loc='best')
ax.grid('on')

confusion_matrix(y_test,tree_cross_validation.predict(X_test))


