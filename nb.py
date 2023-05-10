# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
var_smoothing_range=[1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01]
# smooth=[float(var_smoothing_range[0]),float(var_smoothing_range[1])]
# print(smooth)
# new_list = [float(x) for x in var_smoothing_range]
var_smoothing_scores=[]
t = np.array(var_smoothing_scores, dtype = np.float32)

for i in var_smoothing_range:
    nb=GaussianNB(var_smoothing=i)
    scores=cross_val_score(nb,X_train,y_train,cv=12,scoring='accuracy')
    t=np.append(t,scores.mean())

# new_list = [float(x) for x in var_smoothing_scores]
plt.plot(var_smoothing_range,t)
plt.xlabel('Value of Var_Smoothing')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Performance of Different Var_smoothing on Dev Set')

max_value = max(t)
max_index = np.where(t == max_value)
print(max_index)
print(max_value)

from sklearn.model_selection import GridSearchCV

nb2=GaussianNB(var_smoothing=1e-06)
nb2.fit(X_train,y_train)
accuracy_score(y_test,nb2.predict(X_test))

confusion_matrix(y_test,nb2.predict(X_test))