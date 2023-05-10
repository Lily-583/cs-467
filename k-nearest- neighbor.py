import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#17
import matplotlib.pyplot as plt 
%matplotlib inline
# choose k between 1 to 31
k_range = range(1, 31)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=12, scoring='accuracy')
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy on Dev Set')
plt.title("Performance of different K-values on Dev Set")
plt.show()

#18
max_value = max(k_scores)
max_index = k_scores.index(max_value)
print(max_index)
print(max_value)

#19
from sklearn.model_selection import GridSearchCV
#create new a knn model
knn2 = KNeighborsClassifier()
k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
#create a dictionary of all values we want to test for n_neighbors
#param_grid = {‘n_neighbors’: np.arange(1, 31)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=12)
#fit model to data
knn_gscv.fit(X_train, y_train)
accuracy_score(y_test,knn_gscv.predict(X_test))


#20
confusion_matrix(y_test,knn_gscv.predict(X_test))