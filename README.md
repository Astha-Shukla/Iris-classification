# Iris-classification

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
iris = pd.read_csv("Iris.csv") 
iris.head()
iris.info()
iris.drop(['Id','Species'], axis=1).describe()

iris['Species'].value_counts()
sns.pairplot(iris.drop(['Id'], axis=1),hue='Species')
sub=iris[iris['Species']=='Iris-setosa']
sns.kdeplot(data=sub[['SepalLengthCm','SepalWidthCm']], shade=True, shade_lowest=False)
plt.title('Iris-setosa')
plt.xlabel('Sepal Length Cm')
plt.ylabel('Sepal Width Cm')
sns.kdeplot(data=sub[['PetalLengthCm','PetalWidthCm']], shade=True, shade_lowest=False)
plt.title('Iris-setosa')
plt.xlabel('Petal Length Cm')
plt.ylabel('Petal Width Cm')

sub_virginica=iris[iris['Species']=='Iris-virginica']
# Create a scatter plot of the Sepal
plt.scatter(sub_virginica['SepalLengthCm'], sub_virginica['SepalWidthCm'], marker='o', color='r')
plt.xlabel('Sepal Length Cm')
plt.ylabel('Sepal Width Cm')
plt.title('Sepal Width versus Length for virginica species')

from sklearn.model_selection import train_test_split
X=iris.drop(['Species', 'Id'], axis=1)
y=iris['Species']
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.5, shuffle=True,random_state=100)

from sklearn.svm import SVC
model=SVC(C=1, kernel='rbf', tol=0.001)
model.fit(X_train, y_train)

pred=model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))
print('\n')
print('Accuracy score is: ', accuracy_score(y_test, pred))

iris.head(20)

from sklearn.utils import shuffle
X=iris.drop('Species', axis=1)
y=iris['Species']
print('Before shuffle: ',y[0:20])
X,y = shuffle(X,y, random_state=0)
print("After shuffle: ", y[0:20])

param_grid = {'C': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
# param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': ['auto'], 'kernel': ['rbf']}
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(estimator=SVC(), param_grid=param_grid, scoring='accuracy',cv=3, verbose=1, refit=True )
grid.fit(X, y)

print(grid.best_params_)
print(grid.best_estimator_)
print('Mean cross-validated score of the best_estimator: ', grid.best_score_)
print('The number of cross-validation splits (folds/iterations): ', grid.n_splits_)

from sklearn.model_selection import KFold
X=iris.drop(['Species', 'Id'], axis=1)
y=iris['Species']

param_grid = {'C': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(estimator=SVC(), param_grid=param_grid, scoring='accuracy',
                  cv=KFold(n_splits=3, shuffle=True, random_state=0), verbose=1, refit=True )
grid.fit(X, y)

print(grid.best_params_)
print(grid.best_estimator_)
print('Mean cross-validated score of the best_estimator: ', grid.best_score_)
print('The number of cross-validation splits (folds/iterations): ', grid.n_splits_)

