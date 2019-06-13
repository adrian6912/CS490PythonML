# Part 2 of ICP4

from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd

# Load data 

raw_data = pd.read_csv('glass.csv', dtype=float)

# Set the target data, remove it from raw_data

y = raw_data.Type
raw_data.pop('Type')

# Set the data for the features

columns = raw_data.columns.values
features = raw_data.loc[:, columns]

# Split features and y into training and test data sets

X_train, X_test, y_train, y_test = tts(features, y, test_size=.3)

# GNB model

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("Accuracy of GNB:", metrics.accuracy_score(y_test, y_pred))

# Part 3 of ICP4
# As kernel='rbf' is the default when a kernel isn't specified, I will use kernel = 'poly'

svc = SVC(kernel='poly', gamma='auto')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

print("Accuracy of SVM with Poly kernel:", metrics.accuracy_score(y_test, y_pred))

# Part 4 of ICP4

svc = SVC(kernel='rbf', gamma='auto')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

print("Accuracy of SVM with RBF kernel:", metrics.accuracy_score(y_test, y_pred))

