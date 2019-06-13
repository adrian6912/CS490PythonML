# Part 2 of ICP4

from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd

raw_data = pd.read_csv('glass.csv', dtype=float)
y = raw_data.Type
raw_data.pop('Type')

columns = raw_data.columns.values
features = raw_data.loc[:, columns]

X_train, X_test, y_train, y_test = tts(features, y, test_size=.3)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("Accuracy of GNB:", metrics.accuracy_score(y_test, y_pred))

# Part 3 of ICP4
# Using kernel = 'linear'

svc = SVC(kernel='poly', gamma='auto')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

print("Accuracy of SVM with Poly kernel:", metrics.accuracy_score(y_test, y_pred))

# Part 4 of ICP4
# The SVM automatically uses the RBF kernel when none is provided, so this model is identical to the SVM model above

svc = SVC(kernel='rbf', gamma='auto')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

print("Accuracy of SVM with RBF kernel:", metrics.accuracy_score(y_test, y_pred))

