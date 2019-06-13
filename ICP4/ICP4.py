# Part 2 of ICP4

from sklearn import preprocessing as p
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import numpy as np
import pandas as pd

raw_data = pd.read_csv('glass.csv')
y = raw_data.pop('Type').values
features = [x for x in raw_data.iterrows()]

X_train, X_test, y_train, y_test = tts(features, y, test_size=.1, random_state=69)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

###Errors are still on this one.
