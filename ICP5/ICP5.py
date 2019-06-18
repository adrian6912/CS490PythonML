import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ICP5 Part 1

data = pd.read_csv('train.csv')
x = data.GarageArea
y = data.SalePrice

plt.scatter(x, y)
plt.show()

# Remove the very high values
no_outliers = data[(np.abs(stats.zscore(data.GarageArea)) < 3)]
# Remove the 0's
no_outliers = no_outliers[(no_outliers.GarageArea != 0)]

x = no_outliers.GarageArea
y = no_outliers.SalePrice
plt.scatter(x, y)
plt.show()


# ICP5 Part 2

data = pd.read_csv('winequality-red.csv')
# Drop all 0's
data = data[(data != 0).all(1)]

# Correlation
num_features = data.select_dtypes(include=[np.number])
corr = num_features.corr()
print(corr)
# Alcohol, volatile acidity, and sulphates are the most correlated features in descending order.

# 1. Regression on all features of the dataset
Y = data.quality
X = data.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=.3)

regr = linear_model.LinearRegression()
model = regr.fit(X_train, y_train)

print("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
print('RMSE is: \n', mean_squared_error(y_test, predictions))


# 2. Regression with the highest correlated features: alcohol, volatile acidity, and sulphates.

Y = data.quality
X = data[['alcohol', 'volatile acidity', 'sulphates']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=40, test_size=.3)

regr = linear_model.LinearRegression()
model = regr.fit(X_train, y_train)

print("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
print('RMSE is: \n', mean_squared_error(y_test, predictions))
