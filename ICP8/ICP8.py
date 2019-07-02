import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split
dataset = pd.read_csv("diabetes.csv", header=None).values
import numpy as np
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, 0:8], dataset[:, 8],
                                                    test_size=0.25, random_state=87)
# PART 1 OF ICP8

np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(30, input_dim=8, activation='relu')) # hidden layer
my_first_nn.add(Dense(25, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0,
                                     initial_epoch=0)
print(my_first_nn.summary())
score = my_first_nn.evaluate(X_test, Y_test, verbose=0)
print("Accuracy is: ", score[1])

# PART 2 OF ICP8

data = pd.read_csv("breast cancer.csv")
x = data.drop('diagnosis', axis=1).drop('id', axis=1).drop("Unnamed: 32", axis=1).values
y = data['diagnosis'].astype('category').cat.codes.values


X_train, X_test, Y_train, Y_test = train_test_split(x, y,
                                                    test_size=0.25, random_state=87)
np.random.seed(155)
nn = Sequential() # create model
nn.add(Dense(60, input_dim=30, activation='relu')) # hidden layer
nn.add(Dense(20, activation='relu')) # hidden layer
nn.add(Dense(1, activation='sigmoid')) # output layer
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_fitted = nn.fit(X_train, Y_train, epochs=100, verbose=0,
                                     initial_epoch=0)
print(nn.summary())
score = nn.evaluate(X_test, Y_test, verbose=0)
print("Accuracy is: ", score[1])
