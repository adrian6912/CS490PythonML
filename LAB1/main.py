# Lab 1

# Part 1


thing = [('John', ('Physics', 80)), ('Daniel', ('Science', 90)), ('John', ('Science', 95)), ('Mark', ('Maths', 100)),
         ('Daniel', ('History', 75)), ('Mark', ('Social', 95))]

dict_thing = dict()

for item in thing:
    dict_thing[item[0]] = []
for item in thing:
    dict_thing[item[0]].append(item[1])
for item in thing:
    dict_thing[item[0]] = sorted(dict_thing[item[0]])

print(dict_thing)

# Part 2 Lab


def get_all_substrings(string):
    substrings = []
    length = len(string)
    for i in range(length):
        for j in range(i, length):
            substrings.append(string[i: j+1])
    return substrings

def get_longest(string):
    substrings = get_all_substrings(string)
    longest = ''
    for sub in substrings:
        if len(sub) > len(longest) and no_repeating_chars(sub):
            longest = sub
    return (longest, len(longest))

def no_repeating_chars(string):
    for i in range(len(string)):
        for j in range(len(string)):
            if i == j:
                continue
            elif string[i] == string[j]:
                return False
    return True

# x = input()
x = "pwwkew"

print(get_longest(x))


# Part 3 - Library management system


class Person(object):
    def __init__(self, fname='', lname='', id=0):
        self.fname = fname
        self.lname = lname
        self.id = id


class Student(Person):
    def __init__(self, fname, lname, id, fines_due=0):
        super().__init__(fname, lname, id)
        self.__fines_due = fines_due
        self.checked_out = []

    def check_out_book(self, book):
        self.checked_out.append(book)
        book.check_out()

    def get_fines(self):
        return self.__fines_due


class Book(object):
    def __init__(self, title="", author="", genre="", checked_out=False, pages=0):
        self.title = title
        self.author = author
        self.genre = genre
        self.checked_out = checked_out
        self.pages = pages

    def check_out(self):
        self.checked_out = True

    def check_in(self):
        self.checked_out = False


if __name__ == '__main__':
    student = Student(fname="Joe", lname="Smith", id=5050)
    book = Book("Of mice and men", "John Steinbeck", "Literary Fiction", pages=125)
    student.check_out_book(book)
    print(book.checked_out)
    print(4)

# Part 4

import pandas as pd
import numpy as np
import random as rnd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

data = pd.read_csv("mushrooms.csv")
data = data.apply(LabelEncoder().fit_transform)
x = data.drop('class', axis=1)
print(x)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.3)

regr = linear_model.LinearRegression()
model = regr.fit(X_train, y_train)

print("R^2 is: ", model.score(X_test, y_test))
predictions = model.predict(X_test)
print("RMSE is: ", mean_squared_error(y_test, predictions))

# Got below values after regression.
# R^2 is:  0.7394715256266025
# RMSE is:  0.06506882569451233

# Part 5

from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import preprocessing

data = pd.read_csv("./Iris.csv")
data = data.apply(LabelEncoder().fit_transform)
y = data.Species
x = data.drop('Species', axis=1).drop('Id', axis=1)
scalar = preprocessing.StandardScaler()
scalar.fit(x)
X_scaled = scalar.transform(x)
x = pd.DataFrame(X_scaled, columns=x.columns)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.3)

#Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("Accuracy of GNB:", metrics.accuracy_score(y_test, y_pred) * 100)

#SVM
svc = SVC(kernel='poly', gamma='auto')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("Accuracy of SVM with poly kernel: ", metrics.accuracy_score(y_test, y_pred) * 100)

#KNN
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
print("KNN accuracy is: ", acc_knn)
print(1)

# Output
# It appears that the GNB and SVM performed equally as well
# Accuracy of GNB: 97.77777777777777
# Accuracy of SVM with poly kernel:  97.77777777777777
# KNN accuracy is:  96.19

# Part 6

