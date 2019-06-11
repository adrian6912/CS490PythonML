# ICP3

import numpy as np
import requests as r
from bs4 import BeautifulSoup

# Part 1 of ICP3


class Employee(object):
    """Part 1 of ICP3"""
    employee_count = 0
    total_salary = 0

    def __init__(self, name=None, family=None, salary=None, department=None):
        self.name = name
        self.family = family
        self.salary = salary
        self.department = department
        Employee.employee_count += 1
        Employee.total_salary += salary

    def avg_salary():
        return Employee.total_salary / Employee.employee_count

    def get_name(self):
        return self.name

    def get_family(self):
        return self.family

    def get_salary(self):
        return self.salary

    def get_department(self):
        return self.department


class FulltimeEmployee(Employee):
    pass


one = Employee('John Smith', 0, 50000, 'Accounting')
two = FulltimeEmployee('Jane Smith', 3, 80000, 'IT')
three = Employee('James Howard', 3, 26000, 'Housekeeping')

print("Employee names: ", one.get_name(), two.get_name(), three.get_name())
print("Average salary: ", str(Employee.avg_salary()))


# Part 2 of ICP3

url = 'https://en.wikipedia.org/wiki/Deep_Learning'
html = r.get(url)
soup = BeautifulSoup(html.text, 'html.parser')

print(soup.title.string)

links = soup.find_all('a')

for link in links:
    href = link.get('href')
    print(href)


# Part 3 of ICP3

def replace_max(arr):
    arr[arr.argmax()] = 0


x = np.random.randint(1, high=21, size=15)
print(x)
replace_max(x)
print(x)
