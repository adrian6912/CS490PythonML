# Part 1 of ICP3

import numpy as np
import requests as r
from bs4 import BeautifulSoup


class Employee(object):
    """Part 1 of ICP3"""

    def __init__(self, name=None, family=None, salary=None, department=None):
        self.name = name
        self.family = family
        self.salary = salary
        self.department = department

class FulltimeEmployee(Employee):
    pass

def avg_salary(employee_list):
    salary_total = 0
    for employee in employee_list:
        salary_total += employee.salary
    return salary_total / len(employee_list)


xy = Employee('John Smith', 0, 50000, 'Accounting')
xx = FulltimeEmployee('Jane Smith', 3, 80000, 'IT')
employees = []


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
