# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:53:23 2019

@author: farukkutlu
"""
import numpy as np
import matplotlib.pyplot as plt
import os

a = os.getcwd() + '\\airfoils'
b = os.listdir(a)
c = os.getcwd() + '\\normalized_airfoils'
airfoils = {} # the dictionary to save all airfoils

if not os.path.exists(c):       # Creates the 'normalized_airfoils' folder
    os.mkdir(c)                             # if it does not exist.

def normalize(x_,y_):
    mid = int((len(x_) - len(x_)%2)/2 + 1)      # finds the intermediate point
    x2_1, x2_2, y2_1, y2_2 = x_[:mid], x_[mid:], y_[:mid], y_[mid:]
    if x_[1] > x_[0]:
        x2_1 = [i for i in reversed(x2_1)]
        y2_1 = [i for i in reversed(y2_1)]
    if x_[-1] < x_[-2]:
        x2_2 = [i for i in reversed(x2_2)]
        y2_2 = [i for i in reversed(y2_2)]
    x, y = np.append(x2_1,x2_2), np.append(y2_1,y2_2)
    x_tr, y_tr =0.5*(x[0]+x[-1]), 0.5*(y[0]+y[-1])
    x, y = np.append(x_tr, x), np.append(y_tr, y)
    x, y = np.append(x, x_tr), np.append(y, y_tr)
    return x, y

for i in range(len(b)):
    d = a + '\\' + b[i]
    e = c + '\\' + b[i]
    with open(d, 'r') as file:
        header = file.readline()
        x, y = np.loadtxt(file, dtype=float, unpack=True)
    x,y = normalize(x,y)
    contents = np.array([row for row in zip(x,y)])
    contents.reshape(1,int(len(contents)*len(contents[0])))
    airfoils[b[i]] = contents       # updating the dictionary
    np.savetxt(e, contents, header=header, comments='')
#    plt.figure(figsize=(10, 10))
#    plt.title(header, loc='center')
#    plt.plot(x,y,'-')
#    plt.axes().set_aspect('equal')
#    plt.show()

