# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:33:54 2019

@author: farukkutlu
"""
import numpy as np
import matplotlib.pyplot as plt

with open('normalized_airfoils/goe704.dat', 'r') as file:
    header = file.readline()
    x, y = np.loadtxt(file, dtype=float, unpack=True)

def camberline(x,y):
    x, y = x.tolist(), y.tolist()
    meanx, meany = [], []
    if len(x)%2 == 0:
        mid = x.index(min(x)) + 1
        midx, midy = 0.5*(x[mid]+x[mid-1]), 0.5*(y[mid]+y[mid-1])
        x.insert(mid, midx), y.insert(mid, midy)
    mid = x.index(min(x))
    chord_x, chord_y = [ min(x), x[-1] ], [ y[mid], y[-1] ]
    meanx.append(chord_x[0]), meany.append(chord_y[0])
    x_u, x_d, y_u, y_d = x[:mid], x[mid+1:], y[:mid], y[mid+1:]
    x_u.reverse(), y_u.reverse()
    mx = (np.array(x_u)+np.array(x_d))*0.5
    my = (np.array(y_u)+np.array(y_d))*0.5
    t = np.array(y_u)-np.array(y_d)
    mx, my, t = mx.tolist(), my.tolist(), t.tolist()
    meanx.extend(mx), meany.extend(my)
    meanx.append(x[-1]), meany.append(y[-1])
    t_x = [ x_d[t.index(max(t))], x_u[t.index(max(t))] ]
    t_y = [ y_d[t.index(max(t))], y_u[t.index(max(t))] ]
    return chord_x, chord_y, meanx, meany, t_x, t_y
# Calling the function
chord_x, chord_y, meanx, meany, t_x, t_y = camberline(x, y)
# Plotting the result
plt.figure(figsize=(12,12))
plt.title(header, loc='center', fontsize=16)
plt.plot(x, y, color='k', linestyle='-', linewidth=4, alpha=1)
plt.plot(chord_x,chord_y,color='deepskyblue',linestyle='-',label='Chord Line')
plt.plot(t_x, t_y, color='mediumseagreen', linestyle='-', linewidth=3,
         label='Max Thickness at '+str(round(t_x[0],3))+'c')
plt.plot(meanx, meany, 'k-.', label = 'Mean Camberline', linewidth=2)
plt.axes().set_aspect('equal')
plt.xlim(-0.05, 1.05)
plt.ylim(min(y) - 0.05, max(y) + 0.075)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.legend()
plt.show()
