# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:56:08 2019

@author: farukkutlu
"""

import numpy as np
import matplotlib.pyplot as plt

with open('airfoils/eh2012.dat', 'r') as file:
    header = file.readline()
    x, y = np.loadtxt(file, dtype=float, unpack=True)

class Panel:
    def __init__(self, xa, ya, xb, yb):
        self.xa, self.ya = xa, ya       # panel starting-point
        self.xb, self.yb = xb, yb       # panel ending-point
        self.xc, self.yc = (xa + xb) / 2, (ya + yb) / 2     # panel center
        self.length = np.sqrt((xb - xa)**2 + (yb - ya)**2)  # panel length
        # orientation of panel (angle between x-axis and panel's normal)
        if xb - xa <= 0.0:
            self.beta = np.arccos((yb - ya) / self.length)
        elif xb - xa > 0.0:
            self.beta = np.pi + np.arccos(-(yb - ya) / self.length)

def define_panels(x, y, N):
    R = 0.5*(x.max() - x.min())                     # radius of the circle
    x_c = (x.max() + x.min()) / 2.0                 # x-coordinate of circle center
    theta = np.linspace(0.0, 2.0*np.pi, N+1)        # array of angles
    x_circle = x_c + R*np.cos(theta)                # x-coordinates of circle
    x_last = np.copy(x_circle)                      # x-coordinate of tr. edge
    y_last = np.empty_like(x_last)                  # y-coordinate of tr. edge
    # to close the trailing edge gap
    x, y = np.append(x, x[0]), np.append(y, y[0])
    # calculating the y-points of the panels
    j = 0
    for i in range(N):
        while j < len(x)-1:
            if (x[j] <= x_last[i] <= x[j+1]) or (x[j+1] <= x_last[i] <= x[j]):
                break
            else:
                j += 1
        a = (y[j + 1] - y[j])/(x[j + 1] - x[j])
        b = y[j + 1] - a*x[j + 1]
        y_last[i] = a*x_last[i] + b
    y_last[N] = y_last[0]
    # creating panels
    panels = np.empty(N, dtype=object)
    for i in range(N):
        panels[i] = Panel(x_last[i], y_last[i], x_last[i + 1], y_last[i + 1])
    return panels

def foil_normals(x, y):
    x_c = (x[1:] + x[:-1])/2    # center of x-points
    y_c = (y[1:] + y[:-1])/2    # center of y-points
    d_x = x[1:] - x[:-1]        # distance in x of two points.
    d_y = y[1:] - y[:-1]        # distance in y of two points.
    l = (d_x**2+d_y**2)**0.5    # distance between two points. (length)
    dx =  d_y/l                 # unit vector in x.
    dy = -d_x/l                 # unit vector in y.
    return x_c, y_c, dx, dy

def cusp(x,y):
    vx1, vy1, vx2, vy2 = x[0]-x[1], y[0]-y[1], x[-1]-x[-2], y[-1]-y[-2]
    l1 = (vx1**2+vy1**2)**0.5
    l2 = (vx2**2+vy2**2)**0.5
    theta = np.arccos((vx1*vx2+vy1*vy2)/(l1*l2))
    if 2.5 <= np.rad2deg(theta) <= 5.0:
        cusp_ = 'almost cusped'
    elif 0.0 <= np.rad2deg(theta) <= 2.5:
        cusp_ = 'cusped'
    else:
        cusp_ = 'pointed'
    return cusp_, vx1, vy1, vx2, vy2, np.rad2deg(theta)

def camberline(x,y):
    x, y = x.tolist(), y.tolist()
    if y[0] == y[-1]:
        if len(x)%2 != 0:
            mid = x.index(min(x))
            x1, x2, y1, y2 = x[:mid+1], x[mid:], y[:mid+1], y[mid:]
            meanx, meany = x2 , []
        else:
            mid = x.index(min(x))
            x1, x2, y1, y2 = x[:mid], x[mid+1:], y[:mid], y[mid+1:]
            x1.reverse(), x1.pop()
            meanx, meany = [min(x)] + (np.array(np.array(x1)+np.array(x2))/2).tolist() + [x[-1]], [y[x.index(min(x))]]
    else:
        if len(x)%2 != 0:
            mid = x.index(min(x))
            x1, x2, y1, y2 = x[:mid], x[mid+1:], y[:mid], y[mid+1:]
            meanx, meany = [min(x)] + x2 , [y[mid]]
        else:
            mid = x.index(min(x))
            x1, x2, y1, y2 = x[:mid+1], x[mid+1:], y[:mid+1], y[mid+1:]
            meanx, meany = [min(x)] + x2 , [y[x.index(min(x))]]
    max_t, t, t_x, t_y = 0, 0, 0, 0
    y1.reverse()
    for ty1, ty2 in zip(y1, y2):
        meany.append((ty1+ty2)/2)
        t = ty1 - ty2
        if t > max_t:
            max_t = t
            t_x, t_y = [x[y.index(ty1)], x[y.index(ty1)]], [ty1, ty2]
    if y[0] == y[-1]:
        if len(x)%2 == 0:
            meany.append(y[-1])
    return meanx, meany, t_x, t_y

def plot(header, x, y):
    x_l = x.tolist()
    camberx, cambery, tx, ty = camberline(x, y)
    x,y = np.append(x, x[0]), np.append(y, y[0])
    min_ = x_l.index(min(x_l))
    chordx, chordy = [ min(x), max(x) ], [ y[min_], (y[0]+y[-2])*0.5 ]
    plt.figure(figsize=(15, 15))
    plt.plot(chordx, chordy, color='deepskyblue', linestyle='-', label='Chord Line')  # Chord Line
    plt.plot(camberx, cambery, 'k-.', label = 'Mean Camberline', linewidth=2)   # Mean Camberline
    plt.plot(tx, ty, color='mediumseagreen', linestyle='-', linewidth=3,
             label='Max Thickness at '+str(round(tx[0],3))+'c')                 # Max thickness
    plt.title(header, loc='center', fontsize=16)                                # header
    plt.plot(x, y, color='k', linestyle='-', linewidth=4, alpha=1)              # plot of airfoil
    plt.axes().set_aspect('equal')                                              # aspect ratio
    plt.xlim(-0.05, 1.05)                                                       # x-limit
    plt.ylim(min(y) - 0.05, max(y) + 0.075)                                     # y-limit
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.legend()

def plot_panels(header, x, y):
    plot(header, x, y)
    cusp_, vx1, vy1, vx2, vy2, theta = cusp(x,y)
    panels = define_panels(x, y, N=20)
    # plot paneled geometry
    plt.plot(np.append([panel.xa for panel in panels], panels[0].xa),
                np.append([panel.ya for panel in panels], panels[0].ya),
                linestyle='-', linewidth=2, marker='o', markersize=6, 
                color='red', label='Panel Lines', alpha=1)
    plt.quiver([panel.xc for panel in panels],[panel.yc for panel in panels],
               np.cos([panel.beta for panel in panels]),
               np.sin([panel.beta for panel in panels]), 
           alpha=0.8, scale=20, width=0.004)
    
    plt.quiver(x[0], y[0], vx1, vy1, width = 0.003, color='crimson')
    plt.quiver(x[-2], y[-2], vx2, vy2, width = 0.003, color='crimson')
    t = plt.annotate(str(round(theta,2))+'\u00b0,'+' '+cusp_,
    xy=(1.01,-0.01), xycoords='data', xytext=(-100,-60),
    textcoords='offset points', arrowprops=dict(arrowstyle='fancy',
    fc='0.6', connectionstyle="angle3,angleA=0,angleB=-40"))
    t.set_bbox(dict(facecolor='crimson', alpha=.9, edgecolor='red'))
    plt.ylim(min(y) - 0.075, max(y) + 0.15)             # y-limit
    plt.legend()

plot_panels(header, x, y)

