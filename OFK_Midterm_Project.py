# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 17:59:27 2019

@author: farukkutlu
"""
# Omer Faruk Kutlu - Midterm Project
# Scientific Computations With Python
# Created: 12nd April 2019
import numpy as np
import matplotlib.pyplot as plt
import os

a = os.getcwd() + '\\airfoils'
b = os.listdir(a)
c = os.getcwd() + '\\normalized_airfoils'
k = os.getcwd() + '\\images'
l = os.getcwd() + '\\paneled_images'
airfoils = {} # the dictionary to save all airfoils
if not os.path.exists(c):
    os.mkdir(c)
    print('\x1b[33;40m' + "Directory," + '\x1b[37;40m' + '(' + c + ')' + '\x1b[33;40m'
          + ", is created with 100 airfoils." + '\x1b[0m')
else:    
    print('\x1b[33;40m' + "Directory," + '\x1b[37;40m' + '(' + c + ')' + '\x1b[33;40m' 
          + ", already exists. The data of 100 airfoils have been recreated to the directory." + '\x1b[0m')
if not os.path.exists(k):
    os.mkdir(k)
if not os.path.exists(l):
    os.mkdir(l)
def normalize(x_,y_):
    mid = int((len(x_) - len(x_)%2)/2 + 1)
    x2_1, x2_2, y2_1, y2_2 = x_[:mid], x_[mid:], y_[:mid], y_[mid:]
    if x_[1] > x_[0]:
        x2_1 = [i for i in reversed(x2_1)]
        y2_1 = [i for i in reversed(y2_1)]
    if x_[-1] < x_[-2]:
        x2_2 = [i for i in reversed(x2_2)]
        y2_2 = [i for i in reversed(y2_2)]
    return np.append(x2_1,x2_2), np.append(y2_1,y2_2)

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

def foil_normals(x, y):
    x_c = (x[1:] + x[:-1])/2    # center of x-points
    y_c = (y[1:] + y[:-1])/2    # center of y-points
    d_x = x[1:] - x[:-1]        # distance in x of two points.
    d_y = y[1:] - y[:-1]        # distance in y of two points.
    l = (d_x**2+d_y**2)**0.5    # distance between two points. (length)
    dx =  d_y/l                 # unit vector in x.
    dy = -d_x/l                 # unit vector in y.
    return x_c, y_c, dx, dy

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

def define_panels(x, y, N=40):
    R = (x.max() - x.min()) / 2.0                   # circle radius
    x_center = (x.max() + x.min()) / 2.0            # x-coordinate of circle center
    theta = np.linspace(0.0, 2.0 * np.pi, N + 1)    # array of angles
    x_circle = x_center + R * np.cos(theta)         # x-coordinates of circle
    x_ends = np.copy(x_circle)                      # x-coordinate of panels end-points
    y_ends = np.empty_like(x_ends)                  # y-coordinate of panels end-points
    # extend coordinates to consider closed surface
    x, y = np.append(x, x[0]), np.append(y, y[0])
    # compute y-coordinate of end-points by projection
    I = 0
    for i in range(N):
        while I < len(x) - 1:
            if (x[I] <= x_ends[i] <= x[I + 1]) or (x[I + 1] <= x_ends[i] <= x[I]):
                break
            else:
                I += 1
        a = (y[I + 1] - y[I]) / (x[I + 1] - x[I])
        b = y[I + 1] - a * x[I + 1]
        y_ends[i] = a * x_ends[i] + b
    y_ends[N] = y_ends[0]
    # create panels
    panels = np.empty(N, dtype=object)
    for i in range(N):
        panels[i] = Panel(x_ends[i], y_ends[i], x_ends[i + 1], y_ends[i + 1])
    return panels

def plot(header, x, y):
#    x_c = [(x[i+1]+x[i])/2 for i in range(len(x)-1)]  # center of x-coords
#    y_c = [(y[i+1]+y[i])/2 for i in range(len(y)-1)]  # center of y-coords

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

def plot_normals(header, x, y):
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
    t = plt.annotate(str(round(theta,2))+'\u00b0'+' '+cusp_,
    xy=(0.99,-0.01), xycoords='data', xytext=(-100,-50),
    textcoords='offset points', arrowprops=dict(arrowstyle='fancy',
    fc='0.6', connectionstyle="angle3,angleA=0,angleB=-90"))
    t.set_bbox(dict(facecolor='crimson', alpha=.9, edgecolor='red'))
    plt.ylim(min(y) - 0.075, max(y) + 0.15)             # y-limit
    plt.legend()

def print_all(l):
    bads = []
    for i in range(len(l)):
        dir_ = c + '\\' + b[i]
        with open(dir_, 'r') as file:
            header = file.readline()
            x, y = np.loadtxt(file, dtype=float, unpack=True)
        try:
            plot(header, x, y)
            plt.savefig(os.getcwd() + '\\images\\' + b[i][:-4] + '.jpg', dpi=300)
            plt.show()
        except:
            bads.append(header)
    for name in bads:
        print(name, 'could not be plotted.')
            
def panel_all(l):
    bads = []
    for i in range(len(l)):
        dir_ = c + '\\' + b[i]
        with open(dir_, 'r') as file:
            header = file.readline()
            x, y = np.loadtxt(file, dtype=float, unpack=True)
        try:
            plot_normals(header, x, y)
            plt.savefig(os.getcwd() + '\\paneled_images\\' + b[i][:-4]
            + '.jpg', dpi=300)
            plt.show()
        except:
            bads.append(header)
    for name in bads:
        print(name, 'could not be paneled.')

while True:
    foil_name = input('\x1b[37;40m'+'To close the program type '+'\x1b[33;40m'
                      +'\'close\'\n'+'\x1b[37;40m'
                      +'To open the airfoils list type '+'\x1b[33;40m'
                      +'\'airfoils\'\n'+'\x1b[37;40m'
                      +'To print all the airfoils and save the images, type '+'\x1b[33;40m'
                      +'\'print all\'\n'+'\x1b[37;40m'
                      +'To plot the paneled airfoil, type '+'\x1b[33;40m'
                      +'\'panel airfoilname\'\n'+'\x1b[37;40m'
                      +'To panel all the airfoils and save the images, type '+'\x1b[33;40m'
                      +'\'panel all\'\n'+'\x1b[37;40m'
                      +'Enter the airfoil name or the index from the airfoil '
                      +'list: '+'\x1b[36;40m' + ' ')
    try:
        val = int(foil_name)
        if int(foil_name) <= len(b)-1 and int(foil_name) >= 0:
            path = c + '\\' + b[int(foil_name)]
            with open(path, 'r') as file:
                header = file.readline()
                x, y = np.loadtxt(file, dtype=float, unpack=True)
            plot(header, x, y)
            plt.show()
        else:
            print('Enter an integer between 0 - ' + str(len(b)-1) + '.')
            continue
        
    except ValueError:
        foil_name = foil_name + '.dat'
        if foil_name[:-4] == 'close':
            print('Program is closed.')
            break
        elif foil_name[:-4] == 'airfoils':
            print(b)
            continue
        elif foil_name[:-4] == 'print all':
            print_all(b)
            print('Program is closed.')
            break
        elif foil_name[:-4] == 'panel all':
            panel_all(b)
            print('Program is closed.')
            break
        elif foil_name not in b:
            if 'panel' in foil_name:
                if foil_name[6:] in b:
                    path = c + '\\' + foil_name[6:]
                    with open(path, 'r') as file:
                        header = file.readline()
                        x, y = np.loadtxt(file, dtype=float, unpack=True)
                    plot_normals(header, x, y)
                    plt.show()
                else:
                    print('Please enter an airfoil from the list \'airfoils\'.')
                    continue
            else:
                print('Please enter an airfoil from the list \'airfoils\'.')
                continue
        elif foil_name in b:
            path = c + '\\' + foil_name
            with open(path, 'r') as file:
                header = file.readline()
                x, y = np.loadtxt(file, dtype=float, unpack=True)
            plot(header, x, y)
            plt.show()
