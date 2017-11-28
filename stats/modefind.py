#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
from neurotools.system import *

import os,sys,pickle
from neurotools.stats.density import kdepeak
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def modefind(points,burst=10):
    '''
    Locate post-event mode in one-dimensional point-processes
    with refractoriness, mainly for neuroscience data.
    
    Removes intervals shorter than 10. 
    
    Finds peak using log-KDE approximation
    
    Parameters
    ----------
    points : np.array
    burst : int
        Default is 10. Points smaller than this are excluded
    
    Returns
    -------
    float
        Estimated mode of the distribution
    '''
    points = np.array(points)
    points = points[points>burst] # remove burst
    K   = 5
    x,y = kdepeak(np.log(K+points[points>0]))
    x   = np.exp(x)-K
    y   = y/(K+x)
    mode = x[np.argmax(y)]
    return mode

def logmodeplot(points,K=5,burst=None):
    '''
    Accepts list of ISI times.
    Finds the mode using a log-KDE density estimate
    Plots this along with histogram
    
    Parameters
    ----------
    points : np.array
    burst : int
        Default is None. If a number, points smaller than this are excluded
    
    Returns
    -------
    '''
    points = np.array(points)
    if not burst is None:
        points = points[points>burst] # remove burst
    x,y = kdepeak(np.log(K+points[points>0]))
    x   = np.exp(x)-K
    y   = y/(K+x)
    cla()
    plt.hist(points,60,normed=1,color='k')
    plt.plot(x,y,lw=2,color='r')
    ybar(x[np.argmax(y)],color='r',lw=2)
    plt.draw()
    plt.show()
    mi = np.argmax(y)
    mode = x[mi]
    return mode

def logmode(points,K=5,burst=None):
    '''
    Accepts list of ISI times.
    Finds the mode using a log-KDE density estimate
    
    Parameters
    ----------
    points : np.array
    burst : int
        Default is None. If a number, points smaller than this are excluded
    
    Returns
    -------
    '''
    points = np.array(points)
    if not burst is None:
        points = points[points>burst] # remove burst
    x,y = kdepeak(np.log(K+points[points>0]))
    x   = np.exp(x)-K
    y   = y/(K+x)
    mode = x[np.argmax(y)]
    return mode

def peakfinder5(st,K=5):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    points = np.diff(st)
    points = np.array(points)
    points = points[points>10] # remove burst
    n, bins, patches = hist(points,
        bins=np.linspace(0,500,251),
        facecolor='k',
        normed=1)
    centers = (bins[1:]+bins[:-1])/2
    x,y = kdepeak(points,
        x_grid=np.linspace(0,500,251))
    plot(x,y,color='r',lw=1)
    p1 = x[np.argmax(y)]
    x,y = kdepeak(np.log(K+points[points>0]))
    x = np.exp(x)-K
    y = y/(K+x)
    plt.plot(x,y,color='g',lw=1)
    p2 = x[np.argmax(y)]
    plt.xlim(0,500)
    return p1,p2
