import os,sys,pickle
from itertools import izip
from numpy import *
from neurotools.density import kdepeak

def modefind(allisi,burst=10):
    '''
    Removes intervals shorter than 10m
    Finds peak using log-KDE approximation
    '''
    try:
        allisi = array(allisi)
        allisi = allisi[allisi>burst] # remove burst
        K   = 5
        x,y = kdepeak(log(K+allisi[allisi>0]))
        x   = exp(x)-K
        y   = y/(K+x)
        return x[argmax(y)]
    except:
        return NaN

def logmodeplot(allisi):
    '''
    Accepts list of ISI times.
    Finds the mode using a log-KDE density estimate
    Plots this along with histogram
    Does not remove bursts
    '''
    allisi = array(allisi)
    K   = 5
    x,y = kdepeak(log(K+allisi[allisi>0]))
    x   = exp(x)-K
    y   = y/(K+x)
    cla()
    hist(allisi,60,normed=1,color='k')
    plot(x,y,lw=2,color='r')
    ybar(x[argmax(y)],color='r',lw=2)  
    draw()
    show()
    return x[argmax(y)]
        
def logmode(allisi):
    '''
    Accepts list of ISI times.
    Finds the mode using a log-KDE density estimate
    Does not remove bursts
    '''
    allisi = array(allisi)
    K   = 5
    x,y = kdepeak(log(K+allisi[allisi>0]))
    x   = exp(x)-K
    y   = y/(K+x)
    return x[argmax(y)]
    
def peakfinder5(st):
    '''
    Found this with the old unit classification code.
    Haven't had time to reach it and check out what it does
    '''
    allisi = diff(st)
    allisi = array(allisi)
    allisi = allisi[allisi>10] # remove burst
    n, bins, patches = hist(allisi,bins=linspace(0,500,251),facecolor='k',normed=1)   
    centers = (bins[1:]+bins[:-1])/2
    x,y = kdepeak(allisi,x_grid=linspace(0,500,251))
    plot(x,y,color='r',lw=1)
    p1 = x[argmax(y)]
    K = 5
    x,y = kdepeak(log(K+allisi[allisi>0]))
    x = exp(x)-K
    y = y/(K+x)
    plot(x,y,color='g',lw=1)
    p2 = x[argmax(y)]
    xlim(0,500)
    return p1,p2



