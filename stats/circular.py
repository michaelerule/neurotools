#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Functions for circular statistics.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import matplotlib.pyplot as mp
import scipy.stats
import numpy as np
import matplotlib as plt

def logpolar_gaussian(frame,doplot=False):
    '''
    Generates major and minor axis and one-sigma ellipse contours for a
    log-polar complex Gaussian. Contours are returned as lists of complex
    numbers.
    
    Parameters
    ----------
    frame : np.array
        Complex-valued polar data to model
    doplot : bool
        Whether to generate a plot of the resulting distribution
    
    Returns
    -------
    axis 1 (np.array)
    axis 2 (np.array)
    1-sigma ellipse (np.array)
    '''
    # set to zero mean phase
    theta    = np.angle(np.mean(frame))
    rephased = frame*np.exp(1j*-theta)
    weights  = np.abs(rephased)
    weights  = weights/np.sum(weights)
    x = np.log(np.abs(rephased))
    y = np.angle(rephased)/4
    # use 2D gaussian approximation
    mx = np.dot(weights,x)
    my = np.dot(weights,y)
    cx = x - mx
    cy = y - my
    correction = np.sum(weights)/(np.sum(weights)**2-np.sum(weights**2))
    cxx = np.dot(weights,cx*cx)*correction
    cxy = np.dot(weights,cx*cy)*correction
    cyy = np.dot(weights,cy*cy)*correction
    cm  = np.array([[cxx,cxy],[cxy,cyy]])
    sm  = scipy.linalg.cholesky(cm)
    w,v = scipy.linalg.eig(cm)
    v   = v[0,:]+1j*v[1,:]
    origin = mx + 1j*my
    w = np.sqrt(w)
    axis1  = origin + v[0]*w[0]*np.linspace(-1,1,100)
    axis2  = origin + v[1]*w[1]*np.linspace(-1,1,100)
    circle = np.exp(1j*linspace(0,2*pi,100))
    circle = p2c(np.dot(sm,[circle.real,circle.imag]))+origin
    phase  = np.exp(1j*theta)
    if doplot:
        plot(*c2p(np.exp(axis1)*phase),color='r',lw=2,zorder=Inf)
        plot(*c2p(np.exp(axis2)*phase),color='r',lw=2,zorder=Inf)
        plot(*c2p(np.exp(circle)*phase),color='r',lw=2,zorder=Inf)
    return np.exp(axis1)*phase,np.exp(axis2)*phase,np.exp(circle)*phase


def complex_gaussian(frame,doplot=False):
    '''
    Generate axis and 1-sigma contour for a complex gaussian 
    distribution
    
    Parameters
    ----------
    frame : np.array
        Complex-valued polar data to model
    doplot : bool
        Whether to generate a plot of the resulting distribution
    
    Returns
    -------
    axis1:
        Path for axis 1, encoded as z=x+iy
    axis2:
        Path for axis 2, encoded as z=x+iy
    circle:
        Path for 1-sigma radius ellipse, encoded as z=x+iy
    '''
    # set to zero mean phase
    rephased = frame#*np.exp(1j*-theta)
    weights = np.ones(np.shape(rephased))
    weights = weights/np.sum(weights)
    # convert to log-polar
    x = real(rephased)
    y = imag(rephased)
    # use 2D gaussian approximation
    mx = np.dot(weights,x)
    my = np.dot(weights,y)
    cx = x - mx
    cy = y - my
    #cm = cov(cx,cy)
    correction = np.sum(weights)/(np.sum(weights)**2-np.sum(weights**2))
    cxx = np.dot(weights,cx*cx)*correction
    cxy = np.dot(weights,cx*cy)*correction
    cyy = np.dot(weights,cy*cy)*correction
    cm  = np.array([[cxx,cxy],[cxy,cyy]])
    sm  = cholesky(cm)
    w,v = eig(cm)
    v = v[0,:]+1j*v[1,:]
    origin = mx + 1j*my
    w = np.sqrt(w)
    axis1  = origin + v[0]*w[0]*linspace(-1,1,100)
    axis2  = origin + v[1]*w[1]*linspace(-1,1,100)
    circle = np.exp(1j*linspace(0,2*pi,100))
    circle = p2c(np.dot(sm,[real(circle),imag(circle)]))+origin
    if doplot:
        plt.plot(*c2p(axis1) ,color='r',lw=2,zorder=Inf)
        plt.plot(*c2p(axis2) ,color='r',lw=2,zorder=Inf)
        plt.plot(*c2p(circle),color='r',lw=2,zorder=Inf)
    return axis1,axis2,circle

def logpolar_stats(frame,doplot=False):
    '''
    Generate summary statistics for a log-polar Gaussian distribution
    
    Parameters
    ----------
    frame : np.array
        Complex-valued polar data to model
    doplot : bool
        Whether to generate a plot of the resulting distribution
    
    Returns
    -------
    circle:
        Path for 1σ ellipse, encoded as z=x+iy.
    arc:
        Path for angular arc, encoded as z=x+iy.
    radial:
        Path for radial line, encoded as z=x+iy.
    '''
    z = np.mean(frame)
    r = np.mean(np.abs(frame))
    rl = np.mean(np.log(np.abs(frame)))
    rs = np.std(np.abs(frame))
    rsl = np.std(np.log(np.abs(frame)))
    w = frame / np.abs(frame)
    x = np.mean(w)
    theta = angle(x)
    #R = np.abs(x)
    R = np.abs(z) / r
    sd = np.sqrt(-2*np.log(R))
    print('R,sd',R,sd)
    cv = 1-R
    s = np.exp(rl)*np.exp(1j*theta)
    arc = np.exp(rl+theta*1j)*np.exp(1j*linspace(-sd,sd,100))
    circle = np.exp(1j*linspace(0,2*pi,100))
    circle = real(circle)*rsl + 1j*imag(circle)*sd
    circle = circle+rl+1j*theta
    circle = np.exp(circle)
    radial = np.array([s*np.exp(-rsl),s*np.exp(rsl)])
    if doplot:
        plot(*c2p(circle),color='m',lw=2)
        plot(*c2p(arc),color='m',lw=2)
        plot(*c2p(radial),color='m',lw=2)
    return circle,arc,radial

def abspolar_stats(frame,doplot=False):
    '''
    Generate summary statistics for a polar Gaussian distribution
    
    Parameters
    ----------
    frame : np.array
        Complex-valued polar data to model
    doplot : bool
        Whether to generate a plot of the resulting distribution
    
    Returns
    -------
    circle:
        Path for 1σ ellipse, encoded as z=x+iy.
    arc:
        Path for angular arc, encoded as z=x+iy.
    radial:
        Path for radial line, encoded as z=x+iy.
    '''
    z    = frame
    phi  = angle(np.mean(z**2))/2
    flip = sign(np.cos(np.angle(z)-phi))
    r    = np.abs(z)*flip
    h    = np.angle(z) + pi*np.int32(flip==-1)
    mr   = np.mean(r)
    sr   = np.std(r)
    mt   = phi
    st   = np.sqrt(-2*np.log(np.abs(np.mean(np.exp(1j*h)))))
    arc    = mr*np.exp(1j*(phi+linspace(-st,st,100)))
    circle = np.exp(1j*linspace(0,2*pi,100))
    circle = (real(circle)*sr+mr)*np.exp(1j*(imag(circle)*st+phi))
    radial = np.array([(mr-sr)*np.exp(1j*phi),(mr+sr)*np.exp(1j*phi)])
    if doplot:
        plt.clf()
        plt.plot(*c2p(circle), color='m',lw=2)
        plt.plot(*c2p(arc   ), color='m',lw=2)
        plt.plot(*c2p(radial), color='m',lw=2)
        plt.scatter(*c2p([mr*np.exp(1j*phi)]),color='k',s=5**2)
    return circle,arc,radial

def squared_first_circular_moment(samples, axis=-1, unbiased=True, dof=None):
    '''
    Compute squared first circular moment
    
    Parameters
    ----------
    samples : np.array
        Complex-valued polar data to model
    axis : int, default=-1
        Axis over which to compute moment
    unbiased : bool, default=True
        Whether to apply a bias correction (small samples can have smaller
        circular variance than expected)
    dof : int, defualts to None
        Optional degrees of freedome correction. If None, then the number
        of samples minus one will be used as the degrees of freedoms
        
    Returns
    -------
    squared_average : float
        Squared first circular moment
    '''
    squared_average = np.abs(np.np.mean(samples,axis=axis))**2
    if unbiased:
        if dof is None:
            if not type(axis) == int:
                dof = np.prod(np.np.array(np.shape(samples))[list(axis)])
            else:
                dof = np.shape(samples)[axis]
        squared_average = (dof*squared_average-1)/(dof-1)
    return squared_average

def fit_vonmises(z):
    '''
    Fit a vonMises distribution using circular moments.
    
    Parameters
    ----------
    samples : np.array
        Complex-valued polar data to model
    
    Returns
    -------
    location:
        von Mises location parameter μ
    theta:
        Sample circular mean of the provided data.
    scale:
        von Mises location parameter κ
    '''
    scipy.stats.distributions.vonmises.a = -numpy.pi
    scipy.stats.distributions.vonmises.b = numpy.pi
    theta    = angle(np.mean(z))
    dephased = z*np.exp(-1j*theta)
    location,_,scale = scipy.stats.distributions.vonmises.fit(np.angle(dephased))
    return location,theta,scale
