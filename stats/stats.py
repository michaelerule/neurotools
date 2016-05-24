#!/usr/bin/python
# -*- coding: UTF-8 -*-
# The above two lines should appear in all python source files!
# It is good practice to include the lines below
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division

"""
Routines for computing commonly used summary statistics not otherwise
available in pylab

"""

import neurotools.stats.modefind as modefind
import numpy as np
from numpy import *

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average  = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, math.sqrt(variance))

def crossvalidated_least_squares(a,b,K,regress=np.linalg.lstsq):
    '''
    predicts B from A in K-fold cross-validated blocks using linear
    least squares
    returns 
        model coefficients x
        predicted values of b under crossvalidation
        correlation coefficient
        root mean squared error    
    '''
    N = len(b)
    B = N/K
    x = {}
    predict = []
    for k in range(K):
        start = k*B
        stop  = start+B
        #if stop>N: stop = N
        if k>=K-1: stop = N
        trainB = append(b[:start  ],b[stop:  ])
        trainA = append(a[:start,:],a[stop:,:],axis=0)
        testB  = b[start:stop]
        testA  = a[start:stop,:]
        x[k] = regress(trainA,trainB)[0]
        reconstructed = dot(testA,x[k])
        error = np.mean((reconstructed-testB)**2)
        predict.extend(reconstructed)
    cc  = pearsonr(b,predict)[0]
    rms = sqrt(mean((array(b)-array(predict))**2))
    return x,predict,cc,rms

def print_stats(g,name='',prefix=''):
    '''
    computes, prints, and returns
    mode
    mean
    median
    '''
    mode = modefind.modefind(g,0)
    mn   = np.mean(g)
    md   = np.median(g)
    print prefix,'mode    %s\t%0.4f'%(name,mode)
    print prefix,'mean    %s\t%0.4f'%(name,mn)
    print prefix,'median  %s\t%0.4f'%(name,md)
    return mode,mn,md

def squared_first_circular_moment(samples, axis=-1, unbiased=True, dof=None):
    '''
    Computes the squared first circular moment R² of complex data.
    '''
    squared_average = abs(np.mean(samples,axis=axis))**2
    if unbiased:
        if dof is None:
            if not type(axis) == int:
                dof = np.prod(np.array(shape(samples))[list(axis)])
            else:
                dof = shape(samples)[axis]
        squared_average = (dof*squared_average-1)/(dof-1)
    return squared_average





