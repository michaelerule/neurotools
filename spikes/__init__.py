#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Signal processing routines related to spike trains.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

from . import waveform
from . import ppc

import numpy as np
import matplotlib.mlab as ml
from neurotools.stats.density import knn_1d_density

def pp_xcorr(t1,t2,maxlag):
    '''
    Gets all pairwise relative time lags within maxlag. pp_xcorr 
    stands for point-process cross correlation.

    Parameters
    ----------
    t1,t2: 1d arrays
        Lists of time points to compare
    maxlag: number
        maximum time lag to consider, in the same units as t1 and t2
    '''
    t1 = np.sort(t1)
    t2 = np.sort(t2)
    bufferedlag = maxlag*2
    a = 0
    b = 0
    d = []
    for i,t in enumerate(t1):
        ta = t-bufferedlag
        tb = t+bufferedlag
        # smallest matching value is larger than largest value in t2
        if ta>t2[-1]:
            break
        # if the starting point lies within the array
        # increment a until we are at the starting point
        if ta>t2[0]:
            while t2[a]<ta:
                a+=1
        # largest matching value is smaller than smallest in t2
        # then keep incrementing t1 until we are in range
        if tb<t2[0]:
            continue
        # if the upper limit isn't already at the end of the array
        # increment it until it lies outside the matching region
        while b<len(t2) and t2[b]<tb:
            b+=1
        d.extend(t-t2[a:b])
    return d

def txcorr(t1,t2,maxlag,
    k=100,
    normalize=False,
    sampleat =None):
    '''
    Computes cross correlation between two spike trains provided in
    terms of spike times, over a maximum range of lags. Uses nearest-
    neighbor density estimation to provide an adaptively smoothed
    cross-correlation function.

    Parameters
    ----------
    t1,t2: 1d arrays
        Lists of time points to compare
    maxlag: number
        maximum time lag to consider, in the same units as t1 and t2
    k : positive integer
        number of nearest neighbors to use in the density estimation
    normalize : boolean
        Normalize correlation by zero-lag correlation. Default False
    sampleat : int
        time lags to sample for the density estimation
        defaults to spanning +-lags with 1 time-unit bins if none
    '''
    t1 = np.sort(t1)
    t2 = np.sort(t2)
    bufferedlag = maxlag*2
    a = 0
    b = 0
    d = []
    for i,t in enumerate(t1):
        ta = t-bufferedlag
        tb = t+bufferedlag
        # smallest matching value is larger than largest value in t2
        if ta>t2[-1]:
            break
        # if the starting point lies within the array
        # increment a until we are at the starting point
        if ta>t2[0]:
            while t2[a]<ta:
                a+=1
        # largest matching value is smaller than smallest in t2
        # then keep incrementing t1 until we are in range
        if tb<t2[0]:
            continue
        # if the upper limit isn't already at the end of the array
        # increment it until it lies outside the matching region
        while b<len(t2) and t2[b]<tb:
            b+=1
        d.extend(t-t2[a:b])
    a,b = knn_1d_density(d,k=k,eps=0.01)
    if sampleat is None:
        y = np.interp(np.arange(-maxlag,maxlag+1),a,b)
    else:
        y = np.interp(sampleat,a,b)
    if normalize:
        y *= 1./y[maxlag]
    return y

def pack_cross_correlation_matrix(xc):
    assert len(xc.shape)==1
    k = xc.shape[0]
    assert k%2==1
    m = (k-1)//2
    matrix = np.zeros((m+1,)*2,'float')
    for i in range(m+1):
        matrix[i,:] = xc[m-i:m-i+m+1]
    return matrix

def cut_spikes(s,cut):
    '''
    downsampling spike raster by factor cut
    just sums up the bins (can generate counts >1)
    '''
    return np.array([
        np.sum(s[i:i+cut])
        for i in np.arange(0,len(s),cut)])

def times_to_raster(spikes,duration=1000):
    result = np.zeros((1000,),dtype=np.float32)
    if len(spikes)>0:
        result[spikes]=1
    return result

def bin_spikes_raster(train,binsize=5):
    '''
    Important! This accepts a spike raster, not spike times!
    '''
    bins = int(np.ceil(len(train)/float(binsize)))
    return np.histogram(ml.find(train),bins,(0,bins*binsize))[0]

def bin_spike_times(times,binsize=5):
    '''
    Important! This accepts spike times, not a raster
    '''
    bins = np.ceil(np.max(times)/binsize)
    return np.histogram(times,bins,(0,bins*binsize))[0]



