#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Helper routines for convolutions, mostly related to padding.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
from scipy.signal import convolve2d
from neurotools.util.functions import npdf

def reflect2D(data):
    '''
    Reflects 2D data for use with the discrete cosine 
    transform.
    
    Parameters
    ----------
    data: np.array
        `data` may have dimensions (H,W) or (H,W,N)
        
    Returns
    -------
    result: np.array
        shape (2H,2W) array if `data` is 2D. 
        shaoe (2H,2W,N) array if `data` is 3D.
    '''
    h,w = np.shape(data)[:2]
    dtype = data.dtype
    if len(np.shape(data))==2:
        result = np.zeros((h*2,w*2),dtype=dtype)
    else:
        #assert len(np.shape(data))==3
        h,w = np.shape(data)[:2]
        result = np.zeros((h*2,w*2)+np.shape(data)[2:],dtype=dtype)
    result[:h,:w,...]=data
    result[h:,:w,...]=np.flipud(data)
    result[ :,w:,...]=result[:,w-1::-1,...]
    return result


def reflect2D_1(data):
    '''
    Reflects 2D data, without doubling the data on the edge.
    
    Parameters
    ----------
    data: np.array
        `data` may have dimensions (H,W) or (H,W,N)
        
    Returns
    -------
    result: np.array
        shape (2H-2,2W-2) array if `data` is 2D. 
        shaoe (2H-2,2W-2,N) array if `data` is 3D.
    '''
    h,w = np.shape(data)[:2]
    dtype = data.dtype
    if len(np.shape(data))==2:
        result = np.zeros((h*2-2,w*2-2),dtype=dtype)
    else:
        h,w = np.shape(data)[:2]
        result = np.zeros((h*2-2,w*2-2)+np.shape(data)[2:],dtype=dtype)
    # top left corner is easy: just a copy of the data
    result[:h,:w,...]=data
    # next do the bottom left. the bottom row gets duplicated unless
    # we trim it off
    result[h:,:w,...]=np.flipud(data[:-1,:])[:-1,:]
    # then, copy over what we just did. dont copy the last column (which
    # becomes the first column when flipped)
    result[ :,w:,...]=result[:,w-2:0:-1,...]
    return result

def mirror2d(x):
    '''
    Mirror-pad a 2D signal to implement reflected boundary
    conditions for 2D convolution.
    
    This function is obsolete and superseded by 
    `reflect2D()`. 
    
    Parameters
    ----------
    X: 2D np.array
        Signal to pad
        
    Returns
    -------
    '''
    h,w = np.shape(x)
    mirrored = np.zeros((h*2,w*2),dtype=x.dtype)
    mirrored[:h,:w]=x
    mirrored[h:,:w]=np.flipud(x)
    mirrored[: ,w:]=fliplr(mirrored[:,:w])
    return mirrored

def convolve2dct(x,k):
    '''
    
    Parameters
    ----------
        
    Returns
    -------
    '''
    h,w = np.shape(x)
    x = mirror2d(x)
    x = convolve2d(x,k,'same')
    return x[:h,:w]

def separable2d(X,k,k2=None):
    '''
    Convolve 2D signal `X` with two one-dimensional
    convolutions with kernel `k`.
    
    This uses reflected boundary padding
    
    Parameters
    ----------
    X: 2D np.array
        Signal to convolve
    k: 1D np.array
        Convolution kernel
        
    Other Parameters
    ----------------
    k2: 1D np.array
        Convolution kernel for the section array 
        dimension, if `X` is not square or if different
        horizontal and vertical kernels are desired.
        
    Returns
    -------
    result: 2D np.array
        Convolved result
    '''
    h,w = np.shape(X)
    X = mirror2d(X)
    y = array([convolve(x,k,'same') for x in X])
    if k2==None: k2=k
    y = array([convolve(x,k2,'same') for x in y.T]).T
    return y[:h,:w]

def gausskern2d(sigma,size):
    '''
    Generate 2D Gaussian kernel
    
    Parameters
    ----------
    sigma: positive float
        Kernel standard deviation
    size: positive int
        Size of kernel to generate
        
    Returns
    -------
    kernel: 2D np.float32
        Gaussian kernel
    '''
    k = size/2
    x = float32(arange(-k,k+1))
    p = npdf(0,sigma,x)
    kern = outer(p,p)
    return np.float32(kern / np.sum(kern))

def gausskern1d(sigma,size):
    '''
    Generate 1D Gaussian kernel
    
    Parameters
    ----------
    sigma: positive float
        Kernel standard deviation
    size: positive int
        Size of kernel to generate
        
    Returns
    -------
    kernel: 1D np.float32
        Gaussian kernel
    '''
    k = size/2
    x = float32(arange(-k,k+1))
    kern = npdf(0,sigma,x)
    return np.float32(kern / sum(kern))

