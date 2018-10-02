#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import neurotools.getfftw

from scipy.signal         import convolve2d
from neurotools.functions import npdf

def reflect2D(data):
    '''
    Reflects 2D data ... used in the discrete cosine transform.
    data may have dimensions HxW or HxWxN
    return 2Hx2W or 2Hx2WxN respectively
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
    Reflects 2D data, without doubling the data on the edge
    data may have dimensions HxW or HxWxN
    return 2H-2x2W-2 or 2H-2x2W-2xN respectively
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
    h,w = np.shape(x)
    mirrored = np.zeros((h*2,w*2),dtype=x.dtype)
    mirrored[:h,:w]=x
    mirrored[h:,:w]=np.flipud(x)
    mirrored[: ,w:]=fliplr(mirrored[:,:w])
    return mirrored

def convolve2dct(x,k):
    h,w = np.shape(x)
    x = mirror2d(x)
    x = convolve2d(x,k,'same')
    return x[:h,:w]

def separable2d(X,k,k2=None):
    h,w = np.shape(X)
    X = mirror2d(X)
    y = array([convolve(x,k,'same') for x in X])
    if k2==None: k2=k
    y = array([convolve(x,k2,'same') for x in y.T]).T
    return y[:h,:w]

def gausskern2d(sigma,size):
    k = size/2
    x = float32(arange(-k,k+1))
    p = npdf(0,sigma,x)
    kern = outer(p,p)
    return kern / sum(kern)

def gausskern1d(sigma,size):
    k = size/2
    x = float32(arange(-k,k+1))
    kern = npdf(0,sigma,x)
    return kern / sum(kern)

