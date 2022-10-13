#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function


from pylab import *
from numpy import *

'''
Routines related to 2D boolean arrays used as image masks
depends on neurotools.spatial.geometry

Collected functions from 2018--2023 concerning analyses of 2D data.
These routines work on 2D (x,y) points encoded as complex z=x+iy numbers.
'''


def as_mask(x):
    '''
    Verify that x is a 2D np.bool array, or attempt to convert it to one if 
    not.
    
    Parameters
    ----------
    x: 2D np.ndarray
        This routine understands np.bool, as well as numeric arrays that contain
        only two distinct values, and use a positive value for True and any
        other value for False.
    
    Other Parameters
    ----------------
    
    Returns
    -------
    '''
    x = np.array(x)
    if not len(x.shape)==2:
        raise ValueError('x should be a 2D array')
    if not x.dtype==np.bool:
        # Try something sensible
        x = nan_to_num(float32(x),0.0)
        values = float32(unique(x))
        if len(values)!=2 or all(values<=0) or all(values>0):
            raise ValueError('x should be np.bool, got %s'%x.dtype)
        x = x>0
    return x
    

def mask_to_points(x):
    '''
    Get locations of all `True` pixels in a 2D boolean array encoded as
    z = column + i*row.
    
    Parameters
    ----------
    x: 2D np.bool
    
    Returns
    -------
    z: np.complex64
    '''
    py,px = where(as_mask(x))
    return px+1j*py


from neurotools.signal import circular_gaussian_smooth_2D
def extend_mask(mask,sigma=2,thr=0.5):
    '''
    Extend 2D image mask by blurring and thresholding.
    Note: this uses circular convolution; Pad accordingly. 
    
    Parameters
    ----------
    mask: 2D np.bool
    sigma: float, default 3; gaussian blur radius
    thr: float, default .5; threshold for new mask
    
    Returns
    -------
    np.boo: smoothed mask
    '''
    mask = float32(as_mask(mask))
    return circular_gaussian_smooth_2D(mask,sigma)>thr
    

def pgrid(W,H=None):
    '''
    Create a (W,H) = (Nrows,Ncols) coordinate grid where each cell
    is z = irow + 1j * icol
    
    Parameters
    ----------
    W: int or 2D np.array
        If integer, the number of columns in the grid.
        if np.array, we will take (W,H) from the arrays shape
    H: int
        number of rows in grid; Defaults to H=W if none
    '''
    try:
        w = int(W)
        h = w if H is None else int(H)
    except TypeError:
        W = np.array(W)
        if not len(shape(W))==2:
            raise ValueError('Cannot create 2D grid for shape %s'%W.shape)
        if not H is None:
            raise ValueError('A 2D array was passed, but H was also given')
        w,h = W.shape
    return arange(w)[:,None] +1j*arange(h)[None,:]


def nan_mask(mask,nanvalue=False,value=None):
    '''
    Create a (W,H) = (Nrows,Ncols) coordinate grid where each cell
    is z = irow + 1j * icol
        
    Parameters
    ----------
    mask: 2D np.bool
    '''
    nanvalue = int(not(not nanvalue))
    if value is None:
        value = [1,0][nanvalue]
    use = float32([[NaN,value],[value,NaN]])[nanvalue]
    return use[int32(as_mask(mask))]


def maskout(x,mask,**kwargs):
    '''
    Set pixels in x where mask is False to NaN
    
    Parameters
    ----------
    x: 2D np.float32
    mask: 2D np.bool
    '''
    return x.reshape(mask.shape)*nan_mask(mask,**kwargs)


from neurotools.tools import find
def trim_mask(mask):
    '''
    Trim empty edges of boolean mask
    
    Parameters
    ----------
    mask: 2D np.bool
    '''
    mask = as_mask(mask)
    a,b = find(any(mask,1))[[0,-1]]
    c,d = find(any(mask,0))[[0,-1]]
    return mask[a:b+1,c:d+1]


def mask_crop(x,mask,fill_nan=True):
    '''
    Set pixels in x where mask is False to NaN,
    and then remove empty rows and columns.
    
    Parameters
    ----------
    x: 2D np.float32
    mask: 2D np.bool
    fill_nan: bool
        Whether to fill "false" values with NaN; default is true
    '''
    if fill_nan:
        x = maskout(x,mask)
    return x[np.any(mask,1),:][:,np.any(mask,0)]




    
    
    