#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Routines related to 2D boolean arrays used as image masks
depends on neurotools.spatial.geometry
These routines expect 2D (x,y) points to be encoded as complex z=x+iy numbers.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np

from neurotools.util.array import find
import neurotools.signal as sig


def as_mask(x):
    '''
    Verify that x is a 2D np.bool array, or attempt to convert it to one if 
    not.
    
    Parameters
    ----------
    x: 2D np.array
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
        x = np.nan_to_num(np.float32(x),0.0)
        values = np.float32(unique(x))
        if len(values)!=2 or np.all(values<=0) or np.all(values>0):
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
    py,px = np.where(as_mask(x))
    return px+1j*py


def extend_mask(mask,sigma=2,thr=0.5):
    '''
    Extend 2D image mask by blurring and thresholding.
    Note: this uses circular convolution; Pad accordingly. 
    
    Parameters:
        mask (2D np.bool): Image mask to extend.
        sigma (float, default 3): Gaussian blur radius.
        thr: (float, default .5): Threshold for new mask.
    
    Returns
    -------
    smoothed mask: 2D np.bool
    '''
    mask = np.float32(as_mask(mask))
    return sig.circular_gaussian_smooth_2D(mask,sigma)>thr
    

def pgrid(W,H=None):
    '''
    Create a (W,H) = (Nrows,Ncols) coordinate grid where each cell
    is z = irow + 1j * icol
    
    Parameters
    ----------
    W: int or 2D np.array
        If `int`: the number of columns in the grid.
        if `np.array`: Take (H,W) from the array's shape
    H: int
        Number of rows in grid; Defaults to H=W if H=None.
    '''
    try:
        w = int(W)
        h = w if H is None else int(H)
    except TypeError:
        W = np.array(W)
        if not len(np.shape(W))==2:
            raise ValueError('Cannot create 2D grid for shape %s'%W.shape)
        if not H is None:
            raise ValueError('A 2D array was passed, but H was also given')
        w,h = W.shape
    return np.arange(w)[:,None] +1j*np.arange(h)[None,:]


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
    use = np.float32([[np.NaN,value],[value,np.NaN]])[nanvalue]
    return use[np.int32(as_mask(mask))]


def maskout(x,mask,**kwargs):
    '''
    Set pixels in x where mask is False to NaN
    
    Parameters
    ----------
    x: 2D np.float32
    mask: 2D np.bool
    '''
    return x.reshape(mask.shape)*nan_mask(mask,**kwargs)


def trim_mask(mask):
    '''
    Remove empty edges of boolean mask.
    See `mask_crop(array,mask)` to use a mask to trim 
    another array.
    
    Parameters
    ----------
    mask: 2D np.bool
    '''
    mask = as_mask(mask)
    a,b = find(np.any(mask,1))[[0,-1]]
    c,d = find(np.any(mask,0))[[0,-1]]
    return mask[a:b+1,c:d+1]


def mask_crop(x,mask,fill_nan=True):
    '''
    Set pixels in `x` where `mask` is `False` to `NaN`,
    and then remove empty rows and columns.
    
    See `trim_mask(mask)` to crop out empty rows, columns 
    from a mask.
    
    Parameters
    ----------
    x: 2D np.float32
    mask: 2D np.bool
    fill_nan: bool; default True
        Whether to fill "false" values with NaN
    '''
    if fill_nan:
        x = maskout(x,mask)
    a,b = find(np.any(mask,1))[[0,-1]]
    c,d = find(np.any(mask,0))[[0,-1]]
    return x[a:b+1,c:d+1]


def to_image(x,mask,fill=np.NaN,crop=False):
    '''
    Assign list of values `x` to locations in `mask` that 
    are `True`, in row-major order.
    
    Parameters
    ----------
    x: 1D np.array
    mask: 2D np.bool
    
    Other Parameters
    ----------------
    full: float; default np.NaN
        Fill value for regions outside the mask
    crop: bool; default False
        Whether to remove empty rows/cols of the resulting
        image.
    '''
    mask = as_mask(mask)
    x = np.array(x)
    q = np.full(mask.shape,fill,dtype=x.dtype)
    q[mask] = x
    if crop:
        q = mask_crop(q,mask)
    return q

    
    
    
