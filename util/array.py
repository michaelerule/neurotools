#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Helper functions related to Numpy arrays and other indexing tasks.
"""
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np

def lmap(function,*args):
    '''
    Map, collecting results immediately in a list.
    Also aliased as `lap`
    '''
    return list(map(function,*args))

def slist(x):
    '''
    Convert iterable to sorted list.
    Also aliased as `sls`.
    '''
    return sorted([*x])

def amap(function,*args):
    '''
    Map, collecting results immediately in a Numpy array.
    This will attempt to create a numeric or boolean array first,
    and fall-back to an object array if a `ValueError` is encountered.
    Also aliased as `aap`.
    '''
    a = lmap(function,*args)
    try:
        return np.array(a)
    except ValueError:
        b = np.empty(len(a), object)    
        b[:] = a
        return b

lap = lmap
aap = amap
sls = slist

def arraymap(f,*iterables,**kwargs):
    '''
    Map functionality over numpy arrays
    replaces depricated arraymap from pylab.
    '''
    depth = kwargs['depth'] if 'depth' in kwargs else 0
    if depth<=1:
        return np.array([f(*list(map(np.array,args))) for args in zip(*iterables)])
    kwargs['depth']=depth-1
    def fun(*args):
        return arraymap(f,*args,**kwargs)
    return arraymap(fun,*iterables,**{'depth':0})

def find(x):
    '''
    Replacement to Pylab's lost `find()` function.
    Synonym for `np.where(np.array(x).ravel())[0]`
    '''
    return np.where(np.array(x).ravel())[0]
    
def ezip(*args):
    return enumerate(zip(*args))

def asiterable(x):
    '''
    Attempt to convert an iterable object to a list
    '''
    try: 
        return list(iter(x))
    except TypeError:
        return None
    
def invert_permutation(p):
    '''
    Invert a a permutation
    
    Parameters
    ----------
    x: list of ints
        Permutation to invert
    '''
    p = np.int32(p)
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s
    
def ndargmax(x):
    '''
    Get coordinates of largest value in a multidimensional 
    array
    
    Parameters
    ----------
    x: np.array
    '''
    x = np.array(x)
    return np.unravel_index(np.nanargmax(x),x.shape)


def complex_to_nan(x,value=np.NaN):
    '''
    Replce complex entries with NaN or other value
    
    Parameters
    ----------
    x: np.array
    
    Other Parameters
    ----------------
    value: float; default `np.NaN`
        Value to replace complex entries with
    '''
    x = np.array(x)
    x[np.iscomplex(x)]=value
    return x.real


def make_rebroadcast_slice(x,axis=0,verbose=False):
    '''
    Generate correct slice object for broadcasting 
    stastistics averaged over the given axis back to the
    original shape.
    
    Parameters
    ----------
    x: np.array
    '''
    x = np.array(x)
    naxes = len(np.shape(x))
    if verbose:
        print('x.shape=',np.shape(x))
        print('naxes=',naxes)
    if axis<0:
        axis=naxes+axis
    if axis==0:
        theslice = (None,Ellipsis)
    elif axis==naxes-1:
        theslice = (Ellipsis,None)
    else:
        a = axis
        b = naxes - a - 1
        theslice = (np.s_[:],)*a + (None,) + (np.s_[:],)*b
    if verbose:
        print('axis=',axis)
    return theslice


def deep_tuple(x):
    '''
    Convert x to tuple, deeply.
    Defaults to the identity function if x is not iterable
    '''
    if type(x)==str:
        return x
    try:
        result = tuple(deep_tuple(i) for i in x)
        if len(result)==1: result = result[0]
        return tuple(result)
    except TypeError:
        return x
    assert 0


def deep_map(f,tree):
    '''
    Maps over a tree like structure
    '''
    if hasattr(tree, '__iter__') and not type(tree) is str:
        return tuple([deep_map(f,t) for t in tree])
    else:
        return f(tree)


def to_indices(x):
    '''
    There are two ways to extract a subset from numpy arrays:
    1. providing a boolean array of the same shape
    2. providing a list of indecies
    
    This function is designed to accept either, and return a list 
    of indecies.
    '''
    x = np.array(x)
    if x.dtype==np.dtype('bool'):
        # typed as a boolean, convert this to indicies
        return deep_tuple(np.where(x))
    # Array is not boolean; 
    # Several possibilities
    # It could already be a list of indecies
    # OR it could be boolean data encoded in another numeric type
    symbols = np.unique(x.ravel())
    bool_like = np.all((symbols==1)|(symbols==0))
    if bool_like:
        if len(symbols)<2:
            warnings.warn('Indexing array looks boolean, but contains only the value %s?'%symbols[0])
        return deep_tuple(np.where(x!=0))
    if np.all((symbols==-1)|(symbols==-2)):
        warnings.warn('Numeric array for indexing contains only (-2,-1); Was ~ applied to an int32 array? Assuming -1=True')
        x = (x==-1)
        return deep_tuple(np.where(x))
    if np.all(np.int32(x)==x):
        # Seems like it is already integers?
        return deep_tuple(x)
    raise ValueError('Indexing array is numeric, but contains non-integer numbers?')
        

def onehot(ids):
    '''
    Generate so-called "one-hot"
    representation of class labels from 
    a vector of class identities
    
    Returns
    -------
    labels:
        labels corresponding to each index
    r:
        One-hot label format
    '''
    ids      = np.array(ids)
    labels   = sorted(list(set(ids)))
    nsamples = len(ids)
    nlabels  = len(labels)
    r = np.zeros((nlabels,nsamples))
    for i,l in enumerate(labels):
        r[i,ids==l] = 1
    return labels,r


def _take_axis_slice(shape,axis,index):
    # Redundant to existing numpy functions TODO remove
    ndims = len(shape)
    if axis<0 or axis>=ndims:
        raise ValueError('axis %d invalid for shape %s'%(axis,shape))
    before = axis
    after  = ndims-1-axis
    return (np.s_[:],)*before + (index,) + (np.s_[:],)*after

def _take_axis(x,axis,index):
    # Redundant to existing numpy functions TODO remove
    return x[_take_axis_slice(x.shape,axis,index)]
