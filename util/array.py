#!/usr/bin/python3
# -*- coding: UTF-8 -*-
"""
Helper functions related to Numpy arrays and other indexing
tasks.
"""

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
    This will try to create a numeric or boolean array 
    first, and fall-back to an object array if a 
    `ValueError` is encountered. Also aliased as `aap`.
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
        return np.array([f(*list(map(np.array,args))) \
        for args in zip(*iterables)])
    kwargs['depth']=depth-1
    def fun(*args):
        return arraymap(f,*args,**kwargs)
    return arraymap(fun,*iterables,**{'depth':0})

def find(x):
    '''
    Replacement to Pylab's lost `find()` function.
    Synonym for `np.where(np.array(x).ravel())[0]`
    
    Parameters
    ----------
    x: np.array
    '''
    return np.where(np.array(x).ravel())[0]
    
def ezip(*args):
    '''
    Enumerate and zip, i.e. `enumerate(zip(*args))`
    '''
    return enumerate(zip(*args))

def asiterable(x):
    '''
    Attempt to convert an iterable object to a list.
    This mat eventually be replaced with something fancier,
    but for now just calls `list(iter(x))`.
    
    Parameters
    ----------
    x: iterable
    
    Returns
    -------
    : list
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


def complex_to_nan(x,value=np.nan):
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

def verify_axes(x,axis):
    # Get shape of the array
    x      = np.array(x)
    xshape = np.shape(x)
    naxes  = len(xshape)
    # Convert axis argument into an int32 array      
    if isinstance(axis,int): axis = np.int32([axis])
    else: axis = np.int32([*axis])
    # Check is in range    
    max_axis = naxes-1
    min_axis = -naxes
    if np.any((axis>max_axis)|(axis<min_axis)):
        raise ValueError(
            'An axis %s is out of range for shape %s'%(axis,xshape)
        )
    if len({*axis})<len(axis):
        raise ValueError(
            'axes %s contains duplicates'%(axis,)
        )
    # Convert negative axis index to positive        
    axis = axis % naxes
    return axis

def axes_complement(x,axes):
    '''
    Set of all axes indeces for ``x`` not contained in ``axes``.
    '''
    # Get shape of the array
    x      = np.array(x)
    xshape = np.shape(x)
    naxes  = len(xshape)
    axes   = verify_axes(x,axes)
    other_axes = {*range(naxes)} - {*axes}
    return tuple(list(other_axes))

def reslice(naxes,expand_into):
    '''
    Generate a slice object to expand an array along
    ``expand_into`` to broadcast with an array with ``naxes``
    dimenstions.
    '''
    return tuple([
        None if a in expand_into else np.s_[:] 
        for a in range(naxes)])

def make_rebroadcast_slice(x,axis=0,verbose=False):
    '''
    Generate correct slice object for broadcasting 
    stastistics averaged over the given axis(es) back to the
    original shape.
    
    Parameters
    ----------
    x: np.array
    
    Other Parameters
    ----------------
    axis: int or tuple; default 0
    verbose: boolean; default False
    '''
    
    # Get the actual shape of the array
    x      = np.array(x)
    xshape = np.shape(x)
    naxes  = len(xshape)

    axis = verify_axes(x,axis)

    # Broadcast None over the axes that were collapsed
    theslice = reslice(naxes,axis)
    
    if verbose:
        print('x.shape=',np.shape(x))
        print('naxes=',naxes)
        print('axis=',axis)
        print('slice=',theslice)
    return theslice


def deep_tuple(x):
    '''
    Convert x to tuple, deeply.
    Defaults to the identity function if x is not iterable
    
    Parameters
    ----------
    x: nested iterable
    
    Returns
    ----------
    x: nested iterable
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
    
    Parameters
    ----------
    f: function
    tree: nested iterable
    
    Returns
    -------
    : nested iterable
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
    
    This function is designed to accept either, and return 
    a list of indecies.
    
    Parameters
    ----------
    x: np.array
    '''
    x = np.array(x)
    if x.dtype==np.dtype('bool'):
        # typed as a boolean, convert this to indicies
        return deep_tuple(np.where(x))
    # Array is not boolean:
    # It could already be a list of indecies
    # OR it could be boolean data encoded in another numeric type
    symbols = np.unique(x.ravel())
    bool_like = np.all((symbols==1)|(symbols==0))
    if bool_like:
        if len(symbols)<2:
            warnings.warn(
                'Indexing array looks boolean, but '
                'contains only the value %s?'%symbols[0])
        return deep_tuple(np.where(x!=0))
    if np.all((symbols==-1)|(symbols==-2)):
        warnings.warn(
            'Numeric array for indexing contains only '
            '(-2,-1); Was ~ applied to an int32 array? '
            'Assuming -1=True')
        x = (x==-1)
        return deep_tuple(np.where(x))
    if np.all(np.int32(x)==x):
        # Seems like it is already integers?
        return deep_tuple(x)
    raise ValueError(
        'Indexing array is numeric, but '
        'contains non-integer numbers?')
        

def onehot(ids,n=None,dense=False):
    '''
    Generate "one-hot" representation from integer class labels.
    
    Parameters
    ----------
    ids: np.array
    
    Other Parameters
    ----------------
    n: int or None; default None
        Total number of labels. If None, will default
        to the largest value in `ids`.
    dense: boolean; default False
        Whether to create missing labels. 
        Ignored if the parameter `n` is specified.
    
    Returns
    -------
    labels:
        labels corresponding to each index
    r:
        One-hot label format
    '''
    ids = np.array(ids)
    if ids.shape==tuple():
      ids = np.array([ids])
    labels = sorted(list(set(ids)))
    
    if (n is not None) or dense:
        if not np.allclose(labels,np.int32(labels)):
            raise ValueError(
            'The options `dense` and `n` only supported for '
            '32-bit integer labels. The content of `ids` '
            'doesn\'t look like 32-bit integers.')
        labels = np.int32(labels)
        ids    = np.int32(ids)
        maxid  = np.max(ids)
        if n is None:
            n = maxid+1
        if maxid>=n:
            raise ValueError((
                'The parameter `n` was set to %d, but '
                '`ids` contained the value %d; Ids must '
                'be <= `n`.')%(n,maxid))
        labels = np.arange(n)
        
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
    
def zeroslike(x):
    '''
    Create numpy array of zeros the same shape and type as x
    
    Parameters
    ----------
    x: np.array
    '''
    return np.zeros(x.shape,dtype=x.dtype)

def oneslike(x):
    '''
    Create numpy array of ones the same shape and type as x
    
    Parameters
    ----------
    x: np.array
    '''
    return np.ones(x.shape,dtype=x.dtype)

def split_into_groups(x,group_sizes):
    '''
    Split `np.array` `x` into `len(group_sizes)` groups,
    with the size of the groups specified by `group_sizes`.
    
    This operates along the last axis of `x`
    
    Parameters
    ----------
    x: np.array
        Numpy array to split; Last axis should have the
        same length as `sum(group_sizes)`
    group_sizes: iterable of positive ints
        Group sizes
        
    Returns
    -------
    list
        List of sub-arrays for each group
    '''
    x = np.array(x)
    g = np.int32([*group_sizes])
    if np.any(g<=0): 
        raise ValueError(
            'Group sizes should be positive, got %s'%g)
    if x.shape[-1]!=sum(g):
        raise ValueError(
            'Length of last axis ov `x` shoud match sum of '
            'group sizes, got %s and %s'%(x.shape,g))

    ngroups = len(g)
    edges = np.cumsum(np.concatenate([[0],g]))
        
    return [
        x[...,edges[i]:edges[i+1]] for i in range(ngroups)
    ]


def maybe_integer(x):
    '''
    Cast a float numpy array to int32 or int64, if it would 
    not result in loss of precision. 
    
    Parameters
    ----------
    x: np.array
    
    Returns
    -------
    x: np.array
        np.int32 or np.int64 if possible, otherwise the
        original value of x.  
    '''
    x = np.array(x)
    y = np.int32(x)
    if np.allclose(x,y): return y
    y = np.int64(x)
    if np.allclose(x,y): return y
    return x


def widths_to_edges(widths,startat=0):
    '''
    Convert a list of widths into a list of edges
    delimiting consecutive bands of the given width
    
    Parameters
    ----------
    widths: list of numbers
        Width of each band
        
    Other Parameters
    ----------------
    startat: number, default 0
        Starting position of bands
        
    Returns
    -------
    edges: 1D np.array
    '''
    widths = np.array(widths)
    e = np.cumsum(np.concatenate([[startat],widths]))
    if widths.dtype==np.int32: e=maybe_integer(e)
    return e


def widths_to_limits(widths,startat=0):
    '''
    Convert a list of integer widths into a list of
    [a,b) indecies delimiting the concatenated width 
    
    Parameters
    ----------
    widths: list of integers
        Width of each band
    startat: int; default 0
        Starting index
        
    Returns
    -------
    limits: N×2 np.int32
        List of [start,stop) indecies
    '''
    w = np.array(widths)
    q = np.int32(w)
    r = int(startat)
    if not np.allclose(w,q):
        raise ValueError('Widths should be int32')
    e = np.cumsum(np.concatenate([[r],q]))
    return np.int32([e[:-1],e[1:]]).T    
    

def centers(edges):
    '''
    Get center of histogram bins given as a list of edges.
    
    Parameters
    ----------
    edges: list of numbers
        Edges of histogram bins
        
    Returns
    -------
    centers: 1D np.array
        Center of histogram bins
    '''
    edges = np.array(edges)
    e = np.float32(edges)
    c = (e[1:]+e[:-1])*0.5
    if edges.dtype==np.int32: c=maybe_integer(c)
    return c


def widths_to_centers(widths,startat=0):
    '''
    Get centers of a consecutive collection of histogram
    widths. 
    
    Parameters
    ----------
    widths: list of numbers
        Width of each band
        
    Other Parameters
    ----------------
    startat: number, default 0
        Starting position of bands
    '''
    edges = widths_to_edges(widths,startat=startat)
    return centers(edges)

def extract(zerod):
    '''
    Extract the scalar value from zero-dimensional 
    ``np.ndarray``.
    '''
    return np.array(zerod).reshape(1)[0]


def binspace(start,stop,nbins,eps=1e-9):
    b = np.linspace(0,100,nbins+1)
    b[0] -=eps
    b[-1]+=eps
    return b


def binto(x,y,start,stop,nbins=50):
    b = binspace(0,100,nbins)
    i = np.digitize(x,b)
    return [y[i==j] for j in range(nbins)]


def remove_nonincreasing(x,y):
    keep = np.where(np.diff(y)>=0)[0]
    return x[keep], y[keep]


def remove_nans(*args):
    args = [*map(np.array,args)]
    if len(args)==1 and len(np.shape(args[0]))==2:
        return np.array(remove_nans(*args[0]))
    else:
        ok = np.all([np.isfinite(a) for a in args],axis=0)
        if len(args)>1:
            return [np.array(a)[ok] for a in args]
        return np.array(args[0])[ok]


def allclose_recursive(a,b):
    '''
    Version of `np.allclose` that recurses through a tuple/
    list structure until it reaches an `np.ndarray` to compare
    via `np.allclose` as a base-case.
    '''
    if isinstance(a,np.ndarray) and isinstance(b,np.ndarray):
        return np.allclose(a,b)
    return all([allclose_recursive(ai,bi) \
                for (ai,bi) in zip(a,b)])





