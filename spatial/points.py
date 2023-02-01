#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Collected functions from 2018--2023 concerning analyses of 
2D data.

Most of these routines work on 2D (x,y) points encoded as 
complex z=x+iy numbers.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

from numpy import *

def p2c(p):
    '''
    Convert a point in terms of a length-2 iterable into 
    a complex number
    '''
    p = np.array(p)
    if np.any(np.iscomplex(p)): return p
    if not np.any(int32(p.shape)==2):
        raise ValueError('Shape %s not (x,y)'%(p.shape,))
    which = np.where(int32(p.shape)==2)[0][0]
    p = p.transpose(which,
        *sorted(list({*arange(len(p.shape))}-{which})))
    return p[0]+1j*p[1]

def c2p(z):
    ''' 
    Convert complex-valued np.array with (shape) into a 2×(shape)
    np.float32 array.
    
    Parameters
    ----------
    z: np.complex64 or np.complex128
    '''
    z = np.array(z)
    return np.float32([z.real,z.imag])

def to_xypoint(z):
    '''
    Convert possible complex (x,y) point intoformation
    into float32 (x,y) points.
    
    Parameters
    ----------
    z: np.complex64
        Array of (x,y) points encoded as x+iy complex64
        
    Returns
    -------
    np.float32
        (x,y) poiny array with shape 2 × z.shape
    '''
    z = array(z)
    if np.any(iscomplex(z)):
        return complex64([z.real,z.imag])
    # Possibly already a point? 
    z = float32(z)
    if len(z.shape)<=0:
        raise ValueError('This looks like a scalar, not a point')
    if z.shape[0]==1:
        return z
    if np.sum(int32(z.shape)==2)!=1:
        raise ValueError(
            ('Expected exactly one length-2 axis for (x,y)'+
             'points, got shape %s')%(z.shape,))
    which = np.where(int32(z.shape)==2)[0][0]
    other = {*arange(len(z.shape))}-{which}
    return z.transpose(which,*sorted(list(other)))

def closest(point,otherpoints,radius=inf):
    '''
    Find nearest (x,y) point witin a collection of 
    other points, with maximum distance `radius`
    
    Parameters
    ----------
    point: np.float32 with shape 2
        (x,y) point to match
    otherpoints: np.float32 with shape 2×NPOINTS
        List of (x,y) points to compare
    radius: float
        Maximum allowed distance
        
    Returns
    -------
    imatch: int
        index into otherpoints of the match, or None
        if there is no match within radius
    xymatch: np.float32 with shape 2
        (x,y) coordinates of closestmatch
    distance: float
        distance to match, or NaN if no match
    '''
    radius = float(radius)
    if radius<=0:
        raise ValueError('Error, radius should be positive')
    point       = to_xypoint(point)
    otherpoints = to_xypoint(otherpoints)
    if not point.shape==(2,):
        raise ValueError('Expected (x,y) point as 1st argument')
    otherpoints = otherpoints.reshape(2,np.prod(otherpoints.shape[1:]))
    
    distances = norm(point[:,None] - otherpoints,2,0)
    nearest   = argmin(distances)
    distance  = distances[nearest]
    if distance<=radius:
        return nearest, otherpoints[:,nearest], distance
    return None,full(2,NaN,'f'),NaN

def pair_neighbors(z1,z2,radius=inf):
    '''
    
    Parameters
    ----------
    z1: 1D np.complex64 
        List of x+iy points
    z2: 1D np.complex64 
        List of x+iy points
    radius: float, default `inf`
        Maximum connection distance
    
    Returns
    -------
    edges: NPOINTS × 2 np.int32
        Indecies (i,j) into point lists (z1,z2) of pairs
    points: NPOINTS × 2 np.complex64
        x+iy points from (z1,x2) pairs
    delta: NPOINTS np.float32:
        List of distances for each pair
    '''
    radius = float(radius)
    if radius<=0:
        raise ValueError('Radius should be positive')
        
    z1,z2 = p2c(z1),p2c(z2)
    n1,n2 = len(z1),len(z2)
    distance = abs(z1[:,None]-z2[None,:])
    unused1,unused2 = {*arange(n1)},{*arange(n2)}
    paired = set()
    while len(unused1) and len(unused2):
        # useID → pointID
        ix1 = int32(sorted(list(unused1)))
        ix2 = int32(sorted(list(unused2)))
        # useID → z
        zz1 = z1[ix1]
        zz2 = z2[ix2]
        # useID → nearest useID
        D = distance[ix1,:][:,ix2]
        neighbors1to2 = argmin(D,1)
        neighbors2to1 = argmin(D,0)
        # 
        ok1   = arange(len(ix1)) == neighbors2to1[neighbors1to2]
        ok2   = arange(len(ix2)) == neighbors1to2[neighbors2to1]
        e1to2 = {*zip(ix1[ok1],ix2[neighbors1to2[ok1]])}
        e2to1 = {*zip(ix1[neighbors2to1[ok2]],ix2[ok2])}
        assert len(e1to2-e2to1)==0
        used1,used2 = map(set,zip(*e1to2))
        unused1 -= used1
        unused2 -= used2
        paired |= e1to2

    a,b   = int32([*zip(*paired)])
    pairs = np.array([z1[a],z2[b]])
    delta = abs(pairs[0]-pairs[1])
    keep  = delta<radius
    edges = int32([a,b])
    return edges.T[keep], pairs.T[keep], delta[keep]


def paired_distances(z1,z2):
    '''
    Calculate pairwise distances between two sets of 
    (x,y) points encoded as x+iy complex numbers
    
    Parameters
    ----------
    z1: 1D np.complex64 
        List of x+iy points
    z2: 1D np.complex64 
    
    Returns
    -------
    distance: np.float32 with shape z1.shape+z2.shape
        Array of paired distances
    '''
    z1,z2 = p2c(z1),p2c(z2)
    s1,s2 = z1.shape,z2.shape
    z1,z2 = z1.ravel(),z2.ravel()
    distance = abs(z1[:,None]-z2[None,:])
    return distance.reshape(*(s1+s2))
    
    

def assign_to_regions(regions,points):
    '''
    Assign 2D `points` to Voronoi `regions`, returning the
    indecies into `regions` for each point in `points`
    
    Parameters
    ----------
    regions: 2 × NREGIONS np.float32
        (x,y) coordinates of Vornoi region centers
    points: 2 × NPOINTS np.float32
        (x,y) points to assign to regions
        
    Returns
    -------
    indecies: NPOINTS np.int32
        Index into `regions` for each point    
    '''
    regions = p2c(regions)
    points  = p2c(points)
    return np.argmin(abs(regions[:,None]-points[None,:]),0)


def collect_in_regions(
    regions,
    points,
    return_indecies=False):
    '''
    Bin points to Voronoi regions, returning a list of
    (x,y) points in each region.
    
    Parameters
    ----------
    regions: 2×NREGIONS np.float32
        (x,y) coordinates of Vornoi region centers
    points: 2×NPOINTS np.float32
        (x,y) points to assign to regions
        
    Other Parameters
    ----------------
    return_indecies: boolean; default False
        Whether to additionally return the index
        into the `NPOINTS` assigned to each region
    
    Returns
    -------
    allocated: list
        Length `NREGIONS` list of 2 × NPOINTSINREGION
        np.float32 arrays
    indecies: list
        **Returned only if `return_indecies=True`;**
        Length `NREGIONS` list of NPOINTSINREGION
        np.int32 arrays, each containing the indecies 
        into the length-`NPOINTS` array assigned to
        each region in `regions.
    '''
    i = assign_to_regions(regions,points)
    assigned = [
        points[:,i==j] 
        for j in range(regions.shape[1])]
    if return_indecies:
        indecies = [
            np.where(i==j)[0]
            for j in range(regions.shape[1])]
        return assigned, indecies
    else:
        return assigned


import warnings
def gaussian_from_points(points):
    '''
    Get the mean and (sample) covariance of a point cloud. 
    
    Note: we suppress warnings here because returning NaNs
    for an empty point set is entirely reasonable, and 
    simplifies code elsewhere. 
    
    Parameters
    ----------
    points: NDIM × NPOINTS np.float32
        Point data.
        
    Returns
    -------
    μ: NDIM np.float32
        Mean
    Σ: NDIM × NDIM np.float32
        Sample covariance
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', r'invalid value encountered in divide')
        warnings.filterwarnings(
            'ignore', r'Mean of empty slice.')
        points = np.float32(points)
        ndim, npoints = points.shape
        μ = np.mean(points,1)
        Δ = points - μ[:,None]
        Σ = (Δ@Δ.T)/npoints
        return μ,Σ


def collect_gaussians(regions, s):
    '''
    Return the mean and covariance of points within 
    each Voronoi region. 
    
    Returns
    -------
    means: np.float32
    sigmas: np.float32
    '''
    means,sigmas = zip(*[*map(
        gaussian_from_points,
        collect_in_regions(regions, s)
    )])
    # Collect all ellipses to plot 
    ellipses = []
    _means,_sigmas = [],[] # μ,Σ, dropping ones with NaN
    
    means,sigmas = zip(*[
        (mu,sigma) 
        for (mu,sigma) in zip(means,sigmas)
        if np.all(np.isfinite(sigma)) and np.all(np.isfinite(mu))
    ])
    
    return np.float32(means), np.float32(sigmas)





















