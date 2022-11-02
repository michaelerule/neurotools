#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Collected functions from 2018--2023 concerning analyses of 2D data.

These routines work on 2D (x,y) points encoded as complex z=x+iy numbers.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import neurotools.util.tools as ntools
import neurotools.signal as sig
from scipy.spatial import ConvexHull
import neurotools.spatial.masking

from numpy import *

def p2z(px,py=None):
    '''
    Ensure that a numpy array contains (x,y) points encoded as z = x+iy,
    or convert two arrays (x,y) int z = x+iy format.
    
    Parameters
    ----------
    px: np.array
        x coordinate or points, or, if py is None
         - complex z=x + iy array (in which case this function is a noop)
         - array with on dimension length 2 containing (px,py)
    
    Other Parameters
    ----------------
    py: np.array
        y coordinate of points
    
    Returns
    -------
    np.complex64
    '''
    px = np.array(px)
    # Do nothing if already in the right format
    if np.any(np.iscomplex(px)):
        if not py is None:
            raise ValueError('px is already complex but py is specified')
        return px
    # Interpret px as 2D points if py missing
    px = float32(px)
    if py is None:
        # Try to interpret px as points
        s = px.shape
        if len(s)<=1:
            raise ValueError('px doesn\'t seem to contain 2d points')
        if sum(int32(s)==2):
            raise ValueError('more than one axis of '
            'px.shape=%s is length 2; (x,y) axis is ambiguous'%s)
        xyaxis = which(int32(s)==2)[0]
        x = take_along_axis(px,0,xyaxis)
        y = take_along_axis(px,1,xyaxis)
        return x + 1j*y
    # combine as z = px + i py
    py = np.array(py)
    if not py.shape==px.shape:
        raise ValueError('px and py must have the same shape')
    if np.any(np.iscomplex(py)):
        raise ValueError('Argument py already contains z = x + iy points')
    return np.real(px) + 1j*np.real(py)

# Operator abuse;
@ntools.piper
def z2p(pz):
    '''
    Convert complex points to 2D (x,y) points
    
    Parameters
    ----------
    
    Other Parameters
    ----------------
    
    Returns
    -------
    '''
    pz = np.array(pz)
    if not any(iscomplex(pz)):
        raise ValueError('pz does not seem to contain complex x+iy points')
    return float32([pz.real,pz.imag])


def polar_smooth_contour(z,sigma=2):
    '''
    Smooth the radial and angular components of a closed, circular,
    non-self-intersecting contour `z` in the complex plane. 
    
    To avoid coodinate singularity, `z` should not intersect its own centroid.
    
    Smoothing is accomplished in terms of adjacent samples, and the kernel
    standard deviation has units of samples. See `resample_convex_hull` to
    convert a convex shape with irregular angular sampling to one with regular
    angular sampling for better results.
    
    Parameters
    ----------
    px: 1D complex64 z=x+iy points
    sigma: positive float
    '''
    z = p2z(z)
    c = mean(z)
    z = z - c
    theta = angle(z)
    ct = sig.circular_gaussian_smooth(cos(theta),sigma)
    st = sig.circular_gaussian_smooth(sin(theta),sigma)
    h  = angle((ct+1j*st))
    r  = sig.circular_gaussian_smooth(abs(z)**2,sigma)**0.5
    return r*exp(1j*h) + c



def convex_hull(px,py=None):
    '''
    A wrapper for scipy.spatial.ConvexHull that returns points as z=x+iy
    
    Parameters
    ----------
    
    Other Parameters
    ----------------
    
    Returns
    -------
    '''
    z = p2z(px,py)
    points = z2p(z).T
    hull   = ConvexHull(points)
    verts  = concatenate([hull.vertices,hull.vertices[:1]])
    return points[verts]@[1,1j]


def convex_hull_from_mask(x,Ntheta=None,sigma=None):
    '''
    Extract convex hull containing all pixels in a 2D boolean array that are 
    True. The array x is interpreted as a (rows,cols) matrix where row number 
    is the y coordinate and col number is the x coordinate. 
    
    Parameters
    ----------
    x: 2D np.bool
    
    Other Parameters
    ----------------
    Ntheta: positive int
        If not None, the resulting hull will be resampled at Ntheta uniform
        angular intervals around the centroid
    sigma: positive float
        If not None, resulting hull will be smoothed in polar coordinates
        by a circular Gaussian kernel with standard deviation sigma, where
        sigma is expressed in DEGRESS
    
    Returns
    -------
    z: np.complex64
    '''
    q = convex_hull(neurotools.spatial.masking.mask_to_points(x))
    if not Ntheta is None:
        q = resample_convex_hull(q,Ntheta)
    if not sigma is None:
        sigma = float(sigma)
        if sigma<=0 or sigma>360:
            raise ValueError(('Angular smoothing σ=%f '
            'should be between 0 and 360 degrees')%sigma)
        if Ntheta<30:
            raise ValueError(('Angular smoothing σ=%f degrees specified, '
            'but Ntheta=%d is too few to provide suitable resolution'
            )%(sigma,Ntheta))
        q = polar_smooth_contour(q,sigma/360*Ntheta)
    return q


def resample_convex_hull(z,Ntheta=60):
    '''
    Resample a convex shape at uniform angular intervals around its centroid
    
    Parameters
    ----------
    
    Other Parameters
    ----------------
    Ntheta: positive int; default 60
    
    Returns
    -------
    '''
    if Ntheta<4:
        raise ValueError('# angles to sample should be >4; got %d'%Ntheta)
    
    z = convex_hull(z)
    c = mean(z)
    w = z-c    
    r = abs(w)
    h = angle(w)
    order = argsort(h)
    z,w,r,h = z[order],w[order],r[order],h[order]
    
    angles = linspace(-pi,pi,Ntheta+1)[:-1]
    rpad = concatenate([[r[-1]],r,[r[0]]])
    hpad = concatenate([[h[-1]-2*pi],h,[h[0]+2*pi]])
    r1 = interp(angles,hpad,rpad)
    
    return c + r1*exp(1j*angles)
    
    
def in_hull(z,hull): 
    '''
    Determine if the list of points P lies inside a convex hull
    credit: https://stackoverflow.com/a/52405173/900749
    
    Parameters
    ----------
    z: z=x+iy points to test
    hull: ConvexHull, or points to form one with
    '''
    z = p2z(z)
    s = z.shape
    z = z.ravel()
    if not isinstance(hull,ConvexHull):
        hull = ConvexHull(z2p(hull).T)
    m = hull.equations[:,[1,0]] # half-plane directions
    b = hull.equations[:,-1].T  # half-plane thresholds
    return all(m@z2p(z) <= -b[:,None],0).reshape(*s)


