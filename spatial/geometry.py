#!/usr/bin/python3
# -*- coding: UTF-8 -*-
'''
Collected functions from 2018--2023 
concerning analyses of 2D data.

These routines work on 2D (x,y) points encoded as 
complex z=x+iy numbers.
'''
import neurotools.util.tools as ntools
import neurotools.signal as sig
import neurotools.spatial.masking
import numpy as np
from scipy.spatial import ConvexHull

def p2z(px,py=None):
    '''
    Ensure that a numpy array contains (x,y) points 
    encoded as z = x+iy,
    or convert two arrays (x,y) int z = x+iy format.
    
    Parameters
    ----------
    px: np.array
        x coordinate or points, or, if py is None
         - complex z=x + iy array (in which case this 
           function is a noop)
         - array with on dimension length 2 
           containing (px,py)
    
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
            raise ValueError(
            'px is already complex but py is specified')
        return px
    # Interpret px as 2D points if py missing
    px = np.float32(px)
    if py is None:
        # Try to interpret px as points
        s = px.shape
        if len(s)<=1:
            raise ValueError(
            'px doesn\'t seem to contain 2d points')
        if np.sum(np.int32(s)==2)>1:
            raise ValueError(('more than one axis of '
            'px.shape=%s is length 2; (x,y) axis is '
            'ambiguous')%(s,))
        xyaxis = np.where(np.int32(s)==2)[0][0]
        ndims = len(np.shape(px))
        slices = [slice(None,None,None) for i in range(ndims)]
        slices[xyaxis] = slice(0,1,1)
        x = px[tuple(slices)]
        slices[xyaxis] = slice(1,2,1)
        y = px[tuple(slices)]
        return x + 1j*y
    # combine as z = px + i py
    py = np.array(py)
    if not py.shape==px.shape:
        raise ValueError(
        'px and py must have the same shape')
    if np.any(np.iscomplex(py)):
        raise ValueError(
        'Argument py already contains z = x + iy points')
    return np.real(px) + 1j*np.real(py)


def z2p(pz):
    '''
    Convert complex points to 2D (x,y) points
    
    Parameters
    ----------
    ps: np.complex64
    
    Returns
    -------
    :np.float32
    '''
    pz = np.array(pz)
    if not np.any(np.iscomplex(pz)):
        raise ValueError(
        'pz does not seem to contain complex x+iy points')
    return np.float32([pz.real,pz.imag])


def polar_smooth_contour(z,sigma=2):
    '''
    Smooth the radial and angular components of a closed, 
    circular, non-self-intersecting contour `z` in the 
    complex plane. 
    
    To avoid coodinate singularity, `z` should not 
    intersect its own centroid.
    
    Smoothing is accomplished in terms of adjacent samples, 
    and the kernel standard deviation has units of samples. 
    See `resample_convex_hull` to convert a convex shape 
    with irregular angular sampling to one with regular
    angular sampling for better results.
    
    Parameters
    ----------
    px: 1D complex64 z=x+iy points
    sigma: positive float
    '''
    z = p2z(z)
    c = np.mean(z)
    z = z - c
    theta = np.angle(z)
    ct = sig.circular_gaussian_smooth(np.cos(theta),sigma)
    st = sig.circular_gaussian_smooth(np.sin(theta),sigma)
    h  = np.angle((ct+1j*st))
    r  = sig.circular_gaussian_smooth(np.abs(z)**2,sigma)**0.5
    return r*np.exp(1j*h) + c


def convex_hull(px,py=None):
    '''
    A wrapper for scipy.spatial.ConvexHull that returns 
    points as z=x+iy.
    
    Parameters
    ----------
    px:
    py:
    
    Returns
    -------
    '''
    z = p2z(px,py)
    points = z2p(z).T
    hull   = ConvexHull(points)
    verts  = np.concatenate(
        [hull.vertices,hull.vertices[:1]])
    return points[verts]@[1,1j]


def convex_hull_from_mask(
    x,
    Ntheta=None,
    sigma=None,
    close=True):
    '''
    Extract convex hull containing all pixels in a 2D 
    boolean array that are `True`. The array `x` is 
    interpreted as a (rows,cols) matrix where row number 
    is the `y` coordinate and col number is the `x` 
    coordinate. 
    
    Parameters
    ----------
    x: 2D np.bool
    
    Other Parameters
    ----------------
    Ntheta: positive int
        If not None, the resulting hull will be resampled 
        at `Ntheta` uniform angular intervals around the 
        centroid.
    sigma: positive float
        If not None, resulting hull will be smoothed in
        polar coordinates
        by a circular Gaussian kernel with standard 
        deviation `sigma` (in DEGRESS).
    close: boolean; default True
        Whenter to repeat the first point in the convext
        hull at the end so that it can be plotted directly
        as a closed contour. 
    
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
            raise ValueError(
            ('Angular smoothing σ=%f degrees specified, '
            'but Ntheta=%d is too few to provide suitable'
            ' resolution'
            )%(sigma,Ntheta))
        q = polar_smooth_contour(q,sigma/360.0*Ntheta)
    if close:
        q = np.concatenate([q,[q[0]]])
    return q


def resample_convex_hull(z,Ntheta=60):
    '''
    Resample a convex shape at uniform angular intervals 
    around its centroid
    
    Parameters
    ----------
    z:
    
    Other Parameters
    ----------------
    Ntheta: positive int; default 60
    
    Returns
    -------
    '''
    if Ntheta<4:
        raise ValueError(
        '# angles to sample should be >4; got %d'%Ntheta)
    
    z = convex_hull(z)
    c = np.mean(z)
    w = z-c    
    r = np.abs(w)
    h = np.angle(w)
    order = np.argsort(h)
    z,w,r,h = z[order],w[order],r[order],h[order]
    
    angles = np.linspace(-np.pi,np.pi,Ntheta+1)[:-1]
    rpad = np.concatenate([[r[-1]],r,[r[0]]])
    hpad = np.concatenate([[h[-1]-2*np.pi],h,[h[0]+2*np.pi]])
    r1 = np.interp(angles,hpad,rpad)
    
    return c + r1*np.exp(1j*angles)
    
    
def in_hull(z,hull): 
    '''
    Determine if the list of points P lies inside a convex 
    hull
    
    credit: https://stackoverflow.com/a/52405173/900749
    
    Parameters
    ----------
    z: z=x+iy points to test
    hull: ConvexHull, or points to form one with
    
    Returns
    -------
    in_hull: np.boolean
    '''
    z = p2z(z)
    s = z.shape
    z = z.ravel()
    if not isinstance(hull,ConvexHull):
        if np.any(np.iscomplex(hull)):
            hull = z2p(hull)
        if hull.shape[0]!=2:
            hull=hull.T
        hull = ConvexHull(hull.T)
    m = hull.equations[:,[1,0]] # half-plane directions
    b = hull.equations[:,-1].T  # half-plane thresholds
    return np.all(m@z2p(z) <= -b[:,None],0).reshape(*s)


