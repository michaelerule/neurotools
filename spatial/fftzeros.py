#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Code for identifying critical points in phase gradient maps.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import matplotlib as plt
import pylab as pl
from neurotools.signal import rewrap
from scipy.signal import convolve2d
from neurotools.graphics.plot import *

from neurotools.spatial.triangulation import mergeNearby

def plot_phase_gradient(dz):
    '''
    Plot a phase-gradient map using a hue wheel to show direction
    and local flow lines to indicate magnitude
    
    Parameters
    ----------
    dz : np.array
        Complex-valued square np.array of phase gradient directions.
        The gradient in the first dimension (x) is in the real component,
        anf the second dimension (y) in the imaginary component.
    '''
    plt.cla()
    plt.imshow(np.angle(dz),interpolation='nearest')
    plt.hsv()
    for i,row in list(enumerate(dz))[::1]:
        for j,z in list(enumerate(row))[::1]:
            z *=5
            plt.plot([j,j+np.real(z)],[i,i+np.imag(z)],'w',lw=1)
    h,w = np.shape(dz)
    plt.xlim(0-0.5,w-0.5)
    plt.ylim(h-0.5,0-0.5)

def plot_phase_direction(dz,skip=1,lw=1,zorder=None):
    '''
    Plot a phase-gradient map using a hue wheel to show direction
    and compass needles. Gradient magnitude is not shown.
    
    Parameters
    ----------
    dz : complex128
        phase gradient
        
    Other Parameters
    ----------------
    skip (int): only plot every skip
    lw (numeric): line width
    
    Returns
    -------
    '''
    plt.cla()
    plt.imshow(np.angle(dz),interpolation='nearest')
    plt.hsv()
    for i,row in list(enumerate(dz))[skip/2::skip]:
        for j,z in list(enumerate(row))[skip/2::skip]:
            z = 0.25*skip*z/np.abs(z)
            plt.plot([j,j+np.real(z)],[i,i+np.imag(z)],'w',lw=lw,zorder=zorder)
            z = -z
            plt.plot([j,j+np.real(z)],[i,i+np.imag(z)],'k',lw=lw,zorder=zorder)
    h,w = np.shape(dz)
    plt.xlim(0-0.5,w-0.5)
    plt.ylim(h-0.5,0-0.5)

def dPhidx(phase):
    '''
    Phase derivative in the x direction. 
    The returned array is smaller than the input array by one row and 
    column.
    
    Parameters
    ----------
    phase : np.array
        two-dimensional array of phases in *radians*
    
    Returns
    -------
    np.array
        phases differentiated along the x-axis (first dimension)
    '''
    dx = rewrap(np.diff(phase,1,0))
    dx = (dx[:,1:]+dx[:,:-1])*0.5
    return dx

def dPhidy(phase):
    '''
    Phase derivative in the y direction. 
    The returned array is smaller than the input array by one row and 
    column.
    
    Parameters
    ----------
    phase : np.array
        two-dimensional array of phases in *radians*
    
    Returns
    -------
    np.array
        phases differentiated along the y-axis (second dimension)
    '''
    dy = rewrap(np.diff(phase,1,1))
    dy = (dy[1:,:]+dy[:-1,:])*0.5
    return dy

def unwrap_indecies(tofind):
    '''
    Depricated, use np.where.
    TODO: remove all uses of this function
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    found = pl.find(tofind)
    h,w   = np.shape(tofind)
    return np.int32([(x%w,x/w) for x in found])

def get_phase_gradient_as_complex(data):
    '''
    Computes the phase gradient across an array and stores it in a complex
    number, in analogy to the complex-valued analytic signal. The complex
    phase indicates gradient direction, and the amplitude gradient 
    magnitude.
    
    Parameters
    ----------
    data : np.array
        Complex-valued array of analytic signals
    
    Returns
    -------
    dx : np.float
        Derivative in the x direction (first dimension)
    dy : np.float
        Derivative in the y direction (second dimension)
    dz : np.complex1
        Phase gradient stored as a complex number
    '''
    phase = np.angle(data)
    dx    = dPhidx(phase)
    dy    = dPhidy(phase)
    dz    = dy+1j*dx
    return dx,dy,dz

def getpeaks2d(pp):
    '''
    This function differentiates the array pp in the x and y direction
    and then looks for zero crossings. It returns an array the
    same size as pp but with 1 at points that are local maxima and 0 else.

    Parameters
    ----------
    pp: np.array
        a 2D array in which to search for local maxima
    
    Returns
    -------
    np.array
        an array with 1 at points that are local maxima and 0 elsewhere.
    '''
    dx  = np.diff(pp,1,0)[:,:-1]
    dy  = np.diff(pp,1,1)[:-1,:]
    ddx = np.diff(np.sign(dx),1,0)[:,:-1]/2
    ddy = np.diff(np.sign(dy),1,1)[:-1,:]/2
    maxima = (ddx*ddy==1)*(ddx==-1)
    result = np.int32(np.zeros(np.shape(pp)))
    result[1:-1,1:-1] = maxima
    return result

def coalesce(pp,s1=4,s2=None):
    '''
    Merge nearby peaks using Gaussian smoothing.
    
    Parameters
    ----------
    pp : np.array
        Boolean array with 1 indicating peak locations. 
    
    Other Parameters
    ----------------
    S1 : float
        x axis smoothing scale
    S2 : float
        y axis smoothing scale
    
    Returns
    -------
    pk : np.array
        Boolean array with 1 indicating peak location. The smoothing will
        merge nearby peaks. 
    '''
    if s2==None: s2=s1
    k1 = gausskern1d(s1,min(np.shape(pp)[1],int(ceil(6*s1))))
    k2 = gausskern1d(s2,min(np.shape(pp)[1],int(ceil(6*s2))))
    y  = np.array([convolve(x,k1,'same') for x in pp])
    y  = np.array([convolve(x,k2,'same') for x in y.T]).T
    pk = getpeaks2d(y)
    return pk

def coalesce_points(pp,radius):
    '''
    Merge nearby peaks using nearest-neighbords
    
    Parameters
    ----------
    pp : np.array
        Boolean array with 1 indicating peak locations. 
    radius : float
        Merge radius
    
    Returns
    -------
    pk : np.array
        Boolean array with 1 indicating peak location. The smoothing will
        merge nearby peaks. 
    '''
    if s2==None: s2=s1
    k1 = gausskern1d(s1,min(np.shape(pp)[1],int(ceil(6*s1))))
    k2 = gausskern1d(s2,min(np.shape(pp)[1],int(ceil(6*s2))))
    y  = np.array([convolve(x,k1,'same') for x in pp])
    y  = np.array([convolve(x,k2,'same') for x in y.T]).T
    pk = getpeaks2d(y)
    return pk

def find_critical_points(data,docoalesce=False,radius=4.0,edgeavoid=None):
    '''
    Parameters
    ----------
    data : np.array
        2D array complex phase values
        
    Other Parameters
    ----------------
    docoalesce : bool, False
        Whether to merge nearby critical points
    radius : float, 4.0
        Merge radius to use if `docoalesce` is true
    edgeavoid : float, None
        If not `None`, points `edgeavoid` distance to edge are omitted

    Returns
    -------
    clockwise : numpy.ndarray
        Point locations of centers of clockwise rotating waves
    anticlockwise : numpy.ndarray
        Point locations of centers of antclockwise rotating waves
    saddles : numpy.ndarray
        Saddle points in the phase gradient map
    peaks : numpy.ndarray
        All local minima or maxima in the phase gradient map
    maxima : numpy.ndarray
        Point locations of local maxima in the phase gradient 
    minima : numpy.ndarray
        Point locations of local minima in the phase gradient 
    '''
    dx,dy,dz = get_phase_gradient_as_complex(data)
    
    # extract curl via a kernal
    # take real component, centres have curl +- pi
    curl    = np.complex64([[-1-1j,-1+1j],[1-1j,1+1j]])
    temp    = convolve2d(dz,curl,'same','symm')
    winding = np.real(convolve2d(temp,np.ones((2,2))/4,'valid','symm'))
    
    # extract inflection points ( extrema, saddles )
    # by looking for sign changes in derivatives
    # avoid points close to the known centres
    avoid     = np.abs(winding)>5e-2
    ok        = ~avoid
    ddx       = np.diff(np.sign(dx),1,0)[:,:-1]/2
    ddy       = np.diff(np.sign(dy),1,1)[:-1,:]/2
    
    clockwise = winding>3
    anticlockwise = winding<-3 # close to pi, sometimes a little off
    saddles   = (ddx*ddy==-1)*ok
    peaks     = (ddx*ddy== 1)*ok
    maxima    = (ddx*ddy== 1)*(ddx==-1)*ok
    minima    = (ddx*ddy== 1)*(ddx== 1)*ok
    
    if docoalesce:
        clockwise     = coalesce_points(clockwise,radius)
        anticlockwise = coalesce_points(anticlockwise,radius)
        saddles       = coalesce_points(saddles,radius)
        peaks         = coalesce_points(peaks,radius)
        maxima        = coalesce_points(maxima,radius)
        minima        = coalesce_points(minima,radius)
    
    clockwise     = unwrap_indecies(clockwise)+1
    anticlockwise = unwrap_indecies(anticlockwise)+1
    saddles       = unwrap_indecies(saddles  )+1
    peaks         = unwrap_indecies(peaks    )+1
    maxima        = unwrap_indecies(maxima   )+1
    minima        = unwrap_indecies(minima   )+1
    
    if not edgeavoid is None:
        # remove points near edges
        Nrow,Ncol = data.shape[:2]
        def inbounds(pts):
            if np.prod(pts.shape)==0:
                # No points to check
                return pts
            outofbounds = (pts[:,0]<=edgeavoid)|(pts[:,1]<=edgeavoid)|(pts[:,0]>=Ncol-edgeavoid)|(pts[:,1]>=Nrow-edgeavoid)
            return pts[~outofbounds,:]
        anticlockwise = inbounds(anticlockwise)
        clockwise = inbounds(clockwise)
        saddles   = inbounds(saddles)
        peaks     = inbounds(peaks)
        maxima    = inbounds(maxima)
        minima    = inbounds(minima)
        
    return clockwise, anticlockwise, saddles, peaks, maxima, minima

def plot_critical_points(data,lw=1,ss=14,skip=5,ff=None,plotsaddles=True,aspect='auto',extent=None):
    '''
    
    Parameters
    ----------
    data : np.array
        2D complex-valued array of phases
        
    Other Parameters
    ----------------
    lw : int, default=1
        line width for plotting
    ss : int, default=14
        plotting point size
    skip : int, default=5
        Skip every `skip` points when plotting phase direction map
    plotsaddles : bool, defualt=True
        Whether to plot saddes points
    aspect : string, default='auto'
        Aspect-ratio parameter forwarded to matplotlib
    extent : tuple or None (default)
        extent parameter forwarded to matplotlib
    '''
    clockwise,anticlockwise,saddles,peaks,maxima,minima = find_critical_points(data)
    dx,dy,dz = get_phase_gradient_as_complex(data)
    plt.cla()
    plot_phase_direction(dz,skip=skip,lw=lw,zorder=Inf)
    N = np.shape(data)[0]
    if ff is None: 
        ff = np.arange(N)
    else:
        a,b = ff[0],ff[-1]
        K = b-a
        s = K/float(N)
        for d in [clockwise,anticlockwise,saddles,peaks,maxima,minima]:
            if d.size>0:
                d[:,1] = d[:,1]*s+a
    plot_complex(data,extent=extent,onlyphase=1)
    if len(clockwise)>0: 
        plt.scatter(*clockwise.T,s=ss**2,color='k',edgecolor='k',lw=lw,label='Clockwise')
    if len(anticlockwise)>0: 
        plt.scatter(*anticlockwise.T,s=ss**2,color='w',edgecolor='k',lw=lw,label='Anticlockwise')
    if len(maxima)>0:    
        plt.scatter(*maxima.T   ,s=ss**2,color='r',edgecolor='k',lw=lw,label='Maxima')
    if len(minima)>0:    
        plt.scatter(*minima.T   ,s=ss**2,color='g',edgecolor='k',lw=lw,label='Minima')
    if plotsaddles and len(saddles)>0:   
        plt.scatter(*saddles.T  ,s=ss**2,color=(1,0,1),edgecolor='k',lw=lw,label='Saddles')
    nice_legend()


def find_critical_potential_points(data):
    '''
    Critical points in a potential field (no centers / curl)

    Parameters
    ----------
    data : numeric array, 2D, complex

    Returns
    -------
    saddles : numpy.ndarray
    peaks : numpy.ndarray
    maxima : numpy.ndarray
    minima : numpy.ndarray
    '''

    dx,dy = grad(data)
    ddx       = np.diff(np.sign(dx),1,0)[:,:-1]/2
    ddy       = np.diff(np.sign(dy),1,1)[:-1,:]/2

    saddles   = (ddx*ddy==-1)
    peaks     = (ddx*ddy== 1)
    maxima    = (ddx*ddy== 1)*(ddx==-1)
    minima    = (ddx*ddy== 1)*(ddx== 1)

    saddles   = unwrap_indecies(saddles)+1
    peaks     = unwrap_indecies(peaks  )+1
    maxima    = unwrap_indecies(maxima )+1
    minima    = unwrap_indecies(minima )+1

    return saddles, peaks, maxima, minima

def grad(x):
    dx = np.diff(x,axis=1)
    dy = np.diff(x,axis=0)
    resultx = zeroslike(x)
    resulty = zeroslike(x)
    resultx[:,1:]  += dx
    resultx[:,:-1] += dx
    resultx[:,1:-1]*= 0.5
    resulty[1:,:]  += dy
    resulty[:-1,:] += dy
    resulty[1:-1,:]*= 0.5
    return resultx,resulty

def quickgrad(x):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    dx = np.diff(x,axis=1)[:-1,:]
    dy = np.diff(x,axis=0)[:,:-1]
    return dx,dy

def getp(x):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    return unwrap_indecies(coalesce(x))+1

def get_critical_spectra(ff,wt):
    '''
    smt and smf are time and frequency smoothing scales
    in units of pixels
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    wt      = squeeze(wt)
    dx,dy   = quickgrad(np.abs(wt))
    ddx     = np.diff(np.sign(dx),1,1)[:-1,:]/2
    ddy     = np.diff(np.sign(dy),1,0)[:,:-1]/2
    aextrem = ddx*ddy== 1
    amaxima = aextrem*(ddx==-1)
    aminima = aextrem*(ddx== 1)
    amaxima = unwrap_indecies(amaxima)+1
    aminima = unwrap_indecies(aminima)+1
    maxs = np.float32(amaxima)
    mins = np.float32(aminima)
    maxs[:,1] = ff[amaxima[:,1]]
    mins[:,1] = ff[aminima[:,1]]
    return mins,maxs

def plot_critical_spectra(ff,wt,ss=5,aspect=None):
    '''
    smt and smf are time and frequency smoothing scales
    in units of pixels
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    wt      = squeeze(wt)
    nf,nt = np.shape(wt)
    if aspect is None: aspect = float(nt)/nf/3
    dx,dy   = quickgrad(np.abs(wt))
    ddx     = np.diff(np.sign(dx),1,1)[:-1,:]/2
    ddy     = np.diff(np.sign(dy),1,0)[:,:-1]/2
    aextrem = ddx*ddy== 1
    amaxima = aextrem*(ddx==-1)
    aminima = aextrem*(ddx== 1)
    amaxima = unwrap_indecies(amaxima)+1
    aminima = unwrap_indecies(aminima)+1
    amaxima[:,1] = ff[amaxima[:,1]]
    aminima[:,1] = ff[aminima[:,1]]
    cla()
    plotWTPhase(ff,wt,aspect=aspect)#,interpolation='nearest')
    if len(amaxima)>0: plt.scatter(*amaxima.T,s=ss**2,color='k',edgecolor='w',lw=1)
    if len(aminima)>0: plt.scatter(*aminima.T,s=ss**2,color='w',edgecolor='k',lw=1)
    draw()
    show()
    return aminima,amaxima

def cut_array_data(data,arrayMap,cutoff=1.8,spacing=0.4):
    '''
    data should be a NChannel x Ntimes array
    arrayMap should be an L x K array of channel IDs,
    1-indexed, with "-1" to indicate missing or bad channels
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    packed = packArrayDataInterpolate(data,arrayMap)
    return dctCut(packed,cutoff,spacing)
    
    
def mirror2D(C):
    C = np.array(C)
    C = np.cat([C,C[::-1]])
    C = np.cat([C,C[:,::-1]],1)
    return C
