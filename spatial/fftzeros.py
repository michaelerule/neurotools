#!/usr/bin/python
# -*- coding: UTF-8 -*-
# The above two lines should appear in all python source files!
# It is good practice to include the lines below
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division

'''
Code for identifying critical points in phase gradient maps.
'''
from neurotools.signal.dct import *
from neurotools.plot import *

def plot_phase_gradient(dz):
    cla()
    imshow(angle(dz),interpolation='nearest')
    hsv()
    for i,row in list(enumerate(dz))[::1]:
        for j,z in list(enumerate(row))[::1]:
            z *=5
            plot([j,j+real(z)],[i,i+imag(z)],'w',lw=1)
    h,w = shape(dz)
    xlim(0-0.5,w-0.5)
    ylim(h-0.5,0-0.5)

def plot_phase_direction(dz,skip=1,lw=1,zorder=None):
    '''
    Parameters:
        dz (complex123): phase gradient
        skip (int): only plot every skip
        lw (numeric): line width
    '''
    cla()
    imshow(angle(dz),interpolation='nearest')
    hsv()
    for i,row in list(enumerate(dz))[skip/2::skip]:
        for j,z in list(enumerate(row))[skip/2::skip]:
            z = 0.25*skip*z/abs(z)
            plot([j,j+real(z)],[i,i+imag(z)],'w',lw=lw,zorder=zorder)
            z = -z
            plot([j,j+real(z)],[i,i+imag(z)],'k',lw=lw,zorder=zorder)
    h,w = shape(dz)
    xlim(0-0.5,w-0.5)
    ylim(h-0.5,0-0.5)

def dPhidx(phase):
    dx = rewrap(diff(phase,1,0))
    dx = (dx[:,1:]+dx[:,:-1])*0.5
    return dx

def dPhidy(phase):
    dy = rewrap(diff(phase,1,1))
    dy = (dy[1:,:]+dy[:-1,:])*0.5
    return dy

def unwrap_indecies(tofind):
    found = find(tofind)
    h,w = shape(tofind)
    return int32([(x%w,x/w) for x in found])

def get_phase_gradient_as_complex(data):
    phase = angle(data)
    dx    = dPhidx(phase)
    dy    = dPhidy(phase)
    dz    = dy+1j*dx
    return dx,dy,dz

def getpeaks2d(pp):
    '''
    This function differentiates the array pp in the x and y direction
    and then looks for zero crossings. It should return an array the
    same size as pp but with 1 at points that are local maxima and 0 else.
    
        :pp: a 2D array in which to search for local maxima
    '''
    dx  = diff(pp,1,0)[:,:-1]
    dy  = diff(pp,1,1)[:-1,:]
    ddx = diff(sign(dx),1,0)[:,:-1]/2
    ddy = diff(sign(dy),1,1)[:-1,:]/2
    maxima = (ddx*ddy==1)*(ddx==-1)
    result = int32(zeros(shape(pp)))
    result[1:-1,1:-1] = maxima
    return result

def coalesce(pp,s1=4,s2=None):
    '''
    S1 and S2 are time/frequency smoothing scale.
    This is a major bottleneck.
    '''
    if s2==None: s2=s1
    k1 = gausskern1d(s1,min(shape(pp)[1],int(ceil(6*s1))))
    k2 = gausskern1d(s2,min(shape(pp)[1],int(ceil(6*s2))))
    y = array([convolve(x,k1,'same') for x in pp])
    y = array([convolve(x,k2,'same') for x in y.T]).T
    pk = getpeaks2d(y)
    return pk

def find_critical_points(data,docoalesce=False):
    '''
    Parameters
    ----------
    data : numeric array, 2D, complex
        
    Returns
    -------
    clockwise : numpy.ndarray
    widersyns : numpy.ndarray
    saddles : numpy.ndarray
    peaks : numpy.ndarray
    maxima : numpy.ndarray
    minima : numpy.ndarray
    '''
    dx,dy,dz = get_phase_gradient_as_complex(data)

    # extract curl via a kernal
    # take real component, centres have curl +- pi
    curl = complex64([[-1-1j,-1+1j],[1-1j,1+1j]])
    temp = convolve2d(dz,curl,'same','symm')
    winding = real(convolve2d(temp,ones((2,2))/4,'valid','symm'))

    # extract inflection points ( extrema, saddles )
    # by looking for sign changes in derivatives
    # avoid points close to the known centres
    avoid     = abs(winding)>1e-1 
    ok        = ~avoid
    ddx       = diff(sign(dx),1,0)[:,:-1]/2
    ddy       = diff(sign(dy),1,1)[:-1,:]/2
    
    clockwise = winding>3
    widersyns = winding<-3 # close to pi, sometimes a little 
    saddles   = (ddx*ddy==-1)*ok
    peaks     = (ddx*ddy== 1)*ok
    maxima    = (ddx*ddy== 1)*(ddx==-1)*ok
    minima    = (ddx*ddy== 1)*(ddx== 1)*ok
    
    if docoalesce:
        clockwise = unwrap_indecies(coalesce(clockwise))+1
        widersyns = unwrap_indecies(coalesce(widersyns))+1
        saddles   = unwrap_indecies(coalesce(saddles  ))+1
        peaks     = unwrap_indecies(coalesce(peaks    ))+1
        maxima    = unwrap_indecies(coalesce(maxima   ))+1
        minima    = unwrap_indecies(coalesce(minima   ))+1
    else:
        clockwise = unwrap_indecies(clockwise)+1
        widersyns = unwrap_indecies(widersyns)+1
        saddles   = unwrap_indecies(saddles  )+1
        peaks     = unwrap_indecies(peaks    )+1
        maxima    = unwrap_indecies(maxima   )+1
        minima    = unwrap_indecies(minima   )+1

    return clockwise, widersyns, saddles, peaks, maxima, minima

def plot_critical_points(data,lw=1,ss=14,skip=5,ff=None,plotsaddles=True,aspect='auto',extent=None):
    '''
    Parameters:
        skip (int): only plot every [skip] phase gradient vectors
    '''
    clockwise,widersyns,saddles,peaks,maxima,minima = find_critical_points(data)
    dx,dy,dz = get_phase_gradient_as_complex(data)
    cla()
    plot_phase_direction(dz,skip=skip,lw=lw,zorder=Inf)
    N = shape(data)[0]
    if ff is None: ff = arange(N)
    else:
        a,b = ff[0],ff[-1]
        K = b-a
        s = K/float(N)
        for d in [clockwise,widersyns,saddles,peaks,maxima,minima]:
            if d.size==0: continue
            d[:,1] = d[:,1]*s+a
    plot_complex(data,extent=extent,onlyphase=1)
    if len(clockwise)>0: scatter(*clockwise.T,s=ss**2,color='k',edgecolor='k',lw=lw,label='Clockwise')
    if len(widersyns)>0: scatter(*widersyns.T,s=ss**2,color='w',edgecolor='k',lw=lw,label='Anticlockwise')
    if len(maxima)>0:    scatter(*maxima.T   ,s=ss**2,color='r',edgecolor='k',lw=lw,label='Maxima')
    if len(minima)>0:    scatter(*minima.T   ,s=ss**2,color='g',edgecolor='k',lw=lw,label='Minima')
    if plotsaddles:
        if len(saddles)>0:   scatter(*saddles.T  ,s=ss**2,color=(1,0,1),edgecolor='k',lw=lw,label='Saddles')
    nice_legend()


def find_critical_potential_points(data):
    '''
    Critical points in a potential field (no centers / curl)
    
    Parameters
    ----------
    data : numeric array, 2D, complex
        
    Returns
    -------
    clockwise : numpy.ndarray
    widersyns : numpy.ndarray
    saddles : numpy.ndarray
    peaks : numpy.ndarray
    maxima : numpy.ndarray
    minima : numpy.ndarray
    '''
    
    dx,dy = grad(data)
    ddx       = diff(sign(dx),1,0)[:,:-1]/2
    ddy       = diff(sign(dy),1,1)[:-1,:]/2
    
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
    dx = diff(x,axis=1)
    dy = diff(x,axis=0)
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
    dx = diff(x,axis=1)[:-1,:]
    dy = diff(x,axis=0)[:,:-1]
    return dx,dy

def getp(x):
    return unwrap_indecies(coalesce(x))+1

def get_critical_spectra(ff,wt):
    '''
    smt and smf are time and frequency smoothing scales
    in units of pixels
    '''
    wt      = squeeze(wt)
    dx,dy   = quickgrad(abs(wt))
    ddx     = diff(sign(dx),1,1)[:-1,:]/2
    ddy     = diff(sign(dy),1,0)[:,:-1]/2
    aextrem = ddx*ddy== 1
    amaxima = aextrem*(ddx==-1)
    aminima = aextrem*(ddx== 1)
    amaxima = unwrap_indecies(amaxima)+1
    aminima = unwrap_indecies(aminima)+1
    maxs = float32(amaxima)
    mins = float32(aminima)
    maxs[:,1] = ff[amaxima[:,1]]
    mins[:,1] = ff[aminima[:,1]]
    return mins,maxs

def plot_critical_spectra(ff,wt,ss=5,aspect=None):
    '''
    smt and smf are time and frequency smoothing scales
    in units of pixels
    '''
    wt      = squeeze(wt)
    nf,nt = shape(wt)
    if aspect is None: aspect = float(nt)/nf/3
    dx,dy   = quickgrad(abs(wt))
    ddx     = diff(sign(dx),1,1)[:-1,:]/2
    ddy     = diff(sign(dy),1,0)[:,:-1]/2
    aextrem = ddx*ddy== 1
    amaxima = aextrem*(ddx==-1)
    aminima = aextrem*(ddx== 1)
    amaxima = unwrap_indecies(amaxima)+1
    aminima = unwrap_indecies(aminima)+1
    amaxima[:,1] = ff[amaxima[:,1]]
    aminima[:,1] = ff[aminima[:,1]]
    cla()
    plotWTPhase(ff,wt,aspect=aspect)#,interpolation='nearest')
    if len(amaxima)>0: scatter(*amaxima.T,s=ss**2,color='k',edgecolor='w',lw=1)
    if len(aminima)>0: scatter(*aminima.T,s=ss**2,color='w',edgecolor='k',lw=1)
    draw()
    show()
    return aminima,amaxima

def cut_array_data(data,arrayMap,cutoff=1.8,spacing=0.4):
    '''
    data should be a NChannel x Ntimes array
    arrayMap should be an L x K array of channel IDs,
    1-indexed, with "-1" to indicate missing or bad channels
    '''
    packed = packArrayDataInterpolate(data,arrayMap)
    return dctCut(packed,cutoff,spacing)














