#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Statistical routines to extract spatial summary statistics from 2D 
arrays of complex-valued (phase,amplitude) signals.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
from numpy import pi, e

from scipy.signal.signaltools import convolve2d
from neurotools.tools         import warn
from neurotools.spatial.dct   import dct_upsample,dct_cut_antialias
from neurotools.signal.signal import rewrap

ELECTRODE_SPACING = 0.4

def array_average_amplitude(frames):
    '''
    Computes the average signal amplitude envelope over multi-electrode
    array data. 
    Assumes first two dimensions are (x,y) spatial dimensions.
    
    Parameters
    ----------
    frames : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
        
    Returns
    -------
    np.array
        The average absolute magnitude 
        over the first two axes
    '''
    return np.mean(np.abs(frames),axis=(0,1))

def array_kuramoto(frames):
    '''
    Computes the Kuramoto order parameter 
    over 2D multi-electrode array given complex-valued analytic signals.
    Assumes first two dimensions are (x,y) spatial dimensions.
    
    Parameters
    ----------
    frames : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
        
    Returns
    -------
    np.array
        The Kuramotor order parameter
        over the first two axes
        
    '''
    return np.abs(np.mean(frames/np.abs(frames),axis=(0,1)))

def array_synchrony(frames):
    '''
    Computes the 
    average phase syncrony
    over 2D multi-electrode array given complex-valued analytic signals.
    Assumes first two dimensions are (x,y) spatial dimensions.
    
    Parameters
    ----------
    frames : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
        
    Returns
    -------
    np.array
        The average phase syncrony
        over the first two axes
        
    '''
    return np.abs(np.mean(frames,axis=(0,1)))/np.mean(np.abs(frames),axis=(0,1))

def array_kuramoto_standard_deviation(frames):
    '''
    Computes the 
    Kuramoto order parameter, transformed to units of radians
    over 2D multi-electrode array given complex-valued analytic signals.
    Assumes first two dimensions are (x,y) spatial dimensions.
    
    Parameters
    ----------
    frames : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
        
    Returns
    -------
    np.array
        The Kuramotor order parameter transformed to units of radians
        over the first two axes
        
    '''
    R = array_kuramoto(frames)
    return np.sqrt(-2*np.log(R))

def array_synchrony_standard_deviation(frames):
    '''
    Computes the 
    average phase syncrony, transformed to units of radians
    over 2D multi-electrode array given complex-valued analytic signals.
    Assumes first two dimensions are (x,y) spatial dimensions.
    
    Parameters
    ----------
    frames : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
        
    Returns
    -------
    np.array
        The average phase syncrony
        over the first two axes
        transformed to units of radians
        
    '''
    R = array_synchrony(frames)
    return np.sqrt(-2*np.log(R))

def population_magnitude_weighted_circular_standard_deviation(population):
    '''
    Calls population_synchrony and transforms result to units of
    radians.

    Parameters
    ----------
        
    Returns
    -------
    '''
    syn = population_synchrony(population)
    return np.sqrt(-2*np.log(syn))

def array_linear_synchrony(population):
    '''

    Parameters
    ----------
        
    Returns
    -------
    '''
    return 1/(1-array_synchrony(population))

def array_phase_gradient(frame):
    '''
    Assumes complex input (analytic signals)
    Assumes first 2 dimensions are array dimensions
    Remaining dimensions can be arbitrary
    first dimention is Y second is X ( row major ordering )

    The differentiation kernel is [-0.5, 0, 0.5]

    Parameters
    ----------
        
    Returns
    -------
    '''
    if frame.dtype==np.complex64 or\
        frame.dtype==np.complex128:
        frame=np.angle(frame)
    dy = (frame[1:,:,...]-frame[:-1,:,...]+pi)%(2*pi)-pi
    dx = (frame[:,1:,...]-frame[:,:-1,...]+pi)%(2*pi)-pi
    dy = (dy[:,1:,...]+dy[:,:-1,...])*0.5
    dx = (dx[1:,:,...]+dx[:-1,:,...])*0.5
    return dx+1j*dy

def array_count_centers(rawdata,upsample=3,cut=True,cutoff=0.4):
    '''
    Counting centers -- looks for channels around which phase skips +-pi
    will miss rotating centers closer than half the electrode spacing

    Parameters
    ----------
        
    Returns
    -------
    '''
    # can only handle dim 3 for now
    assert len(rawdata.shape)==3
    if cut:
        data = dct_upsample(dct_cut_antialias(
            rawdata,cutoff,ELECTRODE_SPACING),factor=upsample)
    else:
        data = dct_upsample(data,factor=upsample)
    dz = array_phase_gradient(data)
    curl = np.complex64([[-1-1j,-1+1j],[1-1j,1+1j]])
    curl = np.convolve2d(curl,np.ones((2,2))/4,'full')
    winding = arr([real(np.convolve2d(z,curl,'valid','symm')) for z in dz.transpose((2,0,1))])
    nclockwise = np.sum(np.int32(winding> 3),axis=(1,2))
    nwidersyns = np.sum(np.int32(winding<-3),axis=(1,2))
    return nclockwise, nwidersyns

def array_count_critical(rawdata,upsample=3,cut=True,cutoff=0.4):
    '''
    Count critical points in the phase gradient map

    Parameters
    ----------
        
    Returns
    -------
    '''
    # can only handle dim 3 for now
    assert len(shape(rawdata))==3
    if cut:
        data = dct_upsample(dct_cut_antialias(rawdata,cutoff,ELECTRODE_SPACING),factor=upsample)
    else:
        data = dct_upsample(data,factor=upsample)
    dz = array_phase_gradient(data).transpose((2,0,1))
    # extract curl via a kernal
    # take real component, centres have curl +- pi
    curl = np.complex64([[-1-1j,-1+1j],[1-1j,1+1j]])
    curl = np.convolve2d(curl,np.ones((2,2))/4,'full')
    winding = arr([np.convolve2d(z,curl,'same','symm').real for z in dz])

    # extract inflection points ( extrema, saddles )
    # by looking for sign changes in derivatives
    # avoid points that are close to singular
    ok        = ~(np.abs(winding)<1e-1)[...,:-1,:-1]
    ddx       = np.diff(np.sign(dz.real),1,1)[...,:,:-1]/2
    ddy       = np.diff(np.sign(dz.imag),1,2)[...,:-1,:]/2
    saddles   = (ddx*ddy==-1)*ok
    maxima    = (ddx*ddy== 1)*(ddx==-1)*ok
    minima    = (ddx*ddy== 1)*(ddx== 1)*ok
    sum2 = lambda x: np.sum(np.int32(x),axis=(1,2))
    nclockwise = sum2(winding>3)
    nwidersyns = sum2(winding<-3)
    nsaddles   = sum2(saddles  )
    nmaxima    = sum2(maxima   )
    nminima    = sum2(minima   )
    return nclockwise, nwidersyns, nsaddles, nmaxima, nminima

def array_phasegradient_upper(frame):
    '''
    The average gradient magnitude can be inflated if there is noise
    but provides an upper bound on spatial frequency ( lower bound on
    wavelength ).

    Parameters
    ----------
        
    Returns
    -------
    '''
    warn('expects first two dimensions x,y of 2d array data')
    pg = array_phase_gradient(frame)
    return np.mean(np.abs(pg),axis=(0,1))/(ELECTRODE_SPACING*2*pi)

def array_phasegradient_lower(frame):
    '''
    The magnitude of the average gradient provides a very accurate estimate
    of wavelength even in the presence of noise. However, it will
    understimate the phase gradient if the wave structure is not perfectly
    planar

    Returns cycles/mm
    i.e.
    radians/electrode / (mm/electrode) / (2*pi radians/cycle)

    Parameters
    ----------
        
    Returns
    -------
    '''
    warn('expects first two dimensions x,y of 2d array data')
    pg = array_phase_gradient(frame)
    return np.abs(np.mean(pg,axis=(0,1)))/(ELECTRODE_SPACING*2*pi)

def array_phasegradient_magnitude_sigma(frame):
    '''
    expects first two dimensions x,y of 2d array data

    Parameters
    ----------
        
    Returns
    -------
    '''
    pg = array_phase_gradient(frame)
    return np.std(np.abs(pg),axis=(0,1))

def array_phasegradient_magnitude_cv(frame):
    '''
    Coefficient of variation of the magnitudes of the phase gradients
    expects first two dimensions x,y of 2d array data

    Parameters
    ----------
        
    Returns
    -------
    '''
    pg = array_phase_gradient(frame)
    return np.std(np.abs(pg),axis=(0,1))/np.mean(np.abs(pg),
        axis=(0,1))

def array_phasegradient_pgd_threshold(frame,thresh=0.5):
    '''
    The magnitude of the average gradient provides a very accurate estimate
    of wavelength even in the presence of noise. However, it will
    understimate the phase gradient if the wave structure is not perfectly
    planar.

    Returns cycles/mm
    i.e.
    radians/electrode / (mm/electrode) / (2*pi radians/cycle)

    Parameters
    ----------
        
    Returns
    -------
    '''
    warn('expects first two dimensions x,y of 2d array data')
    pg  = array_phase_gradient(frame)
    use = array_synchrony_pgd(frame)>=thresh
    pg[:,:,~use] = np.NaN
    return np.abs(np.mean(pg,axis=(0,1)))/(ELECTRODE_SPACING*2*pi)

def array_wavelength_pgd_threshold(frame,thresh=0.5):
    '''
    The magnitude of the average gradient provides a very accurate estimate
    of wavelength even in the presence of noise. However, it will
    understimate the phase gradient if the wave structure is not perfectly
    planar

    returns mm/cycle

    Parameters
    ----------
        
    Returns
    -------
    '''
    warn('expects first two dimensions x,y of 2d array data')
    return 1/array_phasegradient_pgd_threshold(frame,thresh)


def array_wavelength_lower_pgd_threshold(frame,thresh=0.5):
    '''
    The average phase gradient magnitude can tolerate non-planar waves, but
    is particularly sensitive to noise. It may be appropriate to combine
    this method with spatial smoothing to denoise the data, if it is safe
    to assume a minimum spatial scale for the underlying wave dynamics.

    returns mm/cycle

    Parameters
    ----------
        
    Returns
    -------
    '''
    warn('expects first two dimensions x,y of 2d array data')
    pg  = array_phase_gradient(frame)
    use = array_synchrony_pgd(frame)>=thresh
    pg[:,:,~use] = np.NaN
    return 1/np.mean(abs(pg),axis=(0,1))/(ELECTRODE_SPACING*2*pi)


def array_speed_pgd_threshold(frame,thresh=0.5):
    '''
    expects first two dimensions x,y of 2d array data
    returns speed for plane waves in mm/s

    Parameters
    ----------
        
    Returns
    -------
    '''
    pg = array_phasegradient_pgd_threshold(frame,thresh) #cycles / mm
    df = np.median(np.ravel(rewrap(np.diff(np.angle(frame),1,2)))) #radians/sample
    warn('ASSUMING FS=1000ms HERE!!!')
    f  = df*1000.0 # radians / s
    g  = f /(2*pi) # cycles / s
    return g/pg # mm /s

def array_speed_upper(frame):
    '''
    expects first two dimensions x,y of 2d array data
    returns speed for plane waves in mm/s

    Parameters
    ----------
        
    Returns
    -------
    '''
    assert 0
    warn('BROKEN DONT USE')
    pg = array_phasegradient_lower(frame) #cycles / mm
    df = np.median(np.ravel(rewrap(np.diff(np.angle(frame),1,2)))) #radians/sample
    warn('ASSUMING FS=1000ms HERE!!!')
    f  = df*1000.0 # radians / s
    g  = f /(2*pi) # cycles / s
    return g/pg # mm /s


def array_speed_lower(frame):
    '''
    expects first two dimensions x,y of 2d array data
    returns speed for plane waves in mm/s

    Parameters
    ----------
        
    Returns
    -------
    '''
    pg = array_phasegradient_upper(frame) #cycles / mm
    df = np.median(np.ravel(rewrap(np.diff(np.angle(frame),1,2)))) #radians/sample
    warn('ASSUMING FS=1000ms HERE!!!')
    f  = df*1000.0 # radians / s
    g  = f /(2*pi) # cycles / s
    return g/pg # mm /s

def array_wavelength_lower(frame):
    '''
    phase gradients are in units of radians per electrode
    we would like units of mm per cycle
    there are 2pi radians per cycle
    there are 0.4mm per electrode
    phase gradient / 2 pi is in units of cycles per electrode
    electrode spacing / (phase gradient / 2 pi)

    Parameters
    ----------
        
    Returns
    -------
    '''
    return 1/array_phasegradient_upper(frame)

def array_wavelength_upper(frame):
    '''
    phase gradients are in units of radians per electrode
    we would like units of mm per cycle
    there are 2pi radians per cycle
    there are 0.4mm per electrode
    phase gradient / 2 pi is in units of cycles per electrode
    electrode spacing / (phase gradient / 2 pi)

    Parameters
    ----------
        
    Returns
    -------
    '''
    return 1/array_phasegradient_lower(frame)

def array_synchrony_pgd(frame):
    '''
    The phase gradient directionality measure from Rubinto et al 2009 is
    abs(mean(pg))/mean(abs(pg))

    Parameters
    ----------
        
    Returns
    -------
    '''
    pg = array_phase_gradient(frame)
    return np.abs(np.mean(pg,axis=(0,1)))/np.mean(np.abs(pg),axis=(0,1))

def array_synchrony_pgd_standard_deviation(frame):
    '''
    The phase gradient directionality measure from Rubinto et al 2009 is
    abs(mean(pg))/mean(abs(pg))

    Parameters
    ----------
        
    Returns
    -------
    '''
    R = array_synchrony_pgd(frame)
    return np.sqrt(-2*np.log(R))

def array_kuramoto_pgd(frame):
    '''
    A related directionality index ignores vector amplitude.

    Parameters
    ----------
        
    Returns
    -------
    '''
    pg = array_phase_gradient(frame)
    return np.abs(np.mean(pg/np.abs(pg),axis=(0,1)))

def array_kuramoto_pgd_standard_deviation(frame):
    '''
    A related directionality index ignores vector amplitude.

    Parameters
    ----------
        
    Returns
    -------
    '''
    pg = array_phase_gradient(frame)
    return np.abs(np.mean(pg/np.abs(pg),axis=(0,1)))

def trim_array(arrayMap):
    '''
    Removes any rows or columns from the array map
    that are empty ( have no channels )

    Parameters
    ----------
        
    Returns
    -------
    '''
    arrayMap = np.int32(arrayMap)
    notDone = True
    while notDone:
        notDone = False
        if all(arrayMap[0,:]==-1):
            arrayMap = arrayMap[1:,:]
            notDone = True
        if all(arrayMap[:,0]==-1):
            arrayMap = arrayMap[:,1:]
            notDone = True
        if all(arrayMap[-1,:]==-1):
            arrayMap = arrayMap[:-1,:]
            notDone = True
        if all(arrayMap[:,-1]==-1):
            arrayMap = arrayMap[:,:-1]
            notDone = True
    return arrayMap

def trim_array_as_if(arrayMap,data):
    '''
    Removes any rows or columns from data if those rows or columns are
    empty ( have no channels ) in the arrayMap

    Parameters
    ----------
        
    Returns
    -------
    '''
    arrayMap = np.int32(arrayMap)
    notDone = True
    while notDone:
        notDone = False
        if all(arrayMap[0,:]==-1):
            arrayMap = arrayMap[1:,:]
            data     = data    [1:,:]
            notDone = True
        if all(arrayMap[:,0]==-1):
            arrayMap = arrayMap[:,1:]
            data     = data    [:,1:]
            notDone = True
        if all(arrayMap[-1,:]==-1):
            arrayMap = arrayMap[:-1,:]
            data     = data    [:-1,:]
            notDone = True
        if all(arrayMap[:,-1]==-1):
            arrayMap = arrayMap[:,:-1]
            data     = data    [:,:-1]
            notDone = True
    return data

def pack_array_data(data,arrayMap):
    '''
    Accepts a collection of signals from array channels, as well as
    an array map containing indecies (1-indexed for backwards compatibility
    with matlab) into that list of channel data.

    This will interpolate missing channels as an average of nearest
    neighbors.

    :param data: NChannel x Ntimes array
    :param arrayMap: array map, 1-indexed, 0 for missing electrodes
    :return: returns LxKxNtimes 3D array of the interpolated channel data

    Parameters
    ----------
        
    Returns
    -------
    '''
    # first, trim off any empty rows or columns from the arrayMap
    arrayMap = trim_array(arrayMap)
    # prepare array into which to pack data
    L,K    = arrayMap.shape
    NCH,N  = data.shape
    packed = np.zeros((L,K,N),dtype=data.dtype)
    J = data.shape[0]
    M = np.sum(arrayMap>0)
    if J!=M:
        warn('bad: data dimension differs from number of array electrodes')
        warn('data %s %s array'%(J,M))
        warn('this may just be because some array channels are removed')
    for i,row in enumerate(arrayMap):
        for j,ch in enumerate(row):
            # need to convert channel to channel index
            if ch<=0:
                # we will need to interpolate from nearest neighbors
                ip = []
                if i>0  : ip.append(arrayMap[i-1,j])
                if j>0  : ip.append(arrayMap[i,j-1])
                if i+1<L: ip.append(arrayMap[i+1,j])
                if j+1<K: ip.append(arrayMap[i,j+1])
                ip = [ch for ch in ip if ch>0]
                assert len(ip)>0
                for chii in ip:
                    packed[i,j,:] += data[chii-1,:]
                packed[i,j,:] *= 1.0/len(ip)
            else:
                assert ch>0
                packed[i,j,:] = data[ch-1,:]
    return packed
