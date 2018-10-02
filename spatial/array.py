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
from scipy.signal.signaltools import convolve2d
from neurotools.tools         import warn
from neurotools.spatial.dct   import dct_upsample,dct_cut_antialias
from neurotools.signal        import rewrap

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
        The average absolute magnitude over the first two axes
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
        The Kuramotor order parameter over the first two axes
        
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
        The average phase syncrony over the first two axes
        
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
        The average phase syncrony over the first two axes
        transformed to units of radians
        
    '''
    R = array_synchrony(frames)
    return np.sqrt(-2*np.log(R))

def array_phase_gradient(frame):
    '''
    Assumes complex input (analytic signals)
    Assumes first 2 dimensions are array dimensions
    Remaining dimensions can be arbitrary
    first dimention is Y second is X ( row major ordering )

    The differentiation kernel is [1 -1; 1 -1]/2, exept at the boundaries.

    The returned phase gradients are two-dimensional, and encoded as 
    a complex number (in analogy to the analytic signal). The gradient
    along the first dimension (x) is encoded in the real part, and the
    gradient along the second dimension (y) is encoded in the imaginary
    part. 

    Parameters
    ----------
    frame : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
        
    Returns
    -------
    np.array
        Arra phase gradients encoded as complex numbers
    '''
    if frame.dtype==np.complex64 or frame.dtype==np.complex128:
        frame=np.angle(frame)
    # This computes the kernel
    #  [0.5, -0.5
    #   0.5, -0.5]
    dy = (np.diff(frame,n=1,axis=0)+np.pi)%(2*np.pi)-np.pi
    dx = (np.diff(frame,n=1,axis=1)+np.pi)%(2*np.pi)-np.pi
    dy = (dy[:,1:,...]+dy[:,:-1,...])*0.5
    dx = (dx[1:,:,...]+dx[:-1,:,...])*0.5
    return dx+1j*dy

def array_count_centers(data,upsample=3,cut=True,cutoff=0.4,electrode_spacing=0.4):
    '''
    Counting centers -- looks for channels around which phase skips +-np.pi
    
    Parameters
    ----------
    data : np.array
        Data should be sent in a 3-dimensional np.array of complex analytic
        signals, where the first two dimensions are the (x,y) spatial 
        dimensions of the multi-electrode array.
    
    Other Parameters
    ----------------
    upsample : int, default is 3
        Upsampling factor for interpolating between electrodes
    cut : bool, default is True
        Whether to apply a spatial frequency cutoff. 
    cutoff : float, default is 0.4
        Spatial scale cutoff for analysis, in mm
    electrode_spacing : float, default 0.4
        Spacing between electrodes in array in mm. Default is 0.4mm for
        the Utah arrays
    
    Returns
    -------
    nclockwise : np.array
        number of clockwise centers found at each time point
    nanticlockwise : np.array
        number of anticlockwise centers found at each time point
    '''
    if not len(shape(data))==3:
        raise ValueError('Data should be formatted with shape (x,y,t)')
    if cut:
        data = dct_upsample(dct_cut_antialias(
            data,cutoff,electrode_spacing),factor=upsample)
    else:
        data = dct_upsample(data,factor=upsample)
    dz   = array_phase_gradient(data)
    curl = np.complex64([[-1-1j,-1+1j],[1-1j,1+1j]])
    curl = np.convolve2d(curl,np.ones((2,2))/4,'full')
    winding = arr([real(np.convolve2d(z,curl,'valid','symm')) for z in dz.transpose((2,0,1))])
    nclockwise     = np.sum(np.int32(winding> 3),axis=(1,2))
    nanticlockwise = np.sum(np.int32(winding<-3),axis=(1,2))
    return nclockwise, nanticlockwise

def array_count_critical(data,upsample=3,cut=True,cutoff=0.4,electrode_spacing=0.4):
    '''
    Count critical points in the phase gradient map

    Parameters
    ----------
    data : np.array
        Data should be sent in a 3-dimensional np.array of complex analytic
        signals, where the first two dimensions are the (x,y) spatial 
        dimensions of the multi-electrode array.
        
    Other Parameters
    ----------------
    upsample : int, default is 3
        Upsampling factor for interpolating between electrodes
    cut : bool, default is True
        Whether to apply a spatial frequency cutoff. 
    cutoff : float, default is 0.4
        Spatial scale cutoff for analysis. 
    electrode_spacing : float, default 0.4
        Spacing between electrodes in array in mm. Default is 0.4mm for
        the Utah arrays
    
    Returns
    -------
    nclockwise : np.array
        number of clockwise centers found at each time point
    nanticlockwise : np.array
        number of anticlockwise centers found at each time point
    nsaddles : np.array
        number of saddle points
    nmaxima : np.array
        number of local maxima
    nminima : np.array
        number of local minima
    '''
    if not len(shape(data))==3:
        raise ValueError('Data should be formatted with shape (x,y,t)')
    if cut:
        data = dct_upsample(dct_cut_antialias(data,cutoff,electrode_spacing),factor=upsample)
    else:
        data = dct_upsample(data,factor=upsample)
    dz = array_phase_gradient(data).transpose((2,0,1))
    # extract curl via a kernal
    # take real component, centres have curl +- np.pi
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
    nanticlockwise = sum2(winding<-3)
    nsaddles   = sum2(saddles  )
    nmaxima    = sum2(maxima   )
    nminima    = sum2(minima   )
    return nclockwise, nanticlockwise, nsaddles, nmaxima, nminima

def array_phasegradient_local(frame,electrode_spacing=0.4):
    '''
    The average local gradient magnitude provides an upper bound on 
    spatial frequency ( lower bound on wavelength ).

    Parameters
    ----------
    frame : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
    electrode_spacing : float, default 0.4
        Spacing between electrodes in array in mm. Default is 0.4mm for
        the Utah arrays
        
    Returns
    -------
    np.array
        Average phase gradient magniture across array, in cycles/mm
    '''
    if len(frame.shape)<2:
        raise ValueError('Array data should be packed as (x,y,time)')
    pg = array_phase_gradient(frame)
    return np.mean(np.abs(pg),axis=(0,1))/(electrode_spacing*2*np.pi)

def array_phasegradient_planar(frame,electrode_spacing=0.4):
    '''
    The magnitude of the average gradient provides an accurate estimate
    of plane-wave wavelength even in the presence of noise. 
    
    Parameters
    ----------
    frame : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
    electrode_spacing : float, default 0.4
        Spacing between electrodes in array in mm. Default is 0.4mm for
        the Utah arrays
        
    Returns
    -------
    np.array
        Magnitude of average phase gradient across array, in cycles/mm
    '''
    if len(frame.shape)<2:
        raise ValueError('Array data should be packed as (x,y,time)')
    pg = array_phase_gradient(frame)
    return np.abs(np.mean(pg,axis=(0,1)))/(electrode_spacing*2*np.pi)

def array_phasegradient_magnitude_sigma(frame):
    '''
    Assess standard deviation of phase gradient magnitudes

    Parameters
    ----------
    frame : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
        
    Returns
    -------
    np.array
        The standard-deviation of the absolute magnitude of the 
        local phase gradient across an array, in cycles/mm
    '''
    if len(frame.shape)<2:
        raise ValueError('Array data should be packed as (x,y,time)')
    pg = array_phase_gradient(frame)
    return np.std(np.abs(pg),axis=(0,1))

def array_phasegradient_magnitude_cv(frame):
    '''
    Coefficient of variation of the magnitudes of the phase gradients

    Parameters
    ----------
    frame : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
        
    Returns
    -------
    np.array
        Coefficient of variation of the magnitudes of the phase gradients, 
        in cycles/mm
    '''
    if len(frame.shape)<2:
        raise ValueError('Array data should be packed as (x,y,time)')
    pg = array_phase_gradient(frame)
    return np.std(np.abs(pg),axis=(0,1))/np.mean(np.abs(pg),
        axis=(0,1))

def array_phasegradient_pgd_threshold(frame,thresh=0.5,electrode_spacing=0.4):
    '''
    The magnitude of the average gradient provides an accurate estimate
    of wavelength even in the presence of noise.

    Waves with 
    phase-gradient directionlity below threshold will be removed to
    reduce the contribution of noise to wavelength estimates.

    Parameters
    ----------
    frame : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
        
    Other Parameters
    ----------------
    thresh : float, default 0.5
        The minimum phase-gradient directionality measure. Waves with 
        phase-gradient directionlity below threshold will be removed to
        further reduce the contribution of noise to wavelength estimates.
    electrode_spacing : float, default 0.4
        Spacing between electrodes in array in mm. Default is 0.4mm for
        the Utah arrays
        
    Returns
    -------
    np.array
        Magnitude of average phase gradient across array, in cycles/mm
        for every time-point
    '''
    if len(frame.shape)<2:
        raise ValueError('Array data should be packed as (x,y,time)')
    pg  = array_phase_gradient(frame)
    use = array_synchrony_pgd(frame)>=thresh
    pg[:,:,~use] = np.NaN
    return np.abs(np.mean(pg,axis=(0,1)))/(electrode_spacing*2*np.pi)

def array_wavelength_pgd_threshold(frame,thresh=0.5):
    '''
    The magnitude of the average gradient provides a very accurate estimate
    of wavelength even in the presence of noise. We can further restrict
    analysis to wavelength with a phase-gradient directionality above
    a certain threshold.

    Parameters
    ----------
    frame : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y).
        
    Other Parameters
    ----------------
    thresh : float, default 0.5
        The minimum phase-gradient directionality measure. Waves with 
        phase-gradient directionlity below threshold will be removed to
        further reduce the contribution of noise to wavelength estimates.
        
    Returns
    -------
    np.array
        wavelength in mm/cycle for wave events above the PGD threshold
    '''
    if len(frame.shape)<2:
        raise ValueError('Array data should be packed as (x,y,time)')
    return 1/array_phasegradient_pgd_threshold(frame,thresh)


def array_wavelength_local_pgd_threshold(frame,thresh=0.5,electrode_spacing=0.4):
    '''
    The average phase gradient magnitude can tolerate non-planar waves,
    but is sensitive to noise. It may be appropriate to combine
    this method with spatial smoothing to denoise the data, if it is safe
    to assume a minimum spatial scale for the underlying wave dynamics.
    
    To be conservative, this routine excludes waves with phase-gradient
    directionality below the provided threshold.

    Parameters
    ----------
    frame : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
        
    Other Parameters
    ----------------
    thresh : float, default 0.5
        The minimum phase-gradient directionality measure. Waves with 
        phase-gradient directionlity below threshold will be removed to
        further reduce the contribution of noise to wavelength estimates.
    electrode_spacing : float, default 0.4
        Spacing between electrodes in array in mm. Default is 0.4mm for
        the Utah arrays
        
    Returns
    -------
    np.array
        wavelength in mm/cycle for wave events above the PGD threshold
    '''
    if len(frame.shape)<2:
        raise ValueError('Array data should be packed as (x,y,time)')
    pg  = array_phase_gradient(frame)
    use = array_synchrony_pgd(frame)>=thresh
    pg[:,:,~use] = np.NaN
    return 1/np.mean(abs(pg),axis=(0,1))/(electrode_spacing*2*np.pi)


def array_speed_pgd_threshold(frame,thresh=0.5,FS=1000.0):
    '''
    Calculate an estimated wave speed for waves above the given PGD 
    threshold.

    Parameters
    ----------
    frame : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
        
    Other Parameters
    ----------------
    thresh : float, default 0.5
        The minimum phase-gradient directionality measure. Waves with 
        phase-gradient directionlity below threshold will be removed to
        further reduce the contribution of noise to wavelength estimates.
    electrode_spacing : float, default 0.4
        Spacing between electrodes in array in mm. Default is 0.4mm for
        the Utah arrays
        
    Returns
    -------
    np.array
        speed for plane waves in mm/s
    '''
    if len(frame.shape)<2:
        raise ValueError('Array data should be packed as (x,y,time)')
    pg = array_phasegradient_pgd_threshold(frame,thresh) #cycles / mm
    df = np.median(np.ravel(rewrap(np.diff(np.angle(frame),1,2)))) #radians/sample
    f  = df*FS # radians / s
    g  = f /(2*np.pi) # cycles / s
    return g/pg # mm /s

def array_speed_local(frame,FS=1000.0):
    '''
    Accuracy can be improved by restricting analysis to waves with high
    phase-gradient directionality, where the coherent propagation direction
    between channels allows for more reliable gradient estimates. High
    signal-to-noise ratio can also mitigate the impact of noise.

    Parameters
    ----------
    frame : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
        
    Other Parameters
    ----------------
    FS : float, default 1000
        Sampling rate
        
    Returns
    -------
    np.array
        speed for plane waves in mm/s
    '''
    if len(frame.shape)<2:
        raise ValueError('Array data should be packed as (x,y,time)')
    pg = array_phasegradient_local(frame) #cycles / mm
    df = np.median(np.ravel(rewrap(np.diff(np.angle(frame),1,2)))) #radians/sample
    f  = df*FS # radians / s
    g  = f /(2*np.pi) # cycles / s
    return g/pg # mm /s

def array_wavelength_local(frame):
    '''
    phase gradients are in units of radians per electrode

    Parameters
    ----------
    frame : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
        
    Returns
    -------
    np.array
        wavelengths in mm/cycle
    '''
    if len(frame.shape)<2:
        raise ValueError('Array data should be packed as (x,y,time)')
    return 1/array_phasegradient_local(frame)

def array_wavelength_planar(frame):
    '''
    phase gradients are in units of radians per electrode

    Parameters
    ----------
    frame : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
        
    Returns
    -------
    np.array
        wavelengths in mm/cycle
    '''
    if len(frame.shape)<2:
        raise ValueError('Array data should be packed as (x,y,time)')
    return 1/array_phasegradient_planar(frame)

def array_synchrony_pgd(frame):
    '''
    The phase gradient directionality measure from Rubinto et al 2009 is
    abs(mean(pg))/mean(abs(pg))
    
    Parameters
    ----------
    frame : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
        
    Returns
    -------
    np.array
        Phase gradient directionlity measure for each timepoint
    '''
    if len(frame.shape)<2:
        raise ValueError('Array data should be packed as (x,y,time)')
    pg = array_phase_gradient(frame)
    return np.abs(np.mean(pg,axis=(0,1)))/np.mean(np.abs(pg),axis=(0,1))

def array_synchrony_pgd_standard_deviation(frame):
    '''
    The phase gradient directionality measure from Rubinto et al 2009 is
    abs(mean(pg))/mean(abs(pg)).
    
    This routine applies a nonlinear transformation so that PGD has units
    of radians, for easier interpretation.

    Parameters
    ----------
    frame : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
        
    Returns
    -------
    np.array
        Phase gradient directionality, transformed to 
        have units of radians in analogy to the standard deviation of a
        circularly wrapped normal distribution. 
    '''
    if len(frame.shape)<2:
        raise ValueError('Array data should be packed as (x,y,time)')
    R = array_synchrony_pgd(frame)
    return np.sqrt(-2*np.log(R))

def array_kuramoto_pgd(frame):
    '''
    A directionality index based on the Kuramoto order parameter. 
    Phase gradient magnitude is discarded.

    Parameters
    ----------
    frame : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
        
    Returns
    -------
    np.array
        Kuramoto gradient directionlity measure for each timepoint
    '''
    pg = array_phase_gradient(frame)
    return np.abs(np.mean(pg/np.abs(pg),axis=(0,1)))

def array_kuramoto_pgd_standard_deviation(frame):
    '''
    A directionality index based on the Kuramoto order parameter. 
    Phase gradient magnitude is discarded.
    
    The resulting order parameter is nonlinearly transformed to have 
    units of radians in analogy to the standard deviation of a 
    circularly wrapped normal distribution.

    Parameters
    ----------
    frame : np.array
        ND numpy array of complex-valued signals with phase and amplitude.
        Must be at least 2D. The first 2 dimensions are spatial
        dimensions (x,y). 
        
    Returns
    -------
    np.array
        Kuramoto gradient directionlity measure for each timepoint, transformed
        to units of radians in analogy to the standard deviation of a
        circularly wrapped normal distribution. 
    '''
    pg = array_phase_gradient(frame)
    return np.abs(np.mean(pg/np.abs(pg),axis=(0,1)))

def trim_array(arrayMap):
    '''
    Removes any rows or columns from the array map
    that are empty ( have no channels )

    Parameters
    ----------
    arrayMap : np.array
        Array map of channel locations. -1 marks missing channels
        
    Returns
    -------
    np.array
        Trimmed map: any rows or columns that are entirely -1 are removed.
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
    empty ( have no channels ) in the arrayMap.
    
    This routine is useful for cropping larger arrays down to smaller 
    sizes, to verify that the statistical estimates do not depend on 
    the array size.

    Parameters
    ----------
    arrayMap : np.array
        Array map of channel locations. -1 marks missing channels
    data : np.array
        A 3-dimensional np.array
        where the first two dimensions are the (x,y) spatial 
        dimensions of the multi-electrode array.
        
    Returns
    -------
    np.array
        the data array input, with rows or columns removed where arrayMap
        is missing channels.
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

    Parameters
    ----------
    data: np.array
        NChannel x Ntimes
    arrayMap : np.array
        array map, 1-indexed, 0 for missing electrodes
        
    Returns
    -------
    np.array
        LxKxNtimes 3D array of the interpolated channel data
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
