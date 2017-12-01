#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Statistics routines to examine population statistics of phases and
complex-valued (phase,amplitude) analytic signals.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
from neurotools.signal.signal import rewrap

ELECTRODE_SPACING = 0.4

def population_kuramoto(pop,axis=0):
    '''
    Compute the Kuramoto order parameter of a population of complex-
    valued phase oscillators. 
    
    If an array is provided, the average is taken over the first array
    dimension unless
    otherwise specified.
    
    Parameters
    ----------
    pop : np.array
        Array of complex-valued phases of a population of phase oscillators.
        Expectations are taken over the first dimension only unless
        otherwise specified.
    
    Other Parameters
    ----------------
    axis : int, default 0
        Axis over which to operate
    
    Returns
    -------
    float or np.array
        Kuramotor order parameter of the population
    '''
    return np.abs(np.mean(pop/np.abs(pop),axis=axis))

def population_synchrony(pop,axis=0):
    '''
    Estimate phase-oscillator population synchrony. This is similar to 
    the Kuramoto order parameter, but weights each oscillator by its 
    amplitude. Since signal measurements typically have nonzero noise, 
    phase estimates from low-amplitude oscillators are less reliable. 
    
    Parameters
    ----------
    pop : np.array
        Array of complex-valued phases of a population of phase oscillators.
        Expectations are taken over the first dimension only unless
        otherwise specified.
    
    Other Parameters
    ----------------
    axis : int, default 0
        Axis over which to operate
    
    Returns
    -------
    float or np.array
        Oscillator synchrony over the population
    '''
    return np.abs(np.mean(pop,axis=axis))/np.mean(np.abs(pop),axis=axis)

def population_polar_std(pop,axis=0):
    '''
    The circular standard deviation of a collection of phase oscillators.
    This is a transformation of `population_synchrony` with units of 
    radians, which is easier to interpret. 
    
    Parameters
    ----------
    pop : np.array
        Array of complex-valued phases of a population of phase oscillators.
        Expectations are taken over the first dimension only unless
        otherwise specified.
    
    Other Parameters
    ----------------
    axis : int, default 0
        Axis over which to operate
    
    Returns
    -------
    float or np.array
        Circular standard deviation over the population
    '''
    return np.sqrt(-2*np.log(population_synchrony(pop,axis=axis)))

def population_average_amplitude(pop,axis=0):
    '''
    Compute the average absolute amplitude of a population of complex-
    valued phase oscillators.
    
    Parameters
    ----------
    pop : np.array
        Array of complex-valued phases of a population of phase oscillators.
        Expectations are taken over the first dimension only unless
        otherwise specified.
    
    Other Parameters
    ----------------
    axis : int, default 0
        Axis over which to operate
    
    Returns
    -------
    float or np.array
        Average amplitude over the population
    '''
    return np.mean(np.abs(pop),axis=axis)

def population_signal_dispersion(pop,axis=0):
    '''
    A standardized measure of the dispersion of a population of complex-
    valued phase oscillators. Computes the determinant of the covariance
    matrix describing the 2D distribution of (phases, amplitudes) in the
    complex plane, raised to the 1/4th power. This has the same units as
    the linear dimension of the signal. For example, if the underlying 
    signals are in mV, then this is a standard measure of how dispersed
    the analytic signals are in the complex plane, also in mV. 
    
    Parameters
    ----------
    pop : np.array
        Array of complex-valued phases of a population of phase oscillators.
        Expectations are taken over the first dimension only unless
        otherwise specified.
    
    Other Parameters
    ----------------
    axis : int, default 0
        Axis over which to operate
    
    Returns
    -------
    float or np.array
        Depending on whether a 1D or ND array was passed
        Standardized measure of dispersion
    '''
    r   = np.real(pop)
    i   = np.imag(pop)
    r   = r-np.mean(r,axis=axis)
    i   = i-np.mean(i,axis=axis)
    # Compute covariances
    Crr = np.mean(r*r,axis=axis)
    Cri = np.mean(r*i,axis=axis)
    Cii = np.mean(i*i,axis=axis)
    # Compute determinent
    Det = Crr*Cii-Cri**2
    return Det**0.25

def population_signal_concentration(pop,axis=0):
    '''
    Returns a standardized, unitless measure of the concentration of a
    population of complex-valued (phase,amplitude) oscillators. 
    This is analagous to the reciprocal of the coefficient of variation
    for univariate data. Larger values indicate more precise distributions.
    
    Parameters
    ----------
    pop : np.array
        Array of complex-valued phases of a population of phase oscillators.
        Expectations are taken over the first dimension only unless
        otherwise specified.
    
    Other Parameters
    ----------------
    axis : int, default 0
        Axis over which to operate
    
    Returns
    -------
    float or np.array
        Depending on whether a 1D or ND array was passed
        Standardized measure of concentration
    '''
    return population_average_amplitude(pop,axis=axis)/population_signal_dispersion(pop,axis=axis)

def population_signal_precision(pop,axis=0):
    '''
    Returns 1/σ where σ=population_signal_dispersion is a standardized
    measure of the dispersion in the population.
    
    Parameters
    ----------
    pop : np.array
        Array of complex-valued phases of a population of phase oscillators.
        Expectations are taken over the first dimension only unless
        otherwise specified.
        
    Other Parameters
    ----------------
    axis : int, default 0
        Axis over which to operate
    
    Returns
    -------
    float or np.array
        Depending on whether a 1D or ND array was passed
        Standardized measure of precision
    '''
    return population_signal_dispersion(pop,axis=axis)**(-1)

def population_signal_phase_dispersion(pop,axis=0):
    '''
    Coefficient of variation of phases, estimated using a local 
    linearization around the mean phase. 
    
    Parameters
    ----------
    pop : np.array
        Array of complex-valued phases of a population of phase oscillators.
        Expectations are taken over the first dimension only unless 
        otherwise specified.
        
    Other Parameters
    ----------------
    axis : int, default 0
        Axis over which to operate
    
    Returns
    -------
    float or np.array
        Depending on whether a 1D or ND array was passed
    '''
    # rotate into best-guess zero phase reference frame
    z = np.mean(pop,axis=axis)
    h = (z/np.abs(z))**-1
    x = pop*h
    # phase dispersion is CV=sig/mu deviation along the imaginary axis
    # the mean is always zero so we don't normalize it
    s = np.std (x.imag,axis=axis)
    m = np.mean(x.real,axis=axis)
    return s/m

def population_signal_phase_std(pop,axis=0):
    '''
    Standard deviation of phases locally linearized around the mean 
    phase
    
    Parameters
    ----------
    pop : np.array
        Array of complex-valued phases of a population of phase oscillators.
        Expectations are taken over the first dimension only unless
        otherwise specified.
        
    Other Parameters
    ----------------
    axis : int, default 0
        Axis over which to operate
    
    Returns
    -------
    float or np.array
        Depending on whether a 1D or ND array was passed
    '''
    # rotate into best-guess zero phase reference frame
    z = np.mean(pop,axis=axis)
    h = (z/np.abs(z))**-1
    x = pop*h
    # phase dispersion is CV=sig/mu deviation along the imaginary axis
    # the mean is always zero so we don't normalize it
    s = std (x.imag,axis=axis)
    return s

def population_signal_amplitude_std(pop,axis=0):
    '''
    Standard deviation of amplitudes locally linearized around the mean
    phase
    
    Parameters
    ----------
    pop : np.array
        Array of complex-valued phases of a population of phase oscillators.
        Expectations are taken over the first dimension only unless
        otherwise specified.
        
    Other Parameters
    ----------------
    axis : int, default 0
        Axis over which to operate
    
    Returns
    -------
    float or np.array
        Depending on whether a 1D or ND array was passed
    '''
    # rotate into best-guess zero phase reference frame
    z = np.mean(pop,axis=axis)
    h = (z/np.abs(z))**-1
    x = np.real(pop*h)
    # amplitude dispersion is CV=sig/mu deviation along the real axis
    s = np.std(x.real,axis=axis)
    return s

def population_signal_amplitude_dispersion(pop,axis=0):
    '''
    Coefficient of variation of amplitudes, locally linearized around
    the mean phase
    
    Parameters
    ----------
    pop : np.array
        Array of complex-valued phases of a population of phase oscillators.
        Expectations are taken over the first dimension only unless
        otherwise specified.
        
    Other Parameters
    ----------------
    axis : int, default 0
        Axis over which to operate
    
    Returns
    -------
    float or np.array
        Depending on whether a 1D or ND array was passed
    '''
    # rotate into best-guess zero phase reference frame
    z = np.mean(pop,axis=0)
    h = (z/np.abs(z))**-1
    x = np.real(pop*h)
    # amplitude dispersion is CV=sig/mu deviation along the real axis
    s = np.std (x,axis=0)
    m = np.mean(x,axis=0)
    return s/m

def population_signal_phase_precision(pop,axis=0):
    '''
    Inverse coefficient of variation of phases, estimated using a local 
    linearization around the mean phase. 
    
    Parameters
    ----------
    pop : pop of oscillator phases; 1D np.complex array
        
    Other Parameters
    ----------------
    axis : int, default 0
        Axis over which to operate
    
    Returns
    -------
    float or np.array
        Depending on whether a 1D or ND array was passed
    '''
    return 1./population_signal_phase_dispersion(pop,axis=axis)

def population_signal_amplitude_precision(pop,axis=0):
    '''
    Inverse coefficient of variation of amplitudes, estimated using a local 
    linearization around the mean phase. 
    
    Parameters
    ----------
    pop : np.array
        Array of complex-valued phases of a population of phase oscillators.
        Expectations are taken over the first dimension only unless
        otherwise specified.
        
    Other Parameters
    ----------------
    axis : int, default 0
        Axis over which to operate
    
    Returns
    -------
    float or np.array
        Depending on whether a 1D or ND array was passed
    
    '''
    return 1./population_signal_amplitude_dispersion(pop,axis=axis)

def population_signal_description(pop,axis=0):
    '''
    Returns a statistical of a population of complex-valued phase 
    oscillators.
    
    Parameters
    ----------
    pop : np.array
        Array of complex-valued phases of a population of phase oscillators.
        Expectations are taken over the first dimension only unless
        otherwise specified.
        
    Other Parameters
    ----------------
    axis : int, default 0
        Axis over which to operate
    
    Returns
    -------
    z : complex
        Average analytic signals
    a : float
        Magnitude of average analytic signal
    h : float
        Mean phase
    s1 : 
        Standard deviation of linearized phases
    s2 : 
        Standard deviation of linearized amplitudes
    '''
    z  = np.mean(pop,axis=axis)
    a  = np.abs(z)
    h  = np.angle(z)
    w  = pop*np.exp(-1j*h)
    s1 = np.std(np.imag(w),axis=axis)
    s2 = np.std(np.real(w),axis=axis)
    return 

def population_synchrony_linear(pop,axis=0):
    '''
    Transformed population synchrony score
    
    Parameters
    ----------
    pop : np.array
        Array of complex-valued phases of a population of phase oscillators.
        Expectations are taken over the first dimension only unless
        otherwise specified.
        
    Other Parameters
    ----------------
    axis : int, default 0
        Axis over which to operate
    
    Returns
    -------
    np.array
        1/(1-population_synchrony(pop))
    '''
    syn = population_synchrony(pop,axis=axis)
    return 1.0/(1.0-syn)

# we need a new measure of sliding coherence which extracts the median
# hilbert frequency and uses it to unwrap the phase. This is important
# because it will allow us to distinguish traveling waves from synch
# 

def population_phase_coherence(data):
    '''
    Extracts median frequency. Uses this to unwrap array phases.
    Applies a per-channel phase shift to zero mean phase. 

    Parameters
    ----------
    data : array-like
        No. of channels × No. of timepoints
        
    Returns
    -------
    np.array

    Example
    -------
    ::
    
        s,a,tr = ('SPK120918', 'M1', 16)
        pop  = get_all_analytic_pop(s,a,tr,epoch=(6,-1000,2001),fa=10,fb=45)
        data = pop[:,500:600]
        sliding = np.array([population_phase_coherence(pop[:,i:i+100]) for i in range(shape(pop)[1]-100)])
    '''
    if not len(data.shape)==2:
        raise ValueError('data should be a 2D np.array of phases, type np.complex')
    h    = np.angle(data)
    dfdt = (np.diff(h,axis=-1)+np.pi)%(2*np.pi)-np.pi
    mdf  = np.median(dfdt,axis=0)
    return np.mean(np.cos(dfdt-mdf),axis=0)

def analytic_signal_coherence(data,window=np.hanning):
    '''
    Extracts median frequency. Uses this to unwrap array phases.
    Applies a per-channel phase shift to zero mean phase. 

    Parameters
    ----------
    data : array-like
        K x N
        K = No. of channels
        N = No. of timepoints
    L : 
        Window length in samples, optional, default is 100
    window : function
        windowing function, optional, default is `np.hanning`
        
    Other Parameters
    ----------------
    axis : int, default 0
        Axis over which to operate
    
    Returns
    -------
    np.array
    
    Example
    -------
    ::
    
        s,a,tr = ('SPK120918', 'M1', 16)
        pop = get_all_analytic_pop(s,a,tr,epoch=(6,-1000,2001),fa=10,fb=45)
        data = pop[:,500:600]
        sliding = np.array([population_signal_coherence(pop[:,i:i+100]) for i in range(shape(pop)[1]-100)])
    '''
    if not len(data.shape)==2:
        raise ValueError('data should be a 2D np.array of phases, type np.complex')
    N = data.shape[-1]
    h = np.angle(data)
    dfdt     = (np.diff(h,axis=-1)+np.pi)%(2*np.pi)-np.pi
    mdf      = np.median(dfdt,axis=0)
    weights  = np.abs(data)[:,:-1]
    weights  = weights*window(N+1)[1:-1]
    weights /= np.sum(weights)
    return np.sum(np.cos(dfdt-mdf)*weights,axis=0)

def population_sliding_signal_coherence(data,L=100,window=np.hanning):
    '''
    Extracts median frequency. Uses this to unwrap array phases.
    Applies a per-channel phase shift to zero mean phase. 

    Parameters
    ----------
    data : array-like
        K x N
        K = No. of channels
        N = No. of timepoints
        
    Other Parameters
    ----------------
    axis : int, default 0
        Axis over which to operate
    L : 
        Window length in samples, optional, default is 100
    window : function
        windowing function, optional, default is `np.hanning`
    
    Returns
    -------
    np.array : 
        Sliding-window Kuramoto order parameter over data
    '''
    if not len(data.shape)==2:
        raise ValueError('data should be a 2D np.array of phases, type np.complex')
    N       = data.shape[-1]
    h       = np.angle(data)
    dfdt    = (np.diff(h,axis=-1)+np.pi)%(2*np.pi)-np.pi
    weights = np.abs(data)
    win     = window(L+2)[1:-1]
    slide   = []
    for i in range(0,N-L-1):
        now = dfdt[...,i:i+L]
        mdf = np.median(now)
        w   = weights[...,i:i+L]*win
        w  /= np.sum(w)
        slide.append(np.sum(np.cos(now-mdf)*w))
    return np.array(slide)

sliding_population_signal_coherence = population_sliding_signal_coherence

def population_normalized_sliding_signal_coherence(data,L=100,window=np.hanning):
    '''
    Extracts median frequency. Uses this to unwrap array phases.
    Applies a per-channel phase shift to zero mean phase. 

    Parameters
    ----------
    data : array-like
        K x N
        K = No. of channels
        N = No. of timepoints
    L : 
        Window length in samples, optional, default is 100
    window : function
        windowing function, optional, default is `np.hanning`
        
    Other Parameters
    ----------------
    axis : int, default 0
        Axis over which to operate
    
    Returns
    -------
    np.array
    '''
    if not len(data.shape)==2:
        raise ValueError('data should be a 2D np.array of phases, type np.complex')
    N       = data.shape[-1]
    h       = np.angle(data)
    dfdt    = (np.diff(h,axis=-1)+np.pi)%(2*np.pi)-np.pi
    weights = np.abs(data)
    win     = window(L+2)[1:-1]
    slide   = []
    for i in range(0,N-L-1):
        now = dfdt[...,i:i+L]
        mdf = np.median(now)
        w   = weights[...,i:i+L]*win
        w  /= np.sum(w)
        mu,sig = weighted_avg_and_std( (now-mdf)/mdf, w)
        slide.append(sig)
    return 1/np.sqrt(1+np.array(slide)**2)

def population_phase_relative_sliding_kuramoto(data,L=100,window=np.hanning):
    '''
    Uses the phase of each channel in the middle of each block as a
    reference point. Separates coherent wave activity from synchrony.

    .. math::
    
        \\textrm{kuramoto order} = \left\langle z/|z| \\right\\rangle
    
    Assumes constant phase velocity, and a constant per-channel
    phase shift, and then computes the order. This is a notion of 
    relative phase stability.
    
    Parameters
    ----------
    data : np.array
        Phase data as np.complex
        
    Other Parameters
    ----------------
    L : int, default is 100
        Window length in samples
    window : function, default is `np.hanning`
        windowing function, optional
        
    Returns
    -------
    np.array
    
    '''
    if not len(data.shape)==2:
        raise ValueError('data should be a 2D np.array of phases, type np.complex')
    K,N     = data.shape
    phases  = np.angle(data)
    phased  = data/np.abs(data)
    dfdt    = (np.diff(phases,axis=-1)+np.pi)%(2*np.pi)-np.pi
    win     = window(L+2)[1:-1]
    win     = win/(K*np.sum(win))
    slide   = []
    for i in range(0,N-L-1):
        # get the median phase velocity
        mf = np.median(dfdt[...,i:i+L])
        # get the local phases
        x  = phases[...,i:i+L]
        # dephase the signal in time
        y  = rewrap(x-np.cumsum([mf]*L))
        # dephase the signal per channel
        z  = np.angle(np.mean(exp(1j*y),axis=-1))
        rephased = (x.T-z).T
        # compute weighted phase order
        slide.append(np.abs(np.sum(exp(1j*rephased)*win)))
    return np.array(slide)

def population_median_phase_velocity(data):
    '''
    median phase velocity.
    
    Parameters
    ----------
    data : np.array
        2D array of phases; Nchannels x Ntimes
        
    Returns
    -------
    medv : np.array
        Median phase velocity within the population for every time point
    '''
    if not len(data.shape)==2:
        raise ValueError('data should be a 2D np.array of phases, type np.complex')
    N       = data.shape[-1]
    h       = np.angle(data)
    dfdt    = (np.diff(h,axis=-1)+np.pi)%(2*np.pi)-np.pi
    medv    = np.median(dfdt,0)
    return medv

def population_median_frequency(data,Fs=1000):
    '''
    Convert from phase in radians/frame to
    Frequency in cycles/s
    
    Parameters
    ----------
    data : np.complex array
        Phase array data
    Fs : int, default 1000
        Sampling rate in Hz
        
    Returns
    -------
    np.array
    '''
    medv = population_median_phase_velocity(data)
    return medv*(Fs/(2*np.pi))

