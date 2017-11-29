#!/usr/bin/python
# -*- coding: UTF-8 -*-
# BEGIN PYTHON 2/3 COMPATIBILITY BOILERPLATE
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
from neurotools.system import *

'''
Routines for calculating population summary statistics of LFP phases. 
'''

import numpy as np
from neurotools.signal.signal import rewrap

ELECTRODE_SPACING = 0.4

def population_kuromoto(population):
    '''
    Parameters
    ----------
    population : population of oscillator phases; 1D np.complex array
    
    Returns
    -------
    '''
    return np.abs(np.mean(population/np.abs(population),axis=0))

def population_synchrony(population):
    '''
    Parameters
    ----------
    population : population of oscillator phases; 1D np.complex array
    
    Returns
    -------
    '''
    return np.abs(np.mean(population,axis=0))/np.mean(np.abs(population),axis=0)

def population_polar_std(population):
    '''
    Parameters
    ----------
    population : population of oscillator phases; 1D np.complex array
    
    Returns
    -------
    '''
    return np.sqrt(-2*np.log(population_synchrony(population)))

def population_average_amplitude(population):
    '''
    Parameters
    ----------
    population : population of oscillator phases; 1D np.complex array
    
    Returns
    -------
    '''
    return np.mean(np.abs(population),axis=0)

def population_signal_concentration(lfp):
    '''
    Parameters
    ----------
    lfp : population of oscillator phases; 1D np.complex array
    
    Returns
    -------
    '''
    return population_average_amplitude(lfp)/population_signal_dispersion(lfp)**(-0.25)

def population_signal_precision(lfp):
    '''
    Parameters
    ----------
    lfp : population of oscillator phases; 1D np.complex array
    
    Returns
    -------
    '''
    return population_signal_dispersion(lfp)**(-1)

def population_signal_dispersion(lfp):
    '''
    Parameters
    ----------
    lfp : population of oscillator phases; 1D np.complex array
    
    Returns
    -------
    '''
    r = np.real(lfp)
    i = np.imag(lfp)
    r = r-np.mean(r,0)
    i = i-np.mean(i,0)
    Crr = np.mean(r*r,0)
    Cri = np.mean(r*i,0)
    Cii = np.mean(i*i,0)
    Det = Crr*Cii-Cri**2
    return Det**0.25

def population_signal_phase_dispersion(lfp):
    '''
    Parameters
    ----------
    lfp : population of oscillator phases; 1D np.complex array
    
    Returns
    -------
    '''
    # rotate into best-guess zero phase reference frame
    z = np.mean(lfp,axis=0)
    h = (z/np.abs(z))**-1
    x = lfp*h
    # phase dispersion is CV=sig/mu deviation along the imaginary axis
    # the mean is always zero so we don't normalize it
    s = np.std (np.imag(x),axis=0)
    m = np.mean(np.real(x),axis=0)
    return s/m

def population_signal_phase_std(lfp):
    '''
    Parameters
    ----------
    lfp : population of oscillator phases; 1D np.complex array
    
    Returns
    -------
    '''
    # rotate into best-guess zero phase reference frame
    z = np.mean(lfp,axis=0)
    h = (z/np.abs(z))**-1
    x = lfp*h
    # phase dispersion is CV=sig/mu deviation along the imaginary axis
    # the mean is always zero so we don't normalize it
    s = std (np.imag(x),axis=0)
    return s

def population_signal_amplitude_std(lfp):
    '''
    Parameters
    ----------
    lfp : population of oscillator phases; 1D np.complex array
    
    Returns
    -------
    '''
    # rotate into best-guess zero phase reference frame
    z = np.mean(lfp,axis=0)
    h = (z/np.abs(z))**-1
    x = np.real(lfp*h)
    # amplitude dispersion is CV=sig/mu deviation along the real axis
    s = std (x,axis=0)
    return s

def population_signal_amplitude_dispersion(lfp):
    '''
    Parameters
    ----------
    lfp : population of oscillator phases; 1D np.complex array
    
    Returns
    -------
    '''
    # rotate into best-guess zero phase reference frame
    z = np.mean(lfp,axis=0)
    h = (z/np.abs(z))**-1
    x = np.real(lfp*h)
    # amplitude dispersion is CV=sig/mu deviation along the real axis
    s = std (x,axis=0)
    m = np.mean(x,axis=0)
    return s/m

def population_signal_phase_precision(lfp):
    '''
    Parameters
    ----------
    lfp : population of oscillator phases; 1D np.complex array
    
    Returns
    -------
    '''
    return 1./population_signal_phase_dispersion(lfp)

def population_signal_amplitude_precision(lfp):
    '''
    Parameters
    ----------
    lfp : population of oscillator phases; 1D np.complex array
    
    Returns
    -------
    '''
    return 1./population_signal_amplitude_dispersion(lfp)

def population_signal_description(lfp):
    '''
    Parameters
    ----------
    lfp : population of oscillator phases; 1D np.complex array
    
    Returns
    -------
    '''
    z = np.mean(lfp,0)
    a = np.abs(z)
    h = np.angle(z)
    w = lfp*exp(-1j*h)
    s1 = np.std(np.imag(w),0)
    s2 = np.std(np.real(w),0)
    return z,a,h,s1,s2

def population_synchrony_linear(population):
    '''
    Parameters
    ----------
    population : population of oscillator phases; 1D np.complex array
    
    Returns
    -------
    1/(1-population_synchrony(population))
    '''
    syn = population_synchrony(population)
    return 1/(1-syn)

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
        K x N
        K = No. of channels
        N = No. of timepoints
    
    Returns
    -------

    Example
    -------
        s,a,tr = ('SPK120918', 'M1', 16)
        lfp = get_all_analytic_lfp(s,a,tr,epoch=(6,-1000,2001),fa=10,fb=45)
        data = lfp[:,500:600]
        sliding = np.array([population_phase_coherence(lfp[:,i:i+100]) for i in range(shape(lfp)[1]-100)])
    '''
    if not len(data.shape)==2:
        raise ValueError('data should be a 2D np.array of phases, type np.complex')
    h = np.angle(data)
    dfdt = (np.diff(h,axis=-1)+np.pi)%(2*np.pi)-np.pi
    mdf = np.median(dfdt,axis=0)
    return np.mean(cos(dfdt-mdf),axis=0)

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
    
    Returns
    -------
    
    Example
    -------
        s,a,tr = ('SPK120918', 'M1', 16)
        lfp = get_all_analytic_lfp(s,a,tr,epoch=(6,-1000,2001),fa=10,fb=45)
        data = lfp[:,500:600]
        sliding = np.array([population_signal_coherence(lfp[:,i:i+100]) for i in range(shape(lfp)[1]-100)])
    '''
    if not len(data.shape)==2:
        raise ValueError('data should be a 2D np.array of phases, type np.complex')
    N = data.shape[-1]
    h = np.angle(data)
    dfdt = (np.diff(h,axis=-1)+np.pi)%(2*np.pi)-np.pi
    mdf = np.median(dfdt,axis=0)
    weights = np.abs(data)[:,:-1]
    weights = weights*window(N+1)[1:-1]
    weights /= np.sum(weights)
    return np.sum(cos(dfdt-mdf)*weights,axis=0)

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
        slide.append(np.sum(cos(now-mdf)*w))
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
    
    Returns
    -------
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

def population_phase_relative_sliding_kuromoto(data,L=100,window=np.hanning):
    '''
    Uses the phase of each channel in the middle of each block as a
    reference point. Separates coherent wave activity from synchrony.

    .. math::
    
        \\textrm{kuromoto order} = \left\langle z/|z| \\right\\rangle
    
    Assumes constant phase velocity, and a constant per-channel
    phase shift, and then computes the order. This is a notion of 
    relative phase stability.
    
    Parameters
    ----------
    data : np.array
        Phase data as np.complex
    L : 
        Window length in samples, optional, default is 100
    window : function
        windowing function, optional, default is `np.hanning`
    
    Returns
    -------
    
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
    '''
    medv = population_median_phase_velocity(data)
    return medv*(Fs/(2*np.pi))





