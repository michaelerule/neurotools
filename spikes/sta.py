#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Functions for spike-triggered statistics
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function



############################################################
# STA routines TODO move to spikes subpackage

def get_snips(signal,times,window):
    '''
    Extract snippits of a time series surronding a list of 
    times. Typically used for spike-triggered statistics
    
    Parameters
    ----------
    signal: 1D np.array
        Timseries to extract snips from.
    times: 1D np.int32
        Indecies of spiking events (samples) in `signal`
    window: positive int
        A region of size `2*window+1` will be extracted
        around each spike time.
    
    Returns
    -------
    snips: NSPIKES×(2*window+1) np.array
        Extracted spike-triggered signal snippits.
    '''
    times = times[times>window]
    times = times[times<len(signal)-window-1]
    snips = np.array([
        signal[t-window:t+window+1] for t in times])
    return snips

def triggered_average(signal,times,window):
    '''
    Calculate spike-triggered average of a signal.
    
    Parameters
    ----------
    signal: 1D np.array
        Timseries to extract snips from.
    times: 1D np.int32
        Indecies of spiking events (samples) in `signal`
    window: positive int
        A region of size `2*window+1` will be extracted
        around each spike time.
    
    Returns
    -------
    STA: length 2*window+1 np.array
        Spike-triggered average of `signal`.
    '''
    return np.mean(get_snips(signal,times,window),0)

def get_triggered_stats(signal,times,window):
    '''
    Get a statistical summary of data in length window 
    around time points.
    
    Parameters
    ----------
    signal: 1D np.array
        Timseries to extract snips from.
    times: 1D np.int32
        Indecies of spiking events (samples) in `signal`
    window: positive int
        A region of size `2*window+1` will be extracted
        around each spike time.
        
    Returns
    -------
    means : 
        Means of `signal` for all time-windows 
        specified in `times`.
    standard-deviations : 
        Standard deviation of `signal` for all time-windows 
        specified in `times`.
    standard-erros : 
        Standard errors of the mean of `signal` for all 
        time-windows specified in `times`.
    '''
    s = get_snips(signal,times,window)
    return np.mean(s,0),np.std(s,0),np.std(s,0)/np.sqrt(len(times))*1.96
    
def fftsta(spikes,x):
    '''
    Spike triggerd average (STA) via FFT
    Signal `x` is z-scored befor calculating the spike-triggered average
    (a.k.a. reverse correlation).
    The returned STA is normalized so that the maximum magnitude is 1.
    
    Parameters
    ----------
    spikes: np.array
        1D spike count vector
    x: np.array
        
    Returns
    -------
    np.float32 : normalized spike-triggered average
    '''
    signal = np.float32((x-np.mean(x))/np.std(x))
    spikes = np.float32(spikes)
    sta    = fftshift(ifft(fft(spikes,axis=1)*\
                    np.conj(fft(x),dtype=np.complex64)),axes=1).real
    return sta/np.max(abs(sta),axis=1)[:,None]
    
