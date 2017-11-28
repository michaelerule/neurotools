#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

#from numpy import *
import numpy as np

try:
    from spectrum.mtm import dpss
except:
    print('could not locate the spectrum module; multitaper methods missing')
    def dpss(*args):
        raise NotImplementedError("Please install the spectrum module to use multi-taper methods")

from neurotools.getfftw import *
from neurotools.signal.signal import zscore
from neurotools.jobs.decorator import memoize

@memoize
def dpss_cached(length,half_bandwidth_parameter):
    '''
    Get a collection of DPSS tapers. The number of tapers equals the
    half bandwidth parameter times two. For legacy reasons the tapers
    are returned transposed such that the first dimension indexes
    tapers rather than time.

    The advantage of using this function is that computing DPSS is
    expensive. This function caches the results in RAM.

    Parameters
    ----------
    length : integer
        length of the domain for which to compute the DPSS
    half_bandwidth_parameter : number
        The number of is the half_bandwidth_parameter*2

    Returns
    -------
    ndarray:
        tapers.T a transposed list of DPSS tapers
    ndarray:
        taper eigenvalues ( weights )
    '''
    tapers,eigen = dpss(int(length),half_bandwidth_parameter)
    return tapers.T,eigen

def multitaper_spectrum(x,k,Fs=1000.0,nodc=True):
    '''
    Parameters
    ----------
    x : ndarray
        Signal to use; spectrum taken over last dimension
    k : int (positive)
        number of tapers (positive)
    Fs: int
        sample rate in Hz (default 1000)

    Returns
    -------
    ndarray
        frequencies,
    ndarray
        average sqrt(power) over tapers.
    '''
    N = x.shape[-1]
    if nodc:
        x = x-np.mean(x,axis=-1)[...,None]
    tapers, eigen = dpss_cached(N,0.4999*k)
    specs = [np.abs(fft(x*t)) for t in tapers]
    freqs = fftfreq(N,1./Fs)
    return freqs[:N//2],np.mean(specs,0)[...,:N//2]

def multitaper_squared_spectrum(x,k,Fs=1000.0,nodc=True):
    '''
    Parameters
    ----------
    x : ndarray
        Signal to use
    k : int (positive)
        number of tapers (positive)
    Fs: int
        sample rate in Hz (default 1000)

    Returns
    -------
    ndarray
        frequencies,
    ndarray
        average squared power over tapers.
    '''
    N = np.shape(x)[-1]
    if nodc:
        x = x-np.mean(x,axis=-1)[...,None]
    tapers, eigen = dpss_cached(N,0.4999*k)
    specs = [np.abs(fft(x*t)) for t in tapers]
    freqs = fftfreq(N,1./Fs)
    return freqs[:N//2],np.mean(specs,0)[...,:N//2]**2

def sliding_multitaper_spectrum(x,window=500,step=100,Fs=1000,BW=5):
    '''
    NOT IMPLEMENTED
    '''
    raise NotImplementedError("This function is not yet implemented")
