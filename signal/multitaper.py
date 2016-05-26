#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from numpy import *

try:
    from spectrum.mtm import dpss
except:
    print('Could not locate the spectrum module, please install it')
    print('Multitaper methods will not work')
    def dpss(*args):
        raise NotImplementedError("Please install the spectrum module")

from neurotools.getfftw import *
from neurotools.signal.signal import zscore
from neurotools.jobs.decorator import memoize

try:
    import nitime
    from nitime.algorithms import coherence
except:
    print('THE "nitime" MODULE IS MISSING')
    print('> sudo easy_install nitime')
    print('(coherence function is undefined)')
    print('(none of the multitaper coherence functions will work)')

@memoize
def dpss_cached(length,half_bandwidth_parameter):
    '''
    Get a collection of DPSS tapers.
    N-tapers = half_bandwidth_parameter*2

    For legacy reasons this also transposes the tapes returned, such that
    the first dimension indexes tapers rather than time.
    '''
    tapers,eigen = dpss(length,half_bandwidth_parameter)
    return tapers.T,eigen

def multitaper_spectrum(x,k,Fs=1000.0,nodc=True):
    '''
    x : signal
    k : number of tapers
    Fs: sample rate (default 1K)
    returns frequencies, average sqrt(power) over tapers.
    '''
    N = shape(x)[-1]
    if nodc:
        x = x-mean(x,axis=-1)[...,None]
    tapers, eigen = dpss_cached(N,0.4999*k)
    specs = [abs(fft(x*t)) for t in tapers]
    freqs = fftfreq(N,1./Fs)
    return freqs[:N/2],mean(specs,0)[...,:N/2]

def multitaper_squared_spectrum(x,k,Fs=1000.0,nodc=True):
    '''
    x : signal
    k : number of tapers
    Fs: sample rate (default 1K)
    returns frequencies, average sqrt(power) over tapers.
    '''
    N = shape(x)[-1]
    if nodc:
        x = x-mean(x,axis=-1)[...,None]
    tapers, eigen = dpss_cached(N,0.4999*k)
    specs = [abs(fft(x*t)) for t in tapers]
    freqs = fftfreq(N,1./Fs)
    return freqs[:N/2],mean(specs,0)[...,:N/2]**2

def sliding_multitaper_spectrum(x,window=500,step=100,Fs=1000,BW=5):
    '''
    NOT IMPLEMENTED
    '''
    raise NotImplementedError("This function is not yet implemented")
