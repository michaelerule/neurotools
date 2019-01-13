#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

try:
    from future.utils import raise_with_traceback
    from future.utils import raise_from
    from future.utils import with_metaclass
    from builtins import str
    from builtins import int
except:
    pass

#from neurotools.tools import memoize
from neurotools.jobs.ndecorator import memoize

# from plottools import *
############################################################################
# Configure the beta band complex morlet transform

@memoize
def prepare_wavelet_fft_basis(fa,fb,resolution,L,w,Fs=1000):
    freqs = arange(fa,fb,resolution)
    M     = 2.*1.*w*Fs/freqs
    # we have to be careful about padding. In general we want the wavelets
    # to all be centered in the array before we take the FFT, so the padding
    # should be symmetric
    #
    # Modification: now normalizing wavelet amplitude to 1
    allwl = []
    fftwl = []
    for i,m in enumerate(M):
        wl = morlet(m,w)
        wl = wl/sum(abs(wl))
        N = len(wl)
        if N>L:
            chop_begin = (N-L)/2
            chop_end   = (N-L)-chop_begin
            wl = wl[chop_begin:-chop_end]
            N = len(wl)
            assert N==L
        padded = zeros(L,dtype=complex64)
        start  = (L-N)/2
        padded[start:start+N]=wl
        reordered = zeros(L,dtype=complex64)
        reordered[L/2:]=padded[:L/2]
        reordered[:L/2]=padded[L/2:]
        allwl.append(reordered)
        fftwl.append(fft(reordered))
    return freqs,array(fftwl)

def fft_cwt(beta,fa,fb,w=4.0,resolution=0.1,Fs=1000):
    '''
    input is n x nch array
    returned is ... nch x nfreq x ntimes
    '''
    if len(shape(beta))==1:
        beta = reshape(beta,shape(beta)+(1,))
    N,NCH        = shape(beta)
    if NCH>N:
        print('MORE CHANNELS THAN DATA CHECK FOR TRANSPOSED')
    padded       = zeros((N*2,NCH),dtype=complex64)
    padded[:N,:] = beta
    padded[N:,:] = beta[::-1,:]
    fft_data = fft(padded,axis=0)
    freqs,wavelets = prepare_wavelet_fft_basis(fa,fb,resolution,N*2,w,Fs)
    result   = array([ifft(fft_data.T*wl,axis=1)[:,:N] for wl in wavelets])
    return freqs,transpose(result,(1,0,2))

def population_coherence_spectrum(lfp,fa,fb,w=4.0,resolution=0.1,Fs=1000):
    '''
    First dimension is nchannels, second is time.
    Use morlet wavelets ( essentially bandpass filter bank ) to compute
    short-timescale coherence.
    for each band: take morlet spectrum over time.
    take kuromoto or synchrony measure over complex vectors attained
    '''
    assert 0
    # this measures synchrony not coherence!
    freqs, transformed = fft_cwt(lfp.T,fa,fb,w,resolution,Fs)
    coherence = abs(mean(transformed,0))/mean(abs(transformed),0)
    return freqs, coherence

def population_eigencoherence(lfp,fa,fb,w=4.0,resolution=0.1,Fs=1000):
    '''
    not implemented.
    
    Uses the eigenvalue spectrum of the pairwise coherence matrix.
    In the case of wavelets, each time-frequency point has one
    complex value.

    The matrix we build: $|z_i z_j|$

    See ramirez et al
    A GENERALIZATION OF THE MAGNITUDE SQUARED COHERENCE SPECTRUM FOR
    MORE THAN TWO SIGNALS: DEFINITION, PROPERTIES AND ESTIMATION
    '''
    assert 0













def mtm_cohere():
    '''
    Multitaper coherence
    '''
