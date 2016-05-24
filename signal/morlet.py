#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division

from neurotools.plot    import *
from neurotools.getfftw import *
from neurotools.tools   import memoize
from scipy.signal.wavelets import morlet

def normalized_morlet(m,w):
    '''
    See morlet(m,w)
    
    This applies post-processing such that the sum absolute magnitued of 
    the wavelet is 1
    '''
    wl = morlet(m,w)
    return wl/sum(abs(wl))

@memoize
def prepare_wavelet_fft_basis(fa,fb,resolution,L,w,Fs):
    ''' Fs=1000. '''
    freqs = arange(fa,fb+1,resolution)
    M     = 2.*1.*w*Fs/freqs
    # we have to be careful about padding. In general we want the wavelets
    # to all be centered in the array before we take the FFT, so the padding
    # should be symmetric
    # Modification: now normalizing wavelet amplitude to 1
    allwl = []
    fftwl = []
    for i,m in enumerate(M):
        wl = normalized_morlet(m,w)
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

def fft_cwt(beta,fa,fb,w=4.0,resolution=0.1,Fs=1000.0):
    '''
    beta is data, should be Ntimes x NCH
    can do multiple at once!
    
    returns Nch x Nfreq x Ntimes
    '''
    fa,fb = map(float,(fa,fb)) # integer math was causing bugs
    if len(shape(beta))==1:
        beta = reshape(beta,shape(beta)+(1,))
    N,NCH = shape(beta)
    if NCH>N:
        warn('MORE CHANNELS THAN DATA CHECK FOR TRANSPOSED')
        warn('YOU KNOW WHAT IM JUST GOING TO FLIP IT FOR YOU')
        beta = beta.T
        N,NCH=NCH,N
    padded       = zeros((N*2,NCH),dtype=complex64)
    padded[:N,:] = beta
    padded[N:,:] = beta[::-1,:]
    fft_data = fft(padded,axis=0,threads=24)
    freqs,wavelets = prepare_wavelet_fft_basis(fa,fb,resolution,N*2,w,float(Fs))
    result   = array([ifft(fft_data.T*wl,axis=1,threads=24)[:,:N] for wl in wavelets])
    return freqs,transpose(result,(1,0,2))

def fft_cwt_transposed(data,fa,fb,w=4.0,resolution=0.1,Fs=1000.0,threads=1):
    '''
    Parameters
    ----------
    data : numeric
        NCH x Ntimes list of signals to transform
    
    Returns
    ------
    freqs : float
        frequencies
    result : wavelt transforms
        Nch x Nfreq x Ntimes
    '''
    fa,fb = map(float,(fa,fb)) # integer math was causing bugs
    if len(shape(data))==1:
        data = data[None,...]#reshape(data,(1,)+shape(data))
    NCH,N = shape(data)
    if NCH>N:
        warn('MORE CHANNELS THAN DATA CHECK FOR TRANSPOSED!!')
        #beta = beta.T
        #N,NCH=NCH,N
    padded       = zeros((NCH,N*2),dtype=complex64)
    padded[:,:N] = data
    padded[:,N:] = data[:,::-1]
    fft_data = fft(padded,axis=-1,threads=threads)
    freqs,wavelets = prepare_wavelet_fft_basis(fa,fb,resolution,N*2,w,float(Fs))
    result   = array([ifft(fft_data*wl,axis=1,threads=threads)[:,:N] for wl in wavelets])
    return freqs,transpose(result,(1,0,2))

def logfreqs(fa,fb,nfreq):
    freqs = 2**linspace(log2(fa),log2(fb),nfreq)
    return freqs

@memoize
def prepare_wavelet_fft_basis_logspace(fa,fb,nfreq,L,w,Fs):
    '''Fs=1000.'''
    freqs = logfreqs(fa,fb,nfreq)
    M     = 2.*1.*w*Fs/freqs
    # we have to be careful about padding. In general we want the wavelets
    # to all be centered in the array before we take the FFT, so the padding
    # should be symmetric
    # Modification: now normalizing wavelet amplitude to 1
    allwl = []
    fftwl = []
    for i,m in enumerate(M):
        wl = normalized_morlet(m,w)
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

def fft_cwt_transposed_logspaced(data,fa,fb,w=4.0,nfreqs=None,threads=1):
    '''
    data is NCH x Ntimes
    returns Nch x Nfreq x Ntimes
    '''
    assert 0 # TODO ADD FS
    if nfreqs is None: nfreqs = int(round(fb-fa))
    if len(shape(data))==1:
        data = reshape(data,(1,)+shape(data))
    NCH,N = shape(data)
    if NCH>N:
        warn('MORE CHANNELS THAN DATA CHECK FOR TRANSPOSED')
        beta = beta.T
        N,NCH=NCH,N
    padded       = zeros((NCH,N*2),dtype=complex64)
    padded[:,:N] = data
    padded[:,N:] = data[:,::-1]
    fft_data = fft(padded,axis=-1,threads=threads)
    freqs,wavelets = prepare_wavelet_fft_basis_logspace(fa,fb,nfreqs,N*2,w,1000.0)
    result   = array([ifft(fft_data*wl,axis=1,threads=threads)[:,:N] for wl in wavelets])
    return freqs,transpose(result,(1,0,2))


############################################################################
############################################################################
if __name__=='__main__':
    #wavelet power test
    signal = randn(1000)
    Fs=1000
    for freq in arange(5,500,5):
        ws   = arange(4,30)
        M    = 2.*1.*ws*Fs/float(freq)
        clf()
        x = []
        bw = []
        for m,w in zip(M,ws):
            wl = normalized_morlet(m,w)
            a,b = fftfreq(len(wl),1./Fs),abs(fft(wl))
            #plot(a,b)
            df = a[1]-a[0]
            bw.append(sum(b)*df)
            s = convolve(signal,wl,'same')
            #print m,w,mean(abs(s)),mean(abs(s)**2)
            x.append(var(s))

        bw = arr(bw)
        x = arr(x)
        plot(x/bw)
        positivey()
        print freq,1/mean(bw/x)




