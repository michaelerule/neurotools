
from numpy import *
from neurotools.getfftw import *
from spectrum.mtm import pmtm,dpss

try:
    import nitime
    from nitime.algorithms import coherence
except:
    print 'THE "nitime" MODULE IS MISSING'
    print '> sudo easy_install nitime'
    print '(coherence function is undefined)'
    print '(none of the multitaper coherence functions will work)'

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
    tapers, eigen = dpss(N,0.4999*k,k)
    specs = [abs(fft(x*t)) for t in tapers.T]
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
    tapers, eigen = dpss(N,0.4999*k,k)
    specs = [abs(fft(x*t)) for t in tapers.T]
    freqs = fftfreq(N,1./Fs)
    return freqs[:N/2],mean(specs,0)[...,:N/2]**2


def sliding_multitaper_spectrum(x,window=500,step=100,Fs=1000,BW=5):
    '''
    NOT IMPLEMENTED
    '''
    assert 0

def multitaper_coherence(x,y,Fs=1000,BW=5):
    '''
    multitaper_coherence(x,y,Fs=1000,BW=5)
    BW is the multitaper bandwidth
    returns freqs, cohere
    '''
    x -= mean(x)
    y -= mean(y)
    method = {'this_method':'multi_taper_csd','BW':BW,'Fs':Fs}
    freqs,cohere = coherence(array([x,y]),method)
    N = len(x)
    freqs = abs(fftfreq(N,1./Fs)[:N/2+1])
    return freqs, cohere[0,1]

def sliding_multitaper_coherence(x,y,window=500,step=100,Fs=1000,BW=5):
    '''
    Sliding multitaper coherence between x and y
    Not implemented / not supported
    '''
    N = len(x)
    assert len(y)==N
    allcohere = []
    for tstart in xrange(0,N-window+1,step):
        ff,cohere = multitaper_coherence(x[tstart:tstart+window],y[tstart:tstart+window],Fs,BW)
        allcohere.append(cohere)
    return ff,array(allcohere)

def mtmchpar((t,(x,y,Fs,BW))):
    ff,cohere = multitaper_coherence(x,y,Fs,BW)
    return (t,cohere)

def sliding_multitaper_coherence_parallel(x,y,window=500,step=100,Fs=1000,BW=5):
    '''
    Sliding multitaper coherence between x and y
    Takes multiple samples over time, but estimates each sample using multi-taper
    See also multitaper_coherence
    '''
    N = len(x)
    assert len(y)==N
    allcohere = []
    problems = [(tstart,(x[tstart:tstart+window],y[tstart:tstart+window],Fs,BW)) for tstart in xrange(0,N-window+1,step)]
    allcohere = squeeze(array(parmap(mtmchpar,problems)))
    freqs = abs(fftfreq(window,1./Fs)[:window/2+1])
    return freqs,allcohere


    
    
    
    
    
    
    
    
    
