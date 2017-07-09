'''
Library of array statistics routines.
'''

#execfile(expanduser('~/Dropbox/bin/stattools.py'))
from stat import *
from numpy import *
import numpy as np
from neurotools.tools import *

ELECTRODE_SPACING = 0.4

def population_kuromoto(population):
    warn('statistics computed over first axis. not for 2d array data')
    return abs(mean(population/abs(population),axis=0))

def population_synchrony(population):
    warn('statistics computed over first axis. not for 2d array data')
    return abs(mean(population,axis=0))/mean(abs(population),axis=0)

def population_polar_std(population):
    warn('statistics computed over first axis. not for 2d array data')
    return sqrt(-2*log(population_synchrony(population)))

def population_average_amplitude(population):
    warn('statistics computed over first axis. not for 2d array data')
    return mean(abs(population),axis=0)

def population_signal_concentration(lfp):
    warn('statistics computed over first axis. not for 2d array data')
    return population_average_amplitude(lfp)/population_signal_dispersion(lfp)**(-0.25)

def population_signal_precision(lfp):
    warn('statistics computed over first axis. not for 2d array data')
    return population_signal_dispersion(lfp)**(-1)

def population_signal_dispersion(lfp):
    warn('statistics computed over first axis. not for 2d array data')
    r = real(lfp)
    i = imag(lfp)
    r = r-mean(r,0)
    i = i-mean(i,0)
    Crr = mean(r*r,0)
    Cri = mean(r*i,0)
    Cii = mean(i*i,0)
    Det = Crr*Cii-Cri**2
    return Det**0.25

def population_signal_phase_dispersion(lfp):
    warn('statistics computed over first axis. not for 2d array data')
    # rotate into best-guess zero phase reference frame
    z = mean(lfp,axis=0)
    h = (z/abs(z))**-1
    x = lfp*h
    # phase dispersion is CV=sig/mu deviation along the imaginary axis
    # the mean is always zero so we don't normalize it
    s = std (imag(x),axis=0)
    m = mean(real(x),axis=0)
    return s/m

def population_signal_phase_std(lfp):
    warn('statistics computed over first axis. not for 2d array data')
    # rotate into best-guess zero phase reference frame
    z = mean(lfp,axis=0)
    h = (z/abs(z))**-1
    x = lfp*h
    # phase dispersion is CV=sig/mu deviation along the imaginary axis
    # the mean is always zero so we don't normalize it
    s = std (imag(x),axis=0)
    return s

def population_signal_amplitude_std(lfp):
    warn('statistics computed over first axis. not for 2d array data')
    # rotate into best-guess zero phase reference frame
    z = mean(lfp,axis=0)
    h = (z/abs(z))**-1
    x = real(lfp*h)
    # amplitude dispersion is CV=sig/mu deviation along the real axis
    s = std (x,axis=0)
    return s

def population_signal_amplitude_dispersion(lfp):
    warn('statistics computed over first axis. not for 2d array data')
    # rotate into best-guess zero phase reference frame
    z = mean(lfp,axis=0)
    h = (z/abs(z))**-1
    x = real(lfp*h)
    # amplitude dispersion is CV=sig/mu deviation along the real axis
    s = std (x,axis=0)
    m = mean(x,axis=0)
    return s/m

def population_signal_phase_precision(lfp):
    warn('statistics computed over first axis. not for 2d array data')
    return 1./population_signal_phase_dispersion(lfp)

def population_signal_amplitude_precision(lfp):
    warn('statistics computed over first axis. not for 2d array data')
    return 1./population_signal_amplitude_dispersion(lfp)

def population_signal_description(lfp):
    warn('statistics computed over first axis. not for 2d array data')
    z = mean(lfp,0)
    a = abs(z)
    h = angle(z)
    w = lfp*exp(-1j*h)
    s1 = std(imag(w),0)
    s2 = std(real(w),0)
    return z,a,h,s1,s2

def population_synchrony_linear(population):
    warn('statistics computed over first axis. not for 2d array data')
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

    Example
    -------
        s,a,tr = ('SPK120918', 'M1', 16)
        lfp = get_all_analytic_lfp(s,a,tr,epoch=(6,-1000,2001),fa=10,fb=45)
        data = lfp[:,500:600]
        sliding = arr([population_phase_coherence(lfp[:,i:i+100]) for i in range(shape(lfp)[1]-100)])
    '''
    h = angle(data)
    dfdt = (diff(h,axis=-1)+pi)%(2*pi)-pi
    mdf = median(dfdt,axis=0)
    return mean(cos(dfdt-mdf),axis=0)

def mirrorpad(data,amount):
    '''
    reflected padding of data
    '''
    assert 0

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

    Example
    -------
        s,a,tr = ('SPK120918', 'M1', 16)
        lfp = get_all_analytic_lfp(s,a,tr,epoch=(6,-1000,2001),fa=10,fb=45)
        data = lfp[:,500:600]
        sliding = arr([population_signal_coherence(lfp[:,i:i+100]) for i in range(shape(lfp)[1]-100)])
    '''
    assert len(shape(data))==2
    N = shape(data)[-1]
    h = angle(data)
    dfdt = (diff(h,axis=-1)+pi)%(2*pi)-pi
    mdf = median(dfdt,axis=0)
    weights = abs(data)[:,:-1]
    weights = weights*window(N+1)[1:-1]
    weights /= sum(weights)
    return sum(cos(dfdt-mdf)*weights,axis=0)

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
    '''
    assert len(shape(data))==2
    N       = shape(data)[-1]
    h       = angle(data)
    dfdt    = (diff(h,axis=-1)+pi)%(2*pi)-pi
    weights = abs(data)
    win     = window(L+2)[1:-1]
    slide   = []
    for i in range(0,N-L-1):
        now = dfdt[...,i:i+L]
        mdf = median(now)
        w   = weights[...,i:i+L]*win
        w  /= sum(w)
        slide.append(sum(cos(now-mdf)*w))
    return arr(slide)

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
    '''
    assert len(shape(data))==2
    N       = shape(data)[-1]
    h       = angle(data)
    dfdt    = (diff(h,axis=-1)+pi)%(2*pi)-pi
    weights = abs(data)
    win     = window(L+2)[1:-1]
    slide   = []
    for i in range(0,N-L-1):
        now = dfdt[...,i:i+L]
        mdf = median(now)
        w   = weights[...,i:i+L]*win
        w  /= sum(w)
        mu,sig = weighted_avg_and_std( (now-mdf)/mdf, w)
        slide.append(sig)
    return 1/sqrt(1+arr(slide)**2)

def population_phase_relative_sliding_kuromoto(data,L=100,window=np.hanning):
    '''
    Uses the phase of each channel in the middle of each block as a
    reference point. Can separate coherent wave activity from synchrony.
    
    $\textrm{kuromoto order} = \left\langle z/|z| \right\rangle$
    
    Assumes constant phase velocity, and a constant per-channel
    phase shift, and then computes the order. This is a notion of 
    relative phase stability.
    
    '''
    assert len(shape(data))==2
    K,N     = shape(data)
    phases  = angle(data)
    phased  = data/abs(data)
    dfdt    = (diff(phases,axis=-1)+pi)%(2*pi)-pi
    win     = window(L+2)[1:-1]
    win     = win/(K*sum(win))
    slide   = []
    for i in range(0,N-L-1):
        # get the median phase velocity
        mf = median(dfdt[...,i:i+L])
        # get the local phases
        x  = phases[...,i:i+L]
        # dephase the signal in time
        y  = rewrap(x-cumsum([mf]*L))
        # dephase the signal per channel
        z  = angle(mean(exp(1j*y),axis=-1))
        rephased = (x.T-z).T
        # compute weighted phase order
        slide.append(abs(sum(exp(1j*rephased)*win)))
    return arr(slide)

def population_median_phase_velocity(data):
    '''
    median phase velocity.
    '''
    assert len(shape(data))==2
    N       = shape(data)[-1]
    h       = angle(data)
    dfdt    = (diff(h,axis=-1)+pi)%(2*pi)-pi
    medv    = median(dfdt,0)
    return medv

def population_median_frequency(data,Fs=1000):
    '''
    Convert from phase in radians/frame to
    Frequency in cycles/s
    '''
    medv = population_median_phase_velocity(data)
    return medv*(Fs/(2*pi))





