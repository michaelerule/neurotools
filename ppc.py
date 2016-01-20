

from spectrum.mtm import dpss
import types
from numpy import *
from neurotools.getfftw import *
from numpy.random import *
from neurotools.tools import warn
from neurotools.signal import phaserand

__PPC_FP_TYPE__=float128

def fftppc_biased(snippits,Fs=1000,taper=None):
    # some precision trouble
    # use quad precition
    # PPC doesn't care about scale so also rescale? 
    snippits = array(snippits,dtype=__PPC_FP_TYPE__)
    snippits = snippits/std(snippits)
    M,window = shape(snippits)
    if not taper is None:
        snippits = snippits*taper;
    if M<=window: warn('WARNING SAMPLES SEEM TRANSPOSED?')
    fs       = fft(snippits)
    average  = mean(fs/abs(fs),0)
    raw      = abs(average)**2
    phases   = angle(average)
    freqs    = fftfreq(window,1./Fs)
    return freqs[:(window+1)/2], raw[:(window+1)/2], phases[:(window+1)/2]

def fftppc(snippits,Fs=1000,taper=None):
    '''
    make sure this is equivalent to the following matlab lines
    raw     = abs(sum(S./abs(S))).^2;
    ppc     = (raw - M)./(M.*(M-1));
    
    the raw computation uses the mean which includes a 1/M factor
    which is then squared so the raw PPC is already /M^2 compared
    to the PPC value intermediate in the matlab code. So we have to
    multiply it by M^2
    '''    
    M,window = shape(snippits)
    ff,raw,phase = fftppc_biased(snippits,Fs,taper=taper)
    unbiased = (raw*M-1)/(M-1)   
    return ff, unbiased, phase

def fftppc_biased_multitaper(snippits,Fs=1000,k=4):
    # some precision trouble
    # use quad precition
    # PPC doesn't care about scale so also rescale? 
    #nippits = array(snippits,dtype=__PPC_FP_TYPE__)
    #snippits = snippits/std(snippits)
    M,window = shape(snippits)
    print M,window
    if M<=window: warn('WARNING SAMPLES SEEM TRANSPOSED?')
    #print 'BROKE FOR MATLAB CHECK REMOVE +1 on K KEEP ALL TAPERS'
    #tapers   = dpss(window,NW=0.499*(k+1),k=(k+1))[0][:,:-1] 
    tapers   = dpss(window,NW=0.499*k,k=k)[0]
    results  = []
    unit  = lambda x:x/abs(x)
    average = [mean(unit(fft(snippits*taper)),0) for taper in tapers.T]
    raw     = mean([abs(x)**2 for x in average],0)
    phases  = angle(mean([exp(2j*pi*angle(x)) for x in average],0))
    freqs = fftfreq(window,1./Fs)
    return freqs[:(window+1)/2], raw[:(window+1)/2], phases[:(window+1)/2]

def fftppc_multitaper(snippits,Fs=1000,k=4):
    # some precision trouble
    # use quad precition
    # PPC doesn't care about scale so also rescale? 
    #snippits = array(snippits,dtype=__PPC_FP_TYPE__)
    M,window = shape(snippits)
    if M<=window: warn('WARNING SAMPLES SEEM TRANSPOSED?')
    ff,raw,phase = fftppc_biased_multitaper(snippits,Fs,k)
    unbiased = (raw*M-1)/(M-1)   
    return ff, unbiased, phase

def nodc(x):
    return x-mean(x)

def discard_spikes_closer_than_delta(signal,times,delta,window):
    '''
    When computing PPC, we need to throw out spikes that are too close 
    together. This is a heuristic to make the spiking samples 
    "more independent". We also need to skip spikes that are so close to
    the edge of our data that we can't get the LFP window surrounding 
    the spike time. Because certain tests that compare across conditions
    require matching the number of spikes, to ensure that the variance
    of the PPC estimator is comparable between the conditions, we expose
    the code for selecting the subset of spikes for PPC here, so that
    it can be used to ensure that both conditions have a matching number
    of spikes
    '''
    N = len(signal)
    times = array(times)
    #print '%d spikes total'%len(times)
    times = times[times>=window]
    times = times[times<N-window]
    #print '%d spikes with enough padding'%len(times)
    usetimes = []
    t_o = -inf
    for t in times:
        if t-t_o>=delta:
            t_o = t
            usetimes.append(t)
    #print '%d spikes far enough apart to be usable'%len(usetimes)
    return usetimes

def pairwise_phase_consistancy(signal,times,window=50,Fs=1000,k=4,multitaper=True,biased=False,delta=100,taper=None):
    '''
    signal: 1D real valued signal
    times:  Times of events relative to signal
    window: Time around event to examine
    Fs:     sample rate for computing freqs
    k:      number of tapers
    Also accepts lists of signals / times
    returns frequencies, ppc values
    
    returns (PPC, frequencies), number of spikes ultimately used to compute
    PPC
    '''
    if multitaper:
        print "Warning: multitaper can introduce a bias into PPC that depends on the number of tapers!"
        print "For a fixed number of tapers, the bias is constant, but be careful"
    
    if not taper is None and multitaper:
        print "A windowing taper was specified, but multitaper mode was also selected? The taper argument is for providing a windowing function when not using multitaper estimation"
        assert 0
    if type(taper) is types.FunctionType:
        taper = taper(window*2+1)
    if biased: warn('skipping bias correction entirely')
    assert window>0
    if len(shape(signal))==1:
        usetimes = discard_spikes_closer_than_delta(signal,times,delta,window)
        snippits = array([nodc(signal[t-window:t+window+1]) for t in usetimes])
    elif len(shape(signal))==2:
        warn('assuming first dimension is trials / repititions')
        signals,alltimes = signal,times
        snippits = []
        for signal,times in zip(signals,alltimes):
            N = len(signal)
            times = array(times)
            times = times[times>=window]
            times = times[times<N-window]
            #snippits.extend([nodc(signal[t-window:t+window+1]) for t in times])
            t_o = -inf
            for t in times:
                if t-t_o>delta:
                    t_o = t
                    snippits.append(nodc(signal[t-window:t+window+1]))
    else: assert 0
    #print shape(snippits)
    if biased:
        if multitaper: return fftppc_biased_multitaper(snippits,Fs,k),snippits
        else:          return fftppc_biased(snippits,Fs,taper=taper),snippits
    else:
        if multitaper: return fftppc_multitaper(snippits,Fs,k),snippits
        else:          return fftppc(snippits,Fs,taper=taper),snippits
    assert 0

def estimate_bias_in_uncorrected_ppc(signal,times,window=50,Fs=1000,nrand=100):
    tried = []
    for i in range(nrand):
        ff,ppc = uncorrectedppc(phaserand(signal),times,window,Fs)
        tried.append(ppc)
    bias = mean(tried,0)
    return ff,bias

def phase_randomized_bias_correction(signal,times,window=50,Fs=1000,nrand=100):
    '''
    Estimates degrees of freedom using phase randomization.
    experimental.
    
    algebra could be dramaticalyl simplified, but keeping all explicit
    for clarity for now
    '''
    warn('AS FAR AS WE KNOW THIS DOESNT REALLY WORK')    
    ff,bias = estimate_bias_in_uncorrected_ppc(signal,times,window,Fs,nrand)
    K = 1.0-bias
    M = K/(1-K)
    print 'estimated degrees of freedom:',M
    print 'nominal degrees of freedom=',len(times)
    ff,raw = uncorrectedppc(signal,times,window,Fs)
    unbiased = (raw*M-1)/(M-1)  
    return ff, unbiased


def temp_code_for_exploring_chance_level_delete_later():
    # bias/ariance analysis
    # my understanding is that No. of samples should not change the mean PPC
    # value, but it will affect the variance. Since the PPC is computed as
    # a sum of complex unit vectors, we can reason about it's bias/variance
    # as a function of samples from a few simple simulations. This will replace
    # brute force simulations of chance level based on phase randomization. 
    # The effect of multitaper on reducing the bias or variance is interesting.
    # I don't know how to account for that.
    nTapers = 1
    for nSamples in range(2,1000):
        simulated = []
        for i in range(100000):
            unbiased = (mean([abs(mean(exp(2*pi*1j*rand(nSamples)))) for t in range(nTapers)])**2*nSamples-1)/(nSamples-1)
            simulated.append(unbiased)
        simulated = sorted(simulated)
        print 'n=%d 90%%=%f'%(nSamples,simulated[int(.90*len(simulated))])
        print 'n=%d 95%%=%f'%(nSamples,simulated[int(.95*len(simulated))])
        print 'n=%d 99%%=%f'%(nSamples,simulated[int(.99*len(simulated))])
        print 'n=%d 99.9%%=%f'%(nSamples,simulated[int(.999*len(simulated))])
        clf()
        hist(simulated,100)
        draw()
        raw_input()

def ppc_chance_level(nSamples,nrandom,p,nTapers=1):
    '''
    Don't use this, it underestimates chance level in the presence of
    Spike trian or LFP autocorrelations
    '''
    assert 0
    simulated = []
    for i in range(nrandom):
        unbiased = (mean([abs(mean(exp(2j*pi*rand(nSamples)))) for t in range(nTapers)])**2*nSamples-1)/(nSamples-1)
        simulated.append(unbiased)
    return sorted(simulated)[int(p*len(simulated))]


def ppc_phaserand_chance_level_sample(
    signal,times,window=50,Fs=1000,k=4,multitaper=True,
    biased=False,delta=100,taper=None):
    '''
    signal: 1D real valued signal
    times:  Times of events relative to signal
    window: Time around event to examine
    Fs:     sample rate for computing freqs
    k:      number of tapers
    Also accepts lists of signals / times

    Uses phase randomization to sample from the null hypothesis distribution.
    Returns the actual PPC samples rather than any summary statistics.
    You can do what you want with the distribution returned.
    '''
    if multitaper:
        print "Warning: multitaper can introduce a bias into PPC that depends on the number of tapers!"
        print "For a fixed number of tapers, the bias is constant, but be careful"
    
    if not taper is None and multitaper:
        print "A windowing taper was specified, but multitaper mode was also selected."
        print "The taper argument is for providing a windowing function when not using multitaper estimation."
        assert 0
    if type(taper) is types.FunctionType:
        taper = taper(window*2+1)
    if biased: warn('skipping bias correction entirely')
    assert window>0
    
    # need to compute PPC from phase randomized samples
    # We can't phase randomize the snippits because there may be some
    # correlation between them, we have to randomize the underlying LFP
    
    if len(shape(signal))==1:
        # only a single trial provided
        usetimes = discard_spikes_closer_than_delta(signal,times,delta,window)
        signal = phaserand(signal)
        snippits = array([nodc(signal[t-window:t+window+1]) for t in usetimes])
    elif len(shape(signal))==2:
        warn('assuming first dimension is trials / repititions')
        signals,alltimes = signal,times
        snippits = []
        for signal,times in zip(signals,alltimes):
            signal = phaserand(signal)
            N = len(signal)
            times = array(times)
            times = times[times>=window]
            times = times[times<N-window]
            t_o = -inf
            for t in times:
                if t-t_o>delta:
                    t_o = t
                    snippits.append(nodc(signal[t-window:t+window+1]))
    else: assert 0
    
    if biased:
        if multitaper: return fftppc_biased_multitaper(snippits,Fs,k),snippits
        else:          return fftppc_biased(snippits,Fs,taper=taper),snippits
    else:
        if multitaper: return fftppc_multitaper(snippits,Fs,k),snippits
        else:          return fftppc(snippits,Fs,taper=taper),snippits
    assert 0
    
    
    





