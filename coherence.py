

from morlet import *
#execfile(expanduser('~/Dropbox/bin/morlet.py'))
#
# sudo easy_install spectrum
from spectrum.mtm import dpss
from os.path import expanduser


############################################################################
# Configure the beta band complex morlet transform

def population_coherence_spectrum(lfp,fa,fb,w=4.0,resolution=0.1,Fs=1000):
    '''
    First dimension is nchannels, second is time.
    Use morlet wavelets ( essentially bandpass filter bank ) to compute
    short-timescale coherence.
    for each band: take morlet spectrum over time. 
    take kuromoto or synchrony measure over complex vectors attained
    '''
    assert 0
    # this measures synchrony not coherence. 
    # disabling it until we understand coherence better
    freqs, transformed = fft_cwt(lfp.T,fa,fb,w,resolution,Fs)
    coherence = abs(mean(transformed,0))/mean(abs(transformed),0)
    return freqs, coherence
    
def population_eigencoherence(lfp,fa,fb,w=4.0,resolution=0.1,Fs=1000):
    '''
    Uses the eigenvalue spectrum of the pairwise coherence matrix. 
    In the case of wavelets, each time-frequency point has one 
    complex value.
    
    The matrix we build will be I think |z_i z_j|
    
    ... this will involve a lot of computation. 
    ... let's not do it.
    
    See ramirez et al
    A GENERALIZATION OF THE MAGNITUDE SQUARED COHERENCE SPECTRUM FOR
    MORE THAN TWO SIGNALS: DEFINITION, PROPERTIES AND ESTIMATION
    '''
    assert 0

@memoize
def dpss_memo(N,NW):
    '''
    ftapers is K x N
    K = NW*2
    '''
    tapers,eigen = dpss(N,NW)
    return tapers.T,eigen

def population_coherence_matrix(lfp):
    '''
    lfp is a Nch x Ntimes matrix of data channels.
    ntapers is a positive integer.
    
    For each pair of channels compute multitaper coherence.
    take the product of each taper with each channel and take the FT
    '''
    NCH,N = shape(lfp)
    tapers,eigen = dpss_memo(N,10.0)
    M = sum(eigen)**2/sum(eigen**2) # adjusted sample size
    tapered = arr([fft(lfp*taper,axis=1) for taper in tapers])
    NT = len(eigen)
    def magsq(z):
        return real(z*conj(z))
    psd = arr([sum([magsq(tapered[k,i,:])*eigen[k] for k in range(NT)],0)/sum(eigen) for i in range(NCH)])
    results = zeros((NCH,NCH,N),dtype=float64)
    for i in range(NCH):
        results[i,i]=1
        for j in range(i):
            a = tapered[:,i,:]
            b = tapered[:,j,:]
            nn = sum([a[k]*conj(b[k])*eigen[k] for k in range(NT)],0)/sum(eigen)
            coherence = magsq(nn)/(psd[i]*psd[j])
            sqrc = sqrt(coherence)
            unbiased = (M*sqrc-1)/(M-1)
            results[i,j,:]=results[j,i,:]=unbiased
    factored = zeros((NCH,NCH,N),dtype=float64)
    spectra  = zeros((NCH,N),dtype=float64)
    for i in range(N):
        w,v = eig(results[:,:,i])
        v[:,w<0]*=-1
        w[w<0]*=-1
        order = argsort(w)
        spectra [:,i]   = w[order]
        factored[:,:,i] = v[:,order]
    freqs = fftfreq(N,1./1000)

    return freqs,results,spectra,factored
    
    #@memoize
def dpss_memo(N,NW):
    '''
    ftapers is K x N
    K = NW*2
    '''
    tapers,eigen = dpss(N,NW)
    return tapers.T,eigen


'''
execfile(expanduser('~/Dropbox/bin/cgid/cgidsetup.py'))
okwarn()
session = 'RUS120518'
area = 'M1'
trial = get_good_trials(session,area)[0]

csd = []
for i in range(0,7000-100,5): 
    print i
    lfp = get_all_lfp(session,area,trial,(6,-1000+i,-1000+i+100))
    s = population_coherence_matrix(lfp)[2][0]
    csd.append(s)

# Coherence example
Fs = 1000
N = Fs*4
t = arange(N)

s1 = cos(t*2*pi*90/Fs)
s2 = cos(t*2*pi*40/Fs)
n1 = randn(N)
n2 = randn(N)
n3 = randn(N)

g1 = n1+n2
g2 = n3+n2

tapers,eigen = dpss_memo(N,10.0)
tapered1 = array([fft(g1*taper,axis=1) for taper in tapers])
tapered2 = array([fft(g2*taper,axis=1) for taper in tapers])
DF = sum(eigen) 
NT = len(eigen)
psd1 = sum([magsq(tapered1[k,:])*eigen[k] for k in range(NT)],0)/DF
psd2 = sum([magsq(tapered2[k,:])*eigen[k] for k in range(NT)],0)/DF
def magsq(z):
    return real(z*conj(z))
nn = sum([tapered1[k]*tapered2(b[k])*eigen[k] for k in range(NT)],0)/DF
coherence = magsq(nn)/(psd[i]*psd[j])
plot(coherence)
'''

