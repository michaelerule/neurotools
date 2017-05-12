'''
Miscellaneous utilities used for fitting spatiotemporal GLMs to 
spiking data. 

UNDER CONSTRUCTION
'''

from neurotools.stats.glm import gradientglmfit
from neurotools.spatial.magickernel import derive_log_cosine_basis

import scipy
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib.mlab import find

def pca(x,n_keep=None):
    '''
    w,v = pca(x,n_keep=None)
    Performs PCA on data x, keeping the first n_keep dimensions
    '''
    assert x.shape[1]<=x.shape[0]
    cov = x.T.dot(x)
    w,v = scipy.linalg.eig(cov)
    o   = np.argsort(-w)
    w,v = w[o].real,v[:,o].real
    if n_keep is None: n_keep = len(w)
    w,v = w[:n_keep],v[:,:n_keep]
    return w,v

def orthogonalize(B):
    '''
    B = orthogonalize(B)
    
    Orthogonalizes a basis; Assumes each row of B is a basis vector
    
    Parameters
    ----------
    B : ndimensions x nbases matrix of basis vectors
    
    Returns
    B : orthonomalized version of B
    '''
    return np.linalg.pinv(scipy.linalg.sqrtm(B.dot(B.T))).dot(B).real

#http://stackoverflow.com/questions/1265665/python-check-if-a-string-represents-an-int-without-using-try-except
def isInt(v):
    v = str(v).strip()
    return v=='0' or (v if v.find('..') > -1 else v.lstrip('-+').rstrip('0').rstrip('.')).isdigit()

def local_ch_map(sr,sc,radius,width=64):
    '''
    chs,ch_map = local_ch_map(unit,radius)

    Parameters
    ----------
    sr,sc  : row, column of cell in image coordinates
    radius : radius (in pixels) to include
    width  : width/height of the stim images 
    '''
    chs = []
    ch_map   = np.zeros((2*radius+1,2*radius+1),'int')-1
    ch_index = 0
    for i in np.arange(-radius,radius+1): # r
        for j in np.arange(-radius,radius+1): # c
            if i*i+j*j>radius**2: continue
            r,c = int(round(i+sr)),int(round(j+sc))
            if r<0 or c<0 or r>=width or c>=width: 
                pass
                #chs.append(-1)
            else: 
                # this is the index into the unraveled image
                chs.append(r*width+c) 
                # this is a map from the local image to channels
                ch_map[i+radius,j+radius] = ch_index
                ch_index+=1
    chs = np.array(chs)
    return chs,ch_map

def spatiotemporal_history_projection(stims,Bh,Bs,pad=0):
    '''
    X = spatiotemporal_history_projection(stims,Bh,Bs,pad=0)
    '''
    if len(stims.shape)==1:
        # only one channel
        stims = stims.reshape(stims.shape+(1,))
    nstim,nch = stims.shape
    assert(pad>=0)
    # pad stimuli if applicable
    if pad>0:
        stims = np.concatenate([np.zeros((1,nch)),stims])
    comp = Bs.dot(stims.T)
    X  = [np.convolve(x,b,'full') for x in comp for b in Bh]
    X  = np.array(X)[:,:nstim]
    return X

def spatiotemporal_history(stims,hlen,hn,hmin,chs=None,ncomp=None,pad=0):
    '''
    X,Bh,Bs = spatiotemporal_history(stimuli,channels,hlen,hn,hmin,ncomp,pad)

    Spatiotemporal features generator. 

    Accepts a vector of stimuli `stims` and a set of indecies into 
    this vector `chs`; constructs `hn` history basis elements 
    spanning time `hlen` with the minum time resolution of `hmin`. 
    Compresses the `chs` subset of `stims` into `ncomp` dimensions
    using PCA. Returns the spatiotemporal history features, as well 
    as the bases `Bs` and `Bh` needed to reconstruct the spatial and 
    temporal dimensions, respectively.

    Parameters
    ----------
    stims : nsamples x nchannels array of stimuli data
    hlen  : No. frames history to include
    hn    : No. history basis function
    hmin  : width of the narrowest history basis function in frames
    chs   : Channels to use
    ncomp : No. PCA compressed features to keep; 
            set to len(channels) to keep all
    pad   : int; No. zero samples added to the begining, 
            used to add pad frames of latency

    Returns
    -------
    X  : nsamples×ncomp*nh spatiotemporal history features
    Bh : hlen×nh history   basis functions
    Bs : nchannels×ncomp   spatial basis functions
    '''
    assert(hn  >0)
    assert(hlen>0)
    assert(hmin>0)
    if len(stims.shape)==1:
        # only one channel
        stims = stims.reshape(stims.shape+(1,))
    nstim,nch = stims.shape
    assert(nch<nstim)
    assert(nch>0)
    # Select channels
    if not chs is None:
        assert(all(chs<nch))
        assert(len(chs)>0)
        stims     = stims[:,chs]
        _,nch = stims.shape
    # compress using PCA
    if not ncomp is None:
        assert(ncomp>0)
        assert(ncomp<=nch)
        w,v = pca(stims,ncomp)
        Bs = v.T
        #Bs = np.sqrt(w[:,None])*v.T
    else:
        Bs = np.eye(nch)
    # Convolve w history basis
    Bh = orthogonalize(derive_log_cosine_basis(hn,hlen,hmin))
    X = spatiotemporal_history_projection(stims,Bh,Bs,pad=0)
    return X,Bh,Bs

def plot_spatiotemporal(B,Bs,Bh,ch_map):
    '''
    Q = plot_spatiotemporal(B,Bs,Bh,ch_map)
    Decompress and visualize spatiotemporal stimulus model
    
    Parameters
    ----------
    B  : nspace * nhist    spatiotemporal parameter vector, compressed
    Bs : nspace × channels spatial compression basis
    Bh : nhist  × hframes  history compression basis
    ch_map : 2d array showing how to map stimulus indecies into a space
    
    Returns
    -------
    Q : the ch_map.shape × hframes decompressed spatiotemporal history filter
    '''
    nspace,nchannels = Bs.shape
    nhist ,hframes   = Bh.shape
    assert(nspace<=nchannels)
    assert(nhist<=hframes)
    assert(B.shape==(nspace*nhist,))
    
    # decompress stimulus and place into channel map
    Q = np.linalg.pinv(Bs).real.dot(B.reshape((nspace,nhist))).dot(np.linalg.pinv(Bh).real.T)[ch_map,:]
    # plot
    # Truncate time history to a perfect square, and reshape data
    K = int(np.sqrt(Q.shape[-1]))
    Q2 = Q.transpose((2,0,1))[:K*K,...].reshape((K,K)+(ch_map.shape))
    # Form grid depiction of spatiotmeporal history
    Q2 = np.concatenate(np.concatenate(Q2,axis=1),axis=1)
    # plot z-scores
    zscores = (Q2-np.mean(Q2))/np.std(Q2)
    plt.imshow(zscores,vmin=-8,vmax=8)
    # draw a better grid
    plt.gca().grid(False)
    for i in range(K):
        plt.axvline(i*ch_map.shape[0],color=(0.1,)*3)
        plt.axhline(i*ch_map.shape[1],color=(0.1,)*3)
    return Q

def plot_histfilter(B,basis,bin_duration_ms):
    '''
    plot_histfilter(B,basis,bin_duration_ms)
    
    Plots stimulus filter in given basis projection
    
    Parameters
    ----------
    B : basis vector weights
    basis : basis set, should be orthogonal
    bin_duration_ms : time durations of bins
    '''
    mm   = np.exp(np.linalg.pinv(basis).dot(B)).real
    time = np.arange(len(mm))*bin_duration_ms
    plt.plot(time,mm)
    

def GLMPenaltyL2_subsampled(X,Y,penalties=None,class_balance=2):
    '''
    Generates objective, gradient, and hessian functions for the penalized
    L2 regularized poisson GLM for design matrix X and spike observations Y.
    
    Uses a subsampling of the no-spike examples to estimate expectations in
    the likelihood.
    
    Args:
        X: N observation by D features design matrix
        Y: N by 1 point-process counts
        penalties: len D-1 list of L2 penalty weights (don't penalize mean)

    Returns:
        objective, gradient(jacobian), hessian
    '''
    N,D = X.shape
    assert N>D
    if penalties is None: penalties = np.zeros((D,),'d')
    if type(penalties) in (float,int):
        penalties = np.ones((D,),dtype='d')*penalties
    assert Y.shape==(N,)
    Y = np.squeeze(Y)
    assert len(Y.shape)==1
    X  = np.float64(X)       # use double precision
    Ey = np.mean(Y)
    Ex1= np.sum(X[Y==1,:],0)/N
    # Now, downsample X
    n1 = np.sum(Y)
    n0 = np.sum(Y==0)
    if any(Y<0):
        print('Spike counts must be positive!')
        assert 0
    if any(Y>10):
        print('Spike count larger than 10 encountered, this is unusual!')
    if class_balance<1:
        print('class_balance is the number of background examples to use for every spike')
        print('It should be an integer greater than or equal to 1')
        assert 0
    if (n1<100):
        print('Fewer than 100 spike exampled provided')
        print('Not enough data to fit model, aborting')
        assert 0
    if (n1*2>n0):
        print('No. spikes is larger than 50% No. background examples')
        print('Poisson GLM approximation is poor, aborting')
        assert 0
    nUse  = n1*class_balance
    nSkip = max(1,int(n0/nUse))
    # Downsample no-spike examples only
    use1  = find(Y>0) 
    use0  = find(Y==0)[::nSkip]
    use   = np.concatenate([use1,use0])
    X = X[use,:]
    Y = Y[use]
    # make sure we didn't throw away any spikes
    assert(sum(Y)==n1)
    # model is fit using expectations
    # we estimate these expectations using the downsampled data
    def objective(H):
        mu = H[0]
        B  = H[1:]
        rate = np.exp(mu+X.dot(B))
        like = Ey*mu+Ex1.dot(B)-np.mean(rate)-np.sum(penalties*B**2)
        return -like
    def gradient(H):
        mu = H[0]
        B  = H[1:]
        rate     = np.exp(mu+X.dot(B))
        dmu      = Ey - np.mean(rate)
        reweight = np.mean(X*rate[:,None],0)
        dbeta = Ex1.T - reweight - 2*penalties*B
        grad  = np.append(dmu, dbeta)
        return -grad
    def hessian(H):
        mu = H[0]
        B  = H[1:]
        rate  = np.exp(mu+X.dot(B))
        dmumu = np.mean(rate)
        dmuB  = np.mean(X*rate[:,None],0)
        dBB   = X.T.dot(rate[:,None]*X)/X.shape[0]
        ddpen = np.diag(np.ones(len(B)))*penalties*2
        hess  = np.zeros((len(H),)*2)
        hess[0 ,0 ] = dmumu
        hess[0 ,1:] = dmuB
        hess[1:,0 ] = dmuB.T
        hess[1:,1:] = dBB + ddpen
        return hess
    return objective, gradient, hessian

def gradientglmfit_subsampled(X,Y,L2Penalty=0,class_balance=2):
    '''
    mu_hat, B_hat = gradientglmfit(X,Y,L2Penalty=0.0)
    Fit Poisson GLM using gradient descent with hessian
    
    Uses a likelihood function where expectations over the features
    are estimated using a random subsampling of the features.
    '''
    objective, gradient, hessian = GLMPenaltyL2_subsampled(X,Y,L2Penalty,class_balance=class_balance)
    initial = np.zeros(X.shape[1]+1)
    M = scipy.optimize.minimize(objective, initial,
        jac   = gradient,
        hess  = hessian,
        method='Newton-CG')['x']
    mu_hat,B_hat = M[0],M[1:]
    return mu_hat, B_hat
