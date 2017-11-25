#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
from neurotools.system import *

'''
Miscellaneous routines from lessons
'''

import neurotools.stats.hmm as hmm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from   neurotools.graphics.color import BLACK,RUST,TURQUOISE,OCHRE,AZURE
from matplotlib.mlab import find

# Adjust plotting preference
# TODO: move these config options elsewhere
plt.rcParams['image.cmap'] = 'parula'
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['figure.figsize'] = (10,3)
plt.rcParams['axes.color_cycle'] = [BLACK,RUST,TURQUOISE,OCHRE,AZURE]
plt.rcParams['legend.borderaxespad'] = 0.

def sample_wiener_process(x0,sigma,dt,N,ntrial=1):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    simulated = np.zeros((N,ntrial),dtype=np.float32)
    x = x0*np.ones((ntrial,),dtype=np.float32)
    for i in range(N):
        x += sigma * np.random.randn(ntrial) * np.sqrt(dt)
        simulated[i] = x
    return simulated

def sample_ou_process(x0,sigma,tau,dt,N,ntrial=1):
    '''
    Sample from OU
    
    Args:
        x0 : initial conditions
        sigma : standard deviation of driving Wiener process
        tau : exponential damping time constant
        dt : time step
        N : number of samples to draw 
        
    Parameters
    ----------
    
    Returns
    -------
    '''
    simulated = np.zeros((N,ntrial),'float')
    x = x0*np.ones((ntrial,),'float')
    for i in range(N):
        x += -(1./tau)*x*dt + sigma * np.random.randn(ntrial) * np.sqrt(dt)
        simulated[i] = x
    return simulated

def add_spikes(Y):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    # plot spikes
    y1,y2 = plt.ylim()
    y3 = y2+.1*(y2-y1)
    for e in find(Y==1):
        plt.plot([e,e],[y2,y3],color='k',lw=.5)
    plt.ylim(y1,y3)
   
def precision1D(N,a,b):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    a =  1 # diagonal weights (precisions)
    b = .1 # off-diagonal (precision of time correlations)
    precision = np.diag(np.ones(N)*a)*2 - np.diag(np.ones(N-1)*b,-1) - np.diag(np.ones(N-1)*b,1)
    return precision
    
def showim(x,**vargs):
    '''
    Parameters
    ----------
    Returns
    -------
    '''
    plt.figure()
    if x.dtype in (np.complex64,np.complex128,np.complex256):
        vmin = min(np.min(x.real),np.min(x.imag),np.min(abs(x)))
        vmax = max(np.max(x.real),np.max(x.imag),np.max(abs(x)))
        plt.subplot(131)
        plt.imshow(x.real,vmin=vmin,vmax=vmax,**vargs)
        plt.subplot(132)
        plt.imshow(x.imag,vmin=vmin,vmax=vmax,**vargs)
        plt.subplot(133)
        plt.imshow(np.abs(x),vmin=vmin,vmax=vmax,**vargs);
    else:
        # real-valued matrix
        plt.imshow(x,**vargs)

def infer_states_Gaussian_ADF(Y,variance,true_states=None,do_plot=True):
    '''
    Parameters
    ----------
    
    Returns
    -------
    '''
    fx0 = hmm.Gaussian(0,0)
    fA  = hmm.DiffusionGaussian(variance)
    fB  = hmm.PoissonObservationModel(1,0)
    L = len(Y)
    fwd,bwd,posterior = forward_backward_abstract(Y,L,fx0,fA,fB)
    m = np.array([p.m for p in posterior])
    v = np.array([p.t for p in posterior])
    s = v**-.5
    if do_plot:
        plt.figure()
        plt.plot(m,color='k')
        plt.fill_between(range(L),m-s*1.95,m+s*1.95,color=(0.1,)*4,lw=0)
        if not true_states is None:
            plt.plot(true_states,color='r');
        plt.xlim(0,L)
    return m,v

class PGOM:
    '''
    Approximate Gaussian distribution to use in abstracted forward-
    backward. Used to condition Gaussian states on Poisson 
    observations. An additional option to combine another Gaussian
    factor into the conditioning has been added. 
    '''
    class Approximate(hmm.TruncatedLogGaussianCoxApproximator):
        def __init__(s,y,g,a=1,b=0):
            '''
            Parameters
            ----------
            
            Returns
            -------
            '''
            TruncatedLogGaussianCoxApproximator.__init__(s,a,b,y)
            s.g = g
        def __mul__(s,o):
            '''
            Parameters
            ----------
            
            Returns
            -------
            '''
            # first combine our Gaussian with the other Gaussian
            # then call the integration method in parent class
            return TruncatedLogGaussianCoxApproximator.__mul__(s,s.g if o is 1 else s.g*o)
    def __init__(s,a=1,b=0):
        '''
        Args:
            g : gaussian distribution with which to combine
            a : gain term ( set to 1 for no gain )
            b : bias term ( set to 0 for no gain )
        '''
        s.a,s.b = a,b
    def __call__(s,args):
        y,p = args
        ''' 
        Args:
            y : poisson count observation 
            p : Gaussian prior
        '''
        assert isinstance(p,Gaussian)
        return s.Approximate(y,p,s.a,s.b)


