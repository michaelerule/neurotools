#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Covariance alignment routines related to 
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import os,sys,traceback,h5py

import numpy as np

from sklearn                import manifold
from scipy.linalg           import solve,lstsq
from sklearn.decomposition  import FactorAnalysis
from sklearn                import manifold
from neurotools.util.hdfmat import printmatHDF5, hdf2dict, getHDF
from neurotools.util.time   import progress_bar


#import matplotlib
#import matplotlib.pyplot as plt
#from matplotlib.patches     import Polygon
#from matplotlib.collections import PatchCollection

NPERMUTATION = 10

def keeprank(Σ,eps = 1e-1):
    '''
    Return all eigenvectors with eigenvalues greater than epsilon times
    the largest eigenvalue.
    
    Parameters
    ----------
    Σ : 2D square array-like (positive semidefinite matrix)
        Covariance matrix 
    
    Returns
    -------
    Leading eigenvectors of Σ.
    '''
    e,v = eigh(Σ)
    e = np.abs(e)
    maxe = np.max(e)
    keep = e>eps*maxe
    return v[:,keep]

def expected_intersection_rank1(Δμ,Σ,N=100):
    '''
    Rank-1 approximation of average overlap. Uses only the largest
    eigenvector.
    
    Parameters
    ----------
    Δμ : 1D array-like (vector)
        Direction of shift in mean activity
    Σ : 2D square array-like (positive semidefinite matrix)
        Covariance matrix of distribution of single-sessiona activity
    
    Returns
    -------
    '''
    # set Δμ to unit length
    Δμ = unit_length(Δμ)
    # get normalized leading eigenvector
    vmax = top_v(Σ)
    # This will be 1 if all aligns perfectly
    e1 = (Δμ@vmax)**2
    # This is the analytic chance level
    e2 = 1/Σ.shape[0]
    return 1-(1-e1)/(1-e2)
    
def expected_intersection(Δμ,Σ):
    '''
    Unnormalized expected intersection.
    
    Parameters
    ----------
    Δμ : 1D array-like (vector)
        Direction of shift in mean activity
    Σ : 2D square array-like (positive semidefinite matrix)
        Covariance matrix of distribution of single-sessiona activity
    
    Returns
    -------
    '''
    # make unitless
    Δμ = unit_length(Δμ)
    Σ  = normedcovariance(Σ)
    return Δμ.T@Σ@Δμ

def expected_intersection_enormed(Δμ,Σ):
    '''
    Normalized expected intersection.
    
    Parameters
    ----------
    Δμ : 1D array-like (vector)
        Direction of shift in mean activity
    Σ : 2D square array-like (positive semidefinite matrix)
        Covariance matrix of distribution of single-sessiona activity
    
    Returns
    -------
    Root-mean-square normalized magnitude of the dot-product between the
    vector Δμ and unit vector directions of random variables sampled from
    a Gaussian with covariance Σ. 
    '''
    # make unitless
    Σ  = normedcovariance(Σ)
    e1 = expected_intersection(Δμ,Σ)
    # Normalize to chance level
    e2 = trace(Σ)/Σ.shape[0]
    e1 = e1**0.5
    e2 = e2**0.5
    value = (e1-e2)/(1-e2)
    return value

def expected_intersection_enormed_chance(Δμ,Σ,N=300,pct=95):
    '''
    Normalized expected intersection chance level.
    
    Parameters
    ----------
    Δμ : 1D array-like (vector)
        Direction of shift in mean activity
    Σ : 2D square array-like (positive semidefinite matrix)
        Covariance matrix of distribution of single-sessiona activity
    
    Other Parameters
    ----------------
    N : int, defaults to 300
        The number of random Monte-Carlo samples to use to asses chance
        level.
    pct : numeric in (0,100), defaults to 90
        The chance-level percentile to report    
    
    Returns
    -------
    The `pct` percentile chance level, via Monte-Carlo sampling, for the
    null hypothesis that Δμ has no more than chance alignment with the
    structure of Σ.
    '''
    # make unitless
    Δμ = unit_length(Δμ)
    Σ  = normedcovariance(Σ)
    # random unit vectors
    K = len(Δμ)
    vv= unit_length(np.random.randn(N,K),axis=1)
    # Expected projection of drift onto variability
    e1 = np.array([v.T@Σ@v for v in vv])
    # Normalize to analytic chance level
    e2 = trace(Σ)/K
    e1 = e1**0.5
    e2 = e2**0.5
    value = (e1-e2)/(1-e2)
    return np.percentile(value,pct)

def rebuild_unit_quality_caches():
    '''
    Rebuilds a disk cache of which units are "good" units (have viable 
    recordings on a given session). 
    '''
    allunits = {}
    for animal in get_subject_ids():
        allunits[animal] = set()
        for session in get_session_ids(animal):
            units = good_units_index(animal,session)
            allunits[animal] |= set(units)
        allunits[animal] = array(sorted(list(allunits[animal])))
        NIDS = len(good_units(animal,session))
        NUNITS = len(allunits[animal])
        print('Animal %d, %d out of %d units available'%(animal,NUNITS,NIDS))


def orthnormedcovariance(C):
    '''
    Given covariance Σ
    Computes I-Σ/λmax
    '''
    N = C.shape[0]
    C = normedcovariance(C)
    Corth = np.eye(N)-C
    return Corth
    
def get_orthogonal_alignment(animal,CUE,PREV,verbose=True,**kwargs):
    '''
    Compute orthogonal complement drift alignment statistics 
    for the given animal, previous cue, and current cue. 
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if verbose:
        print('Getting trial-conditioned neural sigals for %s %s %s'%(animal,CUE,PREV))
    allμ,allΣ,alldμ,alldΣ,allunits =\
        get_trial_conditioned_population_activity(animal,CUE,PREV,**kwargs)
    tvar,tvarch,cvar,cvarch = [],[],[],[]
    sessions  = get_session_ids(animal)
    for i,(s1,s2) in enumerate(zip(sessions[:-1],sessions[1:])):
        if verbose:
            print('Comparing sessions %d and %d'%(s1,s2))
        days_apart = s2-s1

        # Get population statistics for both days
        μ1,μ2   = allμ[i:i+2]
        Σ1,Σ2   = allΣ[i:i+2]
        u1,u2   = allunits[i:i+2]
        dμ1,dμ2 = alldμ[i:i+2]
        dΣ1,dΣ2 = alldΣ[i:i+2]

        # Get units in common
        units = array(sorted(list(set(u1)&set(u2))))
        ix1 = [i for i in range(len(u1)) if u1[i] in units]
        ix2 = [i for i in range(len(u2)) if u2[i] in units]

        # Extract the mean drift vector
        μ1r,μ2r = μ1[:,ix1],μ2[:,ix2]
        Σ1r,Σ2r = Σ1[:,ix1,:][:,:,ix1],Σ2[:,ix2,:][:,:,ix2]
        Δμ = μ2r - μ1r

        # Extract the coding-relevant distributions
        dμ1r,dμ2r = dμ1[:,ix1],dμ2[:,ix2]
        dΣ1r,dΣ2r = dΣ1[:,ix1,:][:,:,ix1],dΣ2[:,ix2,:][:,:,ix2]

        # Compare the drift direction with the coding axis
        # We need the 2nd moment not the covariance 
        # Since the absolut edisplacement matters
        dM21 = dΣ1r + array([outer(μ,μ) for μ in dμ1r])
        
        # alignement with trial-to-trial variability distribution
        tvar   += [array([expected_intersection_enormed(dμ,orthnormedcovariance(Σ)) 
                        for (dμ,Σ) in zip(Δμ,Σ1r)])]
        tvarch += [array([expected_intersection_enormed_chance(dμ,orthnormedcovariance(Σ)) 
                        for (dμ,Σ) in zip(Δμ,Σ1r)])]
                        
        # alignment with coding (location gradient) distribution
        cvar   += [array([expected_intersection_enormed(dμ,orthnormedcovariance(Σ)) 
                        for (dμ,Σ) in zip(Δμ,dM21)])]
        cvarch += [array([expected_intersection_enormed_chance(dμ,orthnormedcovariance(Σ)) 
                        for (dμ,Σ) in zip(Δμ,dM21)])]
    return tvar,tvarch,cvar,cvarch


def alignment_angle(Δμ,Σ):
    '''
    Computed generalzation of alignment angle between a vector
    and a covariance matrix
    '''
    # make unitless and compute complementary space
    N  = Σ.shape[0]
    Δμ = unit_length(Δμ)
    Σ  = normedcovariance(Σ)
    Φ  = orthnormedcovariance(Σ)
    # Alignment statistic and chance-normalized
    x1 = expected_intersection(Δμ,Σ)**0.5
    # Non-alignment statistic and chance-normalized
    y1 = expected_intersection(Δμ,Φ)**0.5
    # Compute subspace alignment angle
    # Smaller = more aligned
    θ = np.angle(x1+y1*1j)*180/np.pi
    return θ

def sample_alignment_angle(Σ,K=10):
    '''
    Computed generalzation of alignment angle between a vector
    and a covariance matrix
    '''
    # make unitless and compute complementary space
    N  = Σ.shape[0]
    Σ  = normedcovariance(Σ)
    Φ  = orthnormedcovariance(Σ)
    # sample chance level
    # get random unit vectors then compute
    # expected projection of drift onto variability
    V = unit_length(np.random.randn(K,N),axis=1)
    x_chance = np.array([v.T@Σ@v for v in V])**0.5
    y_chance = np.array([v.T@Φ@v for v in V])**0.5
    # Compute subspace alignment angle
    # Smaller = more aligned
    θ = np.angle(x_chance+y_chance*1j)*180/np.pi
    # This works too; seems the same and is a bit faster
    #x0 = (np.trace(Σ)/N)**0.5
    #y0 = (np.trace(Φ)/N)**0.5
    #θ = np.angle(x0+y0*1j)*180/np.pi
    return θ

def sample_alignment_self(Σ,K=10):
    '''
    Self-alignment of distribution
    Sanity check for baseline
    '''
    # make unitless and compute complementary space
    N  = Σ.shape[0]
    Σ  = normedcovariance(Σ)
    Φ  = orthnormedcovariance(Σ)
    # Build square-root using eigendecomposition
    e,v = eigh(Σ)
    ch = v@diag(maximum(0,e)**0.5)
    V = unit_length((ch @ randn(N,K)).T,axis=1)
    x_chance = np.array([v.T@Σ@v for v in V])**0.5
    y_chance = np.array([v.T@Φ@v for v in V])**0.5
    return np.angle(x_chance+y_chance*1j)*180/np.pi

def sample_alignment_cross(Σ,Σother,K=10):
    '''
    Self-alignment of distribution
    Sanity check for baseline
    '''
    # make unitless and compute complementary space
    N  = Σ.shape[0]
    Σ  = normedcovariance(Σ)
    Φ  = orthnormedcovariance(Σ)
    # Build square-root using eigendecomposition
    e,v = eigh(Σother)
    ch = v@diag(maximum(0,e)**0.5)
    V = unit_length((ch @ randn(N,K)).T,axis=1)
    x_chance = np.array([v.T@Σ@v for v in V])**0.5
    y_chance = np.array([v.T@Φ@v for v in V])**0.5
    return np.angle(x_chance+y_chance*1j)*180/np.pi

def sample_alignment_complement(Σ,K=10):
    '''
    Self-alignment of distribution
    Sanity check for baseline
    '''
    # make unitless and compute complementary space
    N  = Σ.shape[0]
    Σ  = normedcovariance(Σ)
    Φ  = orthnormedcovariance(Σ)
    # Build square-root using eigendecomposition
    e,v = eigh(Φ)
    ch = v@diag(maximum(0,e)**0.5)
    V = unit_length((ch @ randn(N,K)).T,axis=1)
    x_chance = np.array([v.T@Σ@v for v in V])**0.5
    y_chance = np.array([v.T@Φ@v for v in V])**0.5
    return np.angle(x_chance+y_chance*1j)*180/np.pi

def alignment_angle_unnormalized(Δμ,Σ):
    Δμ = unit_length(Δμ)
    Σ  = normedcovariance(Σ)
    return (Δμ.T@Σ@Δμ)**0.5

def alignment_angle_normalized(Δμ,Σ):
    Σ  = normedcovariance(Σ)
    x1 = alignment_angle_unnormalized(Δμ,Σ)
    x2 = (trace(Σ)/Σ.shape[0])**0.5
    return (x1-x2)/(1-x2)

def sample_alignment_angle_unnormalized(Σ,K=NPERMUTATION):
    N  = Σ.shape[0]
    Σ  = normedcovariance(Σ)
    V  = unit_length(np.random.randn(K,N),axis=1)
    return np.array([v.T@Σ@v for v in V])**0.5

def sample_alignment_angle_normalized(Σ,K=NPERMUTATION):
    Σ  = normedcovariance(Σ)
    xc = sample_alignment_angle_unnormalized(Σ,K)
    x2 = (trace(Σ)/Σ.shape[0])**0.5
    return (xc-x2)/(1-x2)

def sample_alignment_self_unnormalized(Σ,K=NPERMUTATION):
    N  = Σ.shape[0]
    Σ  = normedcovariance(Σ)
    e,v = eigh(Σ)
    ch = v@diag(maximum(0,e)**0.5)
    V  = unit_length((ch @ randn(N,K)).T,axis=1)
    return np.array([v.T@Σ@v for v in V])**0.5

def sample_alignment_self_unnormalized_nounit(Σ,K=NPERMUTATION):
    N  = Σ.shape[0]
    Σ  = normedcovariance(Σ)
    e,v = eigh(Σ)
    ch = v@diag(maximum(0,e)**0.5)
    V  = (ch @ randn(N,K)).T
    return np.array([v.T@Σ@v for v in V])**0.5

def sample_alignment_self_normalized(Σ,K=NPERMUTATION):
    Σ  = normedcovariance(Σ)
    xc = sample_alignment_self_unnormalized(Σ,K)
    x2 = (trace(Σ)/Σ.shape[0])**0.5
    return (xc-x2)/(1-x2)

def sample_alignment_cross_normalized(Σ,Σother,K=NPERMUTATION):
    # make unitless and compute complementary space
    N  = Σ.shape[0]
    Σ  = normedcovariance(Σ)
    e,v = eigh(Σother)
    ch = v@diag(maximum(0,e)**0.5)
    V = unit_length((ch @ randn(N,K)).T,axis=1)
    x_chance = np.array([v.T@Σ@v for v in V])**0.5
    x2 = (trace(Σ)/N)**0.5
    return (x_chance-x2)/(1-x2)

def sample_alignment_complement_normalized(Σ,K=NPERMUTATION):
    # make unitless and compute complementary space
    N  = Σ.shape[0]
    Σ  = normedcovariance(Σ)
    Φ  = orthnormedcovariance(Σ)
    # Build square-root using eigendecomposition
    e,v = eigh(Φ)
    ch = v@diag(maximum(0,e)**0.5)
    V = unit_length((ch @ randn(N,K)).T,axis=1)
    x_chance = np.array([v.T@Σ@v for v in V])**0.5
    x2 = (trace(Σ)/N)**0.5
    return (x_chance-x2)/(1-x2)

def new_alignment_normalized(Δμ,Σ):
    Σ  = normedcovariance(Σ)
    r  = (Δμ@Σ@Δμ.T)/(Δμ.T@Δμ)
    r0 = trace(Σ)/Σ.shape[0]
    r1 = trace(Σ.T@Σ)
    rb = ((r-r0)/(r1-r0))**0.5
    return r,r0,r1,rb

