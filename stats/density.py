
from scipy.stats import gaussian_kde
from numpy import *
import numpy as np

def kdepeak(x, x_grid=None):
    if x_grid==None:
        x_grid = np.linspace(np.min(x),np.max(x),201)
    kde = gaussian_kde(x)
    return x_grid,kde.evaluate(x_grid)

def knn_1d_density(x,k=10,eps=0.01):
    '''
    Uses local K nearest neighbors to estimate a density and center of
    mass at each point in a distribution

    Let's assume x are in units of time with 1ms bin size.
    intervals are in ms, centers are in ms
    kernel is unit mass and estimates local average interval and location
    so centers are in ms, and density is in 1/ms
    so to convert density to Hz you need a scale factor of 100

    returns a local density estimator in units of 1/input_units
    '''
    x=np.sort(x)
    intervals = np.diff(x)
    centers   = (x[1:]+x[:-1])*0.5
    kernel = np.hanning(k)
    kernel /=sum(kernel)
    intervals = np.convolve(intervals,kernel,'same')
    #centers   = convolve(centers  ,kernel,'valid')
    return centers,(eps+1.0)/(eps+intervals)

def adaptive_density_grid(grid,x,k=10,eps=0.01,fill=None):
    '''
    Follow the knn_1d_density estimation with interpolation of the
    density on a grid

    fill: if not given will fill with the mean rate
    '''
    centers,density = knn_1d_density(x,k,eps=eps)
    if len(centers)!=len(density):
        warn('something is wrong')
        warn(len(centers),len(density))
        N = min(len(centers),len(density))
        centers = centers[:N]
        density = density[:N]
    if fill is None: fill=mean(density)
    y = np.interp1d(centers,density,bounds_error=0,fill_value=fill)(grid)
    return y

def knn_1d_density_density(x,k=10,eps=0.01):
    '''
    Uses local K nearest neighbors to estimate a density and center of
    mass at each point in a distribution

    Let's assume x are in units of time with 1ms bin size.
    intervals are in ms, centers are in ms
    kernel is unit mass and estimates local average interval and location
    so centers are in ms, and density is in 1/ms
    so to convert density to Hz you need a scale factor of 100
    '''
    x=sort(x)

    weights = diff(append(-1,find(append(diff(x),1)>0)))
    x = unique(x)
    weights   = (weights[1:]+weights[:-1])*0.5

    intervals = diff(x)
    centers   = (x[1:]+x[:-1])*0.5
    kernel    = hanning(k)
    kernel   /= sum(kernel)
    intervals = convolve(intervals*weights,kernel,'same')
    reweights = convolve(weights,kernel,'same')
    intervals /= reweights
    int
    #centers  = convolve(centers  ,kernel,'valid')
    return centers,(eps+1.0)/(eps+intervals)


def gridhist(ngrid,width,points):
    '''
    Please use numpy.histogram2d instead!
    '''
    quantized = np.int32(points*ngrid/width)
    counts = np.zeros((ngrid,ngrid),dtype=int32)
    for (x,y) in quantized:
        counts[x,y]+=1
    return counts
