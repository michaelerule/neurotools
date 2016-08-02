
from scipy.stats import gaussian_kde
from numpy import *
import numpy as np
from neurotools.signal.signal import get_edges

def kdepeak(x, x_grid=None):
    if x_grid==None:
        x_grid = np.linspace(np.min(x),np.max(x),201)
    kde = gaussian_kde(x)
    return x_grid,kde.evaluate(x_grid)


def knn_1d_density(x,k=10,eps=0.01):
    '''
    Uses local K nearest neighbors to estimate a density and center of
    mass at each point in a distribution. Returns a local density estimator in units of 1/input_units. For example, if a sequence
    of times in seconds is provided, the result is an estimate of
    the continuous time intensity function in units of Hz.

    Parameters
    ----------
    x : ndarray
        List of points to model
    k : integer
        Number of nearest neighbors to use in local density estimate
        Default is 10
    eps : number
        Small correction factor to avoid division by zero

    Returns
    -------
    centers : ndarray
        Point location of density estimates
    density :
        Density values at locations of centers
    '''
    x=np.float64(np.sort(x))
    # Handle duplicates by dithering
    duplicates = get_edges(np.diff(x)==0.)+1
    duplicates[duplicates>=len(x)-1]=len(x)-2
    duplicates[duplicates<=0]=1
    for a,b in zip(*duplicates):
        n = b-a+1
        q0 = x[a]
        q1 = (x[a-1]-q0)
        q2 = (x[b+1]-q0)
        #print(a,b,q0,q1,q2)
        x[a:b+1] += np.linspace(q1,q2,n+2)[1:-1]
    intervals = np.diff(x)
    centers   = (x[1:]+x[:-1])*0.5
    kernel    = np.hanning(min(x.shape[0]-1,k)+2)[1:-1]
    kernel   /=sum(kernel)
    intervals = np.convolve(intervals,kernel,'same')
    density = (eps+1.0)/(eps+intervals)
    return centers,density

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

def gridhist(ngrid,width,points):
    '''
    Please use numpy.histogram2d instead!
    '''
    quantized = np.int32(points*ngrid/width)
    counts = np.zeros((ngrid,ngrid),dtype=int32)
    for (x,y) in quantized:
        counts[x,y]+=1
    return counts
