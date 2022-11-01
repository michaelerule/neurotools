#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Utilities related to spatial analysis of spiking data
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
from scipy.signal import convolve2d

def interp_bin(xypoints,n,eps=1e-9):
    '''
    Parameters
    ----------
    xypoints : np.array
        npoints x 2 array of point locations, all points must be in [0,1]²
    n : positive int
        number of bins; binning grid is n x n 2D
    eps : float
        Default is 1e-9; small padding to prevent points form falling outside boundary

    Returns
    -------
    hist : np.array
        n x n 2D histogram, computed using interpolation
    '''
    hist    = np.zeros((n,n));
    if np.prod(xypoints.shape)<=0:
        return hist
    eps = 1e-9
    x   = xypoints[:,0]*n+0.5;
    y   = xypoints[:,1]*n+0.5;
    x   = np.minimum(np.maximum(x,1+eps),n-eps);
    y   = np.minimum(np.maximum(y,1+eps),n-eps);
    ix  = np.int32(np.floor(x));
    iy  = np.int32(np.floor(y));
    fx  = x-ix;
    fy  = y-iy;
    p22 = fx*fy;
    p21 = fx*(1-fy);
    p12 = fy*(1-fx);
    p11 = (1-fx)*(1-fy);
    npoints = xypoints.shape[0];
    for j in range(npoints):
        jx = ix[j];
        jy = iy[j];
        hist[jx-1,jy-1] += p11[j];
        hist[jx-1,jy  ] += p12[j];
        hist[jx  ,jy-1] += p21[j];
        hist[jx  ,jy  ] += p22[j];
    hist = hist.T;
    return hist
    
    
