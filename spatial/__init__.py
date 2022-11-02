#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

from . import array
from . import dct
from . import distance
from . import fftzeros
from . import geometry
from . import kernels
from . import masking
from . import phase
from . import points
from . import spiking
from . import triangulation


def brute_force_local_2d_maxima(x,R=5):
    '''
    Find points higher than all neighbors within 
    radius r in a 2D array. 
    
    Parameters
    ----------
    x: 2D np.array; Signal in which to locate local maxima.
    R: int; RxR region in which a peak must be a local maxima to be included.
    
    Returns
    -------
    (x,y): tuple of np.int32 arrays with peak coordinates.
    '''
    R   = int(np.ceil(R))
    pad = 2*R
    x   = np.array(x)
    h,w = x.shape
    
    padded = np.zeros((h+pad,w+pad))
    padded[R:-R,R:-R] = x
    
    best = np.full(x.shape,-inf)
    RR = R*R
    for dr in np.arange(-r,r+1):
        for dc in np.arange(-r,r+1):
            if dr==dc==0: continue
            if dr*dr+dc*dc>RR: continue
            best = np.maximum(best, padded[R+dr:R+dr+h,R+dc:R+dc+w])
    
    ispeak = x>=best
    return np.where(ispeak)[::-1]
    
