#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import print_function

'''
Utilities related to spatial kernels
'''

def laplace_kernel():
    return array([[  0.5,   2. ,   0.5],
       [  2. , -10. ,   2. ],
       [  0.5,   2. ,   0.5]])/3.

def laplacian(x):
    '''
    Graph laplacian of a 2D mesh with absorbing boundary
    
    In the middle
    
     0  1  0
     1 -4  1
     0  1  0
    
    At edges
    
     1  0
    -3  1
     1  0
    
    At corners
    
    -2  1
     1  0
     
    Test code:
    
    >>> test = zeros((5,11),'float32')
    >>> test[2,5] = 1
    >>> showim(test)
    >>> showim(laplacian(test))
    '''
    n,m = x.shape
    
    # Middle cases
    result = copy(x)*-4
    
    # Edge cases
    result[ 0, :] = x[0,:]*-3
    result[ :, 0] = x[:,0]*-3
    result[-1, :] = x[-1,:]*-3
    result[ :,-1] = x[:,-1]*-3
    
    # Corner cases
    result[ 0, 0] = x[ 0, 0]*-2
    result[ 0,-1] = x[ 0,-1]*-2
    result[-1, 0] = x[-1, 0]*-2
    result[-1,-1] = x[-1,-1]*-2
    
    # Add neighbors
    result[1: , :]   += x[ :-1,:]
    result[:  ,1:]   += x[ :,  :-1]
    result[:-1, :]   += x[1:,  :]
    result[:  , :-1] += x[ :, 1:]
    
    return result

def gaussian_2D_kernel(sigma):
    '''
    Generate 2D Gaussian kernel as product of 2 1D kernels
    
    >>> showim(gaussian_2D_kernel(1))
    '''
    radius  = int(ceil(sigma*3))
    support = 1+2*radius
    kernel  = ones((support,support),'double')
    kern_1D = exp(-arange(-radius,radius+1)**2/(2*sigma**2))
    kernel *= kern_1D[:,None]
    kernel *= kern_1D[None,:]
    kernel /= sum(kernel)
    return kernel

def absorbing_gaussian(x,sigma):
    support = 1+sigma*6
    normalization = zeros(x.shape,'double')
    result = zeros(x.shape,'double')
    kernel = gaussian_2D_kernel(sigma)
    return convolve2d(x, kernel, mode='same', boundary='symm')

def laplace_kernel():
    return array([[  0.5,   2. ,   0.5],
       [  2. , -10. ,   2. ],
       [  0.5,   2. ,   0.5]])/3.

def absorbing_laplacian(x):
    kernel = laplace_kernel()
    return convolve2d(x, kernel, mode='same', boundary='symm')
    
    

