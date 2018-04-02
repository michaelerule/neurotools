#!/usr/bin/python
# -*- coding: UTF-8 -*-
# The above two lines should appear in all python source files!
# It is good practice to include the lines below
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import print_function
'''
Discrete cosine transform methods.

Primarily used with fftzeros code for finding critical points in phase gradient maps.
'''

from neurotools import getfftw as fft
import numpy as np
from pylab import find
from neurotools.signal.conv import reflect2D
from neurotools.signal.conv import reflect2D_1

def get_mask_antialiased(h_w,aa,spacing,cutoff):
    '''
    Computes a frequency space mask for (h,w) shaped domain with cutoff
    frequency and some anti-aliasing factor as specified by aa.

    Parameters
    ----------
    h : int
        height
    w : int
        width
    aa : int
        antialiasing supsampling factor
    spacing : numeric
        array spacing in mm
    cutoff : numeric
        cutoff scale in mm

    Retruns:
        h*2 x w*2 symmetric mask to be used with DCT for smoothing
    '''
    h,w = h_w # patch migrating from 2.7 to 3
    assert(aa%2==1)
    f1     = fft.fftfreq(h*2*aa,spacing)
    f2     = fft.fftfreq(w*2*aa,spacing)
    ff     = np.abs(outer_complex(f1,f2))
    radius = 1./cutoff
    sup    = np.int32(ff<=radius)
    sup    = np.roll(sup,aa*h+aa//2,0)
    sup    = np.roll(sup,aa*w+aa//2,1)
    mask   = np.array([[np.mean(sup[j*aa:(j+1)*aa,i*aa:(i+1)*aa]) for i in range(w*2)] for j in range(h*2)])
    mask   = fft.fftshift(mask)
    return mask

def get_mask(h_w,spacing,cutoff):
    '''
    Args:
        height: height of array
        width: width of array
        spacing: array spacing in mm
        cutoff: scale in mm
    return: h*2 x w*2 symmetric mask to be used with DCT for smoothing
    '''
    h,w = h_w # patch migrating from 2.7 to 3
    f1     = fft.fftfreq(h*2,spacing)
    f2     = fft.fftfreq(w*2,spacing)
    ff     = np.abs(outerComplex(f1,f2))
    radius = 1./cutoff
    sup    = np.int32(ff<=radius)
    return sup

def dct_cut(data,cutoff,spacing=0.4):
    '''
    Low-pass filters image data by discarding high frequency Fourier
    components.
    Image data is reflected before being processes (i.e.) mirrored boundary
    conditions.
    TODO: I think there is some way to compute the mirrored conditions
    directly without copying the image data.

    This function has been superseded by dct_cut_antialias, which is more
    accurate.
    '''
    print('WARNING DEPRICATED USE dct_cut_antialias')
    h,w    = np.shape(data)[:2]
    mask   = get_mask((h,w),spacing,cutoff)
    mirror = reflect2D(data)
    ff2    = fft.fft2(mirror,axes=(0,1))
    cut    = (ff2.T*mask.T).T # weird shape broadcasting constraints
    result = fft.ifft2(cut,axes=(0,1))[:h,:w,...]
    return result

def outer_complex(a,b):
    '''
    Not quite outer product, intead of a_i * b_k its
    a_i + 1j * b_k
    which I guess is like
    log(outer(exp(a_i),exp(1j*b)))
    '''
    return a[...,None]+1j*b[None,...]

def dct_cut_antialias(data,cutoff,spacing=0.4):
    '''
    Uses brute-force supsampling to anti-alias the frequency space sinc
    function, skirting numerical issues that derives either from
    attempting to evaulate the radial sinc function on a small 2D domain,
    or select a constant frequency cutoff in the frequency space.
    '''
    '''
    # Sinc experiment

    M,N    = 60,80
    data   = randn(M,N)+1j*randn(M,N)
    h,w    = np.shape(data)[:2]

    aa     = 7
    assert(aa%2==1)
    f1     = fftfreq(h*2*aa,spacing)
    f2     = fftfreq(w*2*aa,spacing)
    ff     = abs(outerComplex(f1,f2))
    radius = spacing/cutoff
    sup    = int32(ff<=radius)
    #sup   = fftshift(sup)
    sup    = np.roll(sup,aa*h+aa//2,0)
    sup    = np.roll(sup,aa*w+aa//2,1)
    mask   = array([[mean(sup[j*aa:(j+1)*aa,i*aa:(i+1)*aa]) for i in range(w*2)] for j in range(h*2)])
    mask   = fftshift(mask)

    f1B    = fftfreq(h*2,spacing)
    f2B    = fftfreq(w*2,spacing)
    ffB    = abs(outerComplex(f1B,f2B))
    maskB  = int32(ffB<=radius)

    mirror = reflect2D(data)
    ff2    = fft2(mirror,axes=(0,1))
    cut    = (ff2.T*mask.T).T # weird shape broadcasting constraints
    result = ifft2(cut,axes=(0,1))[:h,:w,...]

    subplot(231)
    imshow(real(sup),interpolation='nearest')
    subplot(232)
    kern   = fft2(mask,axes=(0,1))
    imshow(fftshift(real(kern)),interpolation='nearest')
    subplot(233)
    imshow(fftshift(mask),interpolation='nearest')
    subplot(234)
    imshow(real(result),interpolation='nearest')
    subplot(235)
    imshow(fftshift(real(maskB)),interpolation='nearest')
    '''
    h,w    = np.shape(data)[:2]
    mask   = get_mask_antialiased((h,w),7,spacing,cutoff)
    mirror = reflect2D(data)
    ff2    = fft.fft2(mirror,axes=(0,1))
    cut    = (ff2.T*mask.T).T # weird shape broadcasting constraints
    result = fft.ifft2(cut,axes=(0,1))[:h,:w,...]
    return result

def dct_cut_downsampled(data,cutoff,spacing=0.4):
    '''
    like dctCut but also lowers the sampling rate, creating a compact
    representation from which the whole downsampled data could be
    recovered
    '''
    h,w    = np.shape(data)[:2]
    f1     = fft.fftfreq(h*2,spacing)
    f2     = fft.fftfreq(w*2,spacing)
    wl     = 1./np.abs(reshape(f1,(2*h,1))+1j*f2.reshape(1,2*w))
    mask   = np.int32(wl>=cutoff)
    mirror = reflect2D(data)
    ff     = fft.fft2(mirror,axes=(0,1))
    cut    = (ff.T*mask.T).T # weird shape broadcasting constraints
    empty_cols = find(np.all(mask==0,0))
    empty_rows = find(np.all(mask==0,1))
    delete_col = len(empty_cols)/2 #idiv important here
    delete_row = len(empty_rows)/2 #idiv important here
    keep_cols  = w-delete_col
    keep_rows  = h-delete_row
    col_mask = np.zeros(w*2)
    col_mask[:keep_cols] =1
    col_mask[-keep_cols:]=1
    col_mask = col_mask==1
    row_mask = np.zeros(h*2)
    row_mask[:keep_rows] =1
    row_mask[-keep_rows:]=1
    row_mask = row_mask==1
    cut = cut[row_mask,...][:,col_mask,...]
    w,h = keep_cols,keep_rows
    result = fft.ifft2(cut,axes=(0,1))[:h,:w,...]
    return result

def dct_upsample(data,factor=2):
    '''
    Uses the DCT to supsample array data. Nice for visualization. Uses a
    discrete cosine transform to smoothly supsample image data. Boundary
    conditions are handeled as reflected.

    Data is made symmetric, fourier transformed, then inserted into the
    low-frequency components of a larger array, which is then inverse
    transformed.

    This could probably be optimized with a customized array implimentation
    to avoid the copying.

    Parameters:
        data (ndarray): frist two dimensions should be height and width.
            data may contain aribtrary number of additional dimensions
        factor (int): supsampling factor.

    Test code
    grid = arange(10)&1
    grid = grid[None,:]^grid[:,None]
    amp  = dct_upsample(grid,supSAMPLE).real
    '''
    h,w = np.shape(data)[:2]
    mirrored = reflect2D_1(data)
    ff = fft.fft2(mirrored,axes=(0,1))
    h2,w2 = np.shape(mirrored)[:2]
    newshape = (h2*factor,w2*factor) + np.shape(data)[2:]
    result = np.zeros(newshape,dtype=ff.dtype)
    H = h+1#have to add one more on the left to grab the nyqist term
    W = w+1
    result[:H ,:W ,...]=ff[:H , :W,...]
    result[-h:,:W ,...]=ff[-h:, :W,...]
    result[:H ,-w:,...]=ff[:H ,-w:,...]
    result[-h:,-w:,...]=ff[-h:,-w:,...]
    result = fft.ifft2(result,axes=(0,1))
    result = result[:h*factor-factor+1,:w*factor-factor+1,...]
    return result*(factor*factor)

def iterated_upsample(data,niter=1):
    for i in range(niter):
        data = dct_upsample(data)
    return data



def dct_upsample_notrim(data,factor=2):
    '''
    Uses the DCT to supsample array data. Nice for visualization. Uses a
    discrete cosine transform to smoothly supsample image data. Boundary
    conditions are handeled as reflected.

    Data is made symmetric, fourier transformed, then inserted into the
    low-frequency components of a larger array, which is then inverse
    transformed.

    This could probably be optimized with a customized array implimentation
    to avoid the copying.

    Parameters:
        data (ndarray): frist two dimensions should be height and width.
            data may contain aribtrary number of additional dimensions
        factor (int): supsampling factor.
    '''
    print('ALIGNMENT BUG DO NOT USE YET')
    assert 0
    h,w      = np.shape(data)[:2]
    mirrored = reflect2D_1(data)
    ff       = fft.fft2(mirrored,axes=(0,1))
    h2,w2    = np.shape(mirrored)[:2]
    newshape = (h2*factor,w2*factor) + np.shape(data)[2:]
    result   = np.zeros(newshape,dtype=ff.dtype)
    H = h+1#have to add one more on the left to grab the nyqist term
    W = w+1
    result[:H ,:W ,...]=ff[:H , :W,...]
    result[-h:,:W ,...]=ff[-h:, :W,...]
    result[:H ,-w:,...]=ff[:H ,-w:,...]
    result[-h:,-w:,...]=ff[-h:,-w:,...]
    result = fft.ifft2(result,axes=(0,1))
    result = result[0:h*factor+factor,0:w*factor+factor,...]
    return result*(factor*factor)
