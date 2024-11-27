#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

'''
A couple image-like subroutines. This may overlap a bit with `stats.spatial`
'''

import numpy as np
import matplotlib.pyplot as plt

try:
    import skimage
    from skimage import data, img_as_float
    from skimage import exposure
except:
    def noskimage(*args,**kwargs):
        raise ImportError("skimage module not loaded")
    data = img_as_float = exposure = noskimage
    
import scipy
blur = scipy.ndimage.filters.gaussian_filter

def median_center(image):
    h = image
    h = (h-np.median(h))+0.5
    h[h<0]=0
    h[h>1]=1
    return h

def mean_center(image):
    h = image
    h = (h-np.mean(h))+0.5
    h[h<0]=0
    h[h>1]=1
    return h

def unitize(image):
    image = image - np.min(image)
    image /= np.max(image)
    return image


def visualize_derivatives(image):
    laplacian = scipy.ndimage.filters.laplace(image)
    lhist = mean_center(
        blur(exposure.equalize_hist(unitize(laplacian)),1))
    plt.imshow(lhist,
        origin='lower',interpolation='nearest',cmap='gray',extent=(0,64,)*2)
    plt.title('Laplacian')
    return gradient, laplacian

def visualize_derivatives(image):
    '''
    Plot gradient on left and Laplacian on right.
    Only tested on 2D 1-channel float imags
    '''
    dx1,dy1 = np.gradient(image)
    gradient = dx1 + 1j*dy1
    a1 = np.abs(gradient)
    plt.figure(None,(12,6))
    plt.subplot(121)
    a1 = mean_center(blur(exposure.equalize_hist(unitize(a1)),1))
    plt.imshow(a1,
        origin='lower',interpolation='nearest',cmap='gray',extent=(0,64,)*2)
    plt.title('Gradient Magnitude')
    plt.subplot(122)
    laplacian = scipy.ndimage.filters.laplace(image)
    lhist = mean_center(
        blur(exposure.equalize_hist(unitize(laplacian)),1))
    plt.imshow(lhist,
        origin='lower',interpolation='nearest',cmap='gray',extent=(0,64,)*2)
    plt.title('Laplacian')
    return gradient, laplacian

