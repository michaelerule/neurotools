#!/usr/bin/python
# -*- coding: UTF-8 -*-
# BEGIN PYTHON 2/3 COMPATIBILITY BOILERPLATE
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
import sys
if sys.version_info<(3,):
    from itertools import imap as map
# END PYTHON 2/3 COMPATIBILITY BOILERPLATEion

'''
A couple image-like subroutines. This may overlap a bit with `stats.spatial`
'''

import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage import exposure

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

