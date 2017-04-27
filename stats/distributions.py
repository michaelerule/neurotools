#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

'''
Functions for computing the log-PDF of common distributions.
These yield a more digits of precision than their counterparts in
scipy.stats by computing log-probability values using high precision
128-bit floats.
'''

import numpy as np
from neurotools.functions import log_factorial, slog

def poisson_logpdf(k,l):
    '''
    Gives the log-pdf for a poisson distribution with rate l 
    evaluated at points k. k should be a vector of integers.
    '''
    # k,l = map(np.float128,(k,l))
    return k*slog(l)-l-np.array([log_factorial(x) for x in k])

def poisson_pdf(k,l):
    return np.exp(poisson_logpdf(k,l))

# log(sqrt(2*pi)) computed to high precision on Wolfram Alpha.
logsqrt2pi = np.float128(
    '0.91893853320467274178032973640561763986139747363778341281')

def gaussian_logpdf(mu,sigma,x):
    '''
    Non-positive standar deviations will be clipped
    '''
    mu,sigma,x = map(np.float128,(mu,sigma,x))
    x = (x-mu)/sigma
    return -0.5*x*x - slog(sigma) - logsqrt2pi

def gaussian_pdf(mu,sigma,x):
    return np.exp(gaussian_logpdf(mu,sigma,x))
