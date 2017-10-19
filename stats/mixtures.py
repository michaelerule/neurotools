#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
from neurotools.system import *

'''
Functions relating to distributions. Most of this should be available
in numpy, scipy, and scikits, but occassionally we need our own
implementations of things.
'''

import numpy as np
from neurotools.stats.distributions import poisson_logpdf, poisson_pdf

def two_class_poisson_mixture_model(counts):
    '''
    Estimates a Poisson mixture model with two distributions
    Originally written as a toy example
    '''
    # Start with the hypothesis that the top 50% of the data and
    # bottom 50% are drawn from different distributions. Initialize
    # the means (lambda) Of each poisson distribution from the means
    # of these classes.
    N = len(counts)

    # Buffer to store updated class estimates
    classes     = np.zeros((N,),'int')
    new_classes = np.int32(counts>np.median(counts))

    # Iterate until the class labels do not change
    nIter = 0
    while not np.all(classes==new_classes):
        classes[:] = new_classes

        # Re-estimate distribution parameters based on the proposed
        # classes. To define the mixture model we need the means
        # (lambda) of each Poisson distribution, as well as the
        # weights of each distribution in the mixture (pr1 and pr0
        # here)
        mu0 = np.mean(counts[classes==0])
        mu1 = np.mean(counts[classes==1])
        pr1 = np.mean(classes)
        pr0 = 1.-pr1

        # We must compute the likelihood that each observation comes from each
        # distribution, so we'll need the Poisson likelihood.
        # Since we'll just be comparing two likelihoods, any quantity that is
        # monotonically related to this likelihood will also work.
        # The poisson likelihood is
        # \[
        # \Pr(x;\lambda) = \frac{\lambda^x}{x!} exp(-\lambda)
        # \]
        l0, l1 = mu0, mu1

        # For all of our comparisons, x will be fixed (we'll be testing the same
        # point against different distributions) so we can skip the factor x. We
        # can also take the logarithm of this expression for better numerical
        # stability, and use
        # \[
        # f(x;\lambda) = x \cdot \ln(\lambda) - \lambda
        # \]
        lnl0, lnl1 = np.log(1e-6+l0), np.log(1e-6+l1)

        # If we want to allow each distribution in the mixture a different
        # weight we need to add that in to the probability as a
        # multiplicative parameter. This comes out as an addative parameter.
        # \[
        # f(x;\lambda) = x \cdot \ln(\lambda) - \lambda + \ln(\omega)
        # \]
        lnw0, lnw1 = np.log(1e-6+pr0), np.log(1e-6+pr1)

        # When comparing two classes, we really just need to know which
        # probability is larger. It suffices to compute the difference
        # between the log probabilities
        # \[
        # x \cdot \ln(\lambda_1/\lambda_0)
        # - (\lambda_1-\lambda_0)
        # + \ln(\omega_1/\omega_0)
        # \]
        # This can be factored into a simple multiplier and consant
        multiplier = (lnl1-lnl0)
        constant   = (lnw1-lnw0)-(l1-l0)

        new_classes = np.int32(counts*multiplier+constant>0)

        nIter += 1
        print(nIter,'iterations')
        print(mu0,mu1,pr0,pr1)

    return classes,mu0,mu1,pr0,pr1
