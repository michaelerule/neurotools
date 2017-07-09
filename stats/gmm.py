#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np

def GMM(PCA,NCLASS=2):
    '''
    \[ PDF = Pr(G) (2pi)^(k/2)|S|^(-1/2)exp[-1/2 (x-mu)' S^(-1) (x-mu)] \]
    \[ logPDF = logPr(G) k/2 log(2pi)-1/2log(|S|)-1/2(x-mu)'S^(-1)(x-mu) \]
    Pr is inverse monotonic with $logPr(G)-log(|S|)-(x-mu)'S^(-1)(x-mu)$
    '''
    N          = PCA.shape()[1]
    initsize   = N/NCLASS
    classes    = np.zeros((N,))
    oldclasses = np.zeros((N,))
    Pr         = np.zeros((N,NCLASS))
    partition  = (2*pi)**(-0.5*NCLASS)
    for i in range(NCLASS):
        classes[i*initsize:(i+1)*initsize] = i
    for ii in range(2000):
        for i in range(NCLASS):
            c = PCA[:,classes==i]
            Mu = np.mean(c,1)
            Cm = np.cov((c.T-Mu).T)
            k  = np.shape(c)[1]
            Pm = np.pinv(Cm)
            center = (PCA.T-Mu)
            normalize = partition*k/(N+1.)/np.sqrt(np.det(Cm))
            Pr[:,i] = np.exp(-0.5*np.array([np.dot(x,np.dot(Pm,x.T)) for x in center]))*normalize
        oldclasses[:]=classes
        classes = argmax(Pr,1)
        if all(oldclasses==classes):break
    classification = (Pr.T/sum(Pr,1)).T
    return classification,Pr

def GMM1D(PCA,NCLASS=2):
    PCA      = np.squeeze(PCA)
    N        = len(PCA)
    initsize = N/NCLASS
    classes  = np.zeros((N,))
    Pr       = np.zeros((N,NCLASS))
    partition = (2*pi)**(-0.5*NCLASS)
    for i in range(NCLASS):
        classes[i*initsize:(i+1)*initsize] = i
    for ii in range(2000):
        for i in range(NCLASS):
            c  = PCA[classes==i]
            Mu = np.mean(c)
            Cm = np.var(c)
            k  = len(c)
            Pm = 1./Cm
            center = (PCA-Mu)
            normalize = partition*k/(N+1.)/np.sqrt(Cm)
            Pr[:,i] = np.exp(-0.5*Pm*center**2)*normalize
        classes = np.argmax(Pr,1)
    classification = (Pr.T/np.sum(Pr,1)).T
    return classification,Pr
