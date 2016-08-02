#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

def GMM(PCA,NCLASS=2):
    '''
    PDF = Pr(G) (2pi)^(k/2)|S|^(-1/2)exp[-1/2 (x-mu)' S^(-1) (x-mu)]
    logPDF = logPr(G) k/2 log(2pi)-1/2log(|S|)-1/2(x-mu)'S^(-1)(x-mu)
    Pr is inverse monotonic with logPr(G)-log(|S|)-(x-mu)'S^(-1)(x-mu)
    '''
    N        = shape(PCA)[1]
    initsize = N/NCLASS
    classes  = zeros((N,))
    oldclasses  = zeros((N,))
    Pr       = zeros((N,NCLASS))
    partition = (2*pi)**(-0.5*NCLASS)
    for i in range(NCLASS):
        classes[i*initsize:(i+1)*initsize] = i
    for ii in range(2000):
        for i in range(NCLASS):
            c = PCA[:,classes==i]
            Mu = mean(c,1)
            Cm = cov((c.T-Mu).T)
            k  = shape(c)[1]
            Pm = pinv(Cm)
            center = (PCA.T-Mu)
            normalize = partition*k/(N+1.)/sqrt(det(Cm))
            Pr[:,i] = exp(-0.5*array([dot(x,dot(Pm,x.T)) for x in center]))*normalize
        oldclasses[:]=classes
        classes = argmax(Pr,1)
        if all(oldclasses==classes):break
    classification = (Pr.T/sum(Pr,1)).T
    return classification,Pr

def GMM1D(PCA,NCLASS=2):
    PCA      = squeeze(PCA)
    N        = len(PCA)
    initsize = N/NCLASS
    classes  = zeros((N,))
    Pr       = zeros((N,NCLASS))
    partition = (2*pi)**(-0.5*NCLASS)
    for i in range(NCLASS):
        classes[i*initsize:(i+1)*initsize] = i
    for ii in range(2000):
        for i in range(NCLASS):
            c  = PCA[classes==i]
            Mu = mean(c)
            Cm = var(c)
            k  = len(c)
            Pm = 1./Cm
            center = (PCA-Mu)
            normalize = partition*k/(N+1.)/sqrt(Cm)
            Pr[:,i] = exp(-0.5*Pm*center**2)*normalize
        classes = argmax(Pr,1)
    classification = (Pr.T/sum(Pr,1)).T
    return classification,Pr