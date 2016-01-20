"""
Commonly used functions
"""

from numpy import *

def softhresh(x):
    return 1./(1+exp(-x))

def npdf(mu,sigma,x):
    partition = 1./(sigma*sqrt(2*pi))
    x = (x-mu)/sigma
    return partition * exp(-0.5*x**2)
