

import matplotlib.pyplot as mp
import scipy.stats

def fit_vonmises(z):
    scipy.stats.distributions.vonmises.a = -numpy.pi
    scipy.stats.distributions.vonmises.b = numpy.pi
    theta    = angle(mean(z))
    dephased = z*exp(-1j*theta)
    location,_,scale = scipy.stats.distributions.vonmises.fit(angle(dephased))
    return location,theta,scale

