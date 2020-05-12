#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
try:
    try:
        import pyfftw
    except:
        import fftw3 as pyfftw
    from pyfftw.interfaces.numpy_fft import *
    from pyfftw.interfaces.numpy_fft import fft as ftw
    ftwthr = lambda x:ftw(x,threads=__N_CPU__-2)
except:
    #print('could not locate fftw library, falling back to numpy')
    import numpy
    from numpy.fft import *
    ftw = numpy.fft
    ftwthr = ftw
