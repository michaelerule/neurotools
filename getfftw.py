#!/usr/bin/python
# -*- coding: UTF-8 -*-
# The above two lines should appear in all python source files!
# It is good practice to include the lines below
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
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
    print('FFTW LIBRARY MISSING, USING NUMPY')
    print('PLEASE INSTALL FFTW USING')
    print('> sudo apt-get install python-fftw')
    print('OR')
    print('> sudo pip install fftw')
    import numpy
    from numpy.fft import *
    ftw = numpy.fft
    ftwthr = ftw
