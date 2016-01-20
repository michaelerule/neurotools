

try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import *
    from pyfftw.interfaces.numpy_fft import fft as ftw
    ftwthr = lambda x:ftw(x,threads=__N_CPU__-2)
except:
    print 'FFTW LIBRARY MISSING, USING NUMPY'
    print 'PLEASE INSTALL FFTW USING'
    print '> sudo apt-get install python-fftw'
    print 'OR'
    print '> sudo pip install fftw'
    import numpy
    from numpy.fft import *
    ftw = numpy.fft
    ftwthr = ftw




