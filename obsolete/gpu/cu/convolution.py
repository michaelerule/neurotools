#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

'''
Utilities for performing certain Naive convolutions in PyCuda. Sadly I 
have not accelerated anything using the FT yet
'''

def gpuboxconv_core(data,size):
    '''
    
    '''
    cells=int(len(data))
    timepoints=int(len(data[0]))
    data=gpuint(flat(data))
    newtimepoints=int(timepoints-size)
    n=int(newtimepoints*cells)
    if (n<=0) :
        print("ERROR : VALID POST-CONVOLUTION SIZE IS NEGATIVE!!!")
        return None
    newdata=gpuarray.zeros(n,np.int32)
    kernel('int *destination, int *source, int size, int cells, int timepoints','''
        const int newtimepoints = timepoints-size+1;
        const int cellID        = tid/newtimepoints;
        const int offset        = tid%newtimepoints;
        int *buffer             = &source[cellID*timepoints+offset];
        int sum = 0;
        for (int j=0; j<size; j++)
            sum += buffer[j];
        destination[tid]=sum;
    ''')(n)(newdata, data, np.int32(size), np.int32(cells), np.int32(timepoints))
    cpudata = cpu(newdata)
    del newdata
    del data
    return cut(cpudata,newtimepoints)

gpuboxconv = lambda size:lambda data:flat([gpuboxconv_core(data[i:i+MAXPROCESS],size) for i in xrange(0,len(data),MAXPROCESS)])


