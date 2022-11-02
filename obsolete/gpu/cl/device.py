#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np

try:
    import pyopencl as cl
except:
    import sys
    def missing(*args,**kwargs):
        if 'sphinx' in sys.modules:
            print('Please locate and install the pyOpenCL GPU library')
        else:
            raise ValueError('Please locate and install pyOpenCL GPU library')
    # TODO: shadow missing function with the above, which raises an error?
    cl = None

from neurotools.obsolete.gpu.cl import *
from neurotools.obsolete.gpu.cl.function import *
zero_device = elemental("int *d","d=0")

def zeros_float(N):
    arr = np.zeros(N);
    buf = cl.Buffer(ctx, mf.READ_WRITE|mf.USE_HOST_PTR, hostbuf=arr)
    return buf
