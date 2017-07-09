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
Collected code for data analysis within the PyCuda framework. 
Code quality is poor, do not use
'''

try:
    import pycuda
    import pycuda.driver as cuda
    import pycuda.compiler
    import pycuda.tools
except:
    print('PyCuda is missing; please install it to use CUDA routines')
    pycuda = cuda = None

if cuda:
    cuda.init()

    # We will use the current context if one is initialized, otherwise we create
    # a new context.
    #ctx = pycuda.tools.make_default_context()
    dev = pycuda.tools.get_default_device()
    ctx = dev.make_context()

else:
    print('PyCuda is missing; please install it to use CUDA routines')
    dev = ctx = None
