#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

"""
OpenCL utility routines
"""

import numpy
try:
    import pyopencl
except:
    import sys
    def missing(*args,**kwargs):
        if 'sphinx' in sys.modules:
            print('Please locate and install the pyOpenCL GPU library')
        else:
            raise ValueError('Please locate and install pyOpenCL GPU library')
    # TODO: shadow missing function with the above, which raises an error?
    pyopencl = None

if pyopencl:
    print('Detecting PyOpenCL platforms...')
    for platform in pyopencl.get_platforms():
        print('\t',platform)

if pyopencl:
    platforms = pyopencl.get_platforms()
    ctx   = pyopencl.Context(
            dev_type=pyopencl.device_type.ALL,
            properties=[(pyopencl.context_properties.PLATFORM, platforms[0])])
    queue = pyopencl.CommandQueue(ctx)
    mf    = pyopencl.mem_flags



