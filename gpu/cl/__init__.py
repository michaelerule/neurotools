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

"""
OpenCL utility routines
"""

import numpy
try:
    import pyopencl
except:
    print('pyopencl is not installed!')
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



