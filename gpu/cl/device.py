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

import numpy as np
import pyopencl as cl

#from orix.cl import *
#from orix.cl.function import *
#zero_device = elemental("int *d","d=0")

def zeros_float(N):
    arr = np.zeros(N);
    buf = cl.Buffer(ctx, mf.READ_WRITE|mf.USE_HOST_PTR, hostbuf=arr)
    return buf
