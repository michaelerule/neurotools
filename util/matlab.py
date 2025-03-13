#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import nested_scopes
"""
Matlab-related helpers; More in util.hdfmat
"""
import numpy as np
from .tools import piper

@piper
def mi(a):
    a = np.array(a)
    if shape(a)==():
        a = a[None][0]
    print(len(a), np.shape(a), a.dtype)
    return a







