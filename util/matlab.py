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

    if hasattr(a,'items') and hasattr(a,'keys') and hasattr(a,'values'):
        print("Dictionary:")
        for v in a.values():
            print("  "+v)
        return a

    try:
        a = np.array(a)
        if np.shape(a)==():
            a = a[None][0]
        print("length=%d"%len(a), "shape=%s"%(np.shape(a),),"type=%s"%a.dtype)
    except:
        print("Not a numpy array?")
    return a







