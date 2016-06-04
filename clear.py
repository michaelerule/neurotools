#!/usr/bin/python
# -*- coding: UTF-8 -*-
# The above two lines should appear in all python source files!
# It is good practice to include the lines below
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import print_function

'''
Having unused variables floating around the global namespace can be a source
of bugs.

In MATLAB, it is idiomatic to call "close all; clear all;" at the beginning
of a script, to ensure that previously defined globals don't cuase
surprising behaviour.

This stack overflow post addresses this problem

http://stackoverflow.com/questions/3543833/
how-do-i-clear-all-variables-in-the-middle-of-a-python-script

This module defines the functions saveContext (aliased as clear) and
restoreContext that can be used to restore the interpreter to a state
closer to initialization.

this has not been thoroughly tested.

'''

__saved_context__ = {}

def saveContext():
    import sys
    __saved_context__.update(sys.modules[__name__].__dict__)

clear = saveContext

def restoreContext():
    import sys
    names = sys.modules[__name__].__dict__.keys()
    for n in names:
        if n not in __saved_context__:
            del sys.modules[__name__].__dict__[n]
