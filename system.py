#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

'''
Collection of misc utilities
'''

import sys
__IS_PY3__ = sys.version_info>=(3,0)

if __IS_PY3__:
    def execfile(filepath, globals=None, locals=None):
        '''
        http://stackoverflow.com/questions/
        436198/what-is-an-alternative-to-execfile-in-python-3
        '''
        if globals is None:
            globals = {}
        globals.update({
            "__file__": filepath,
            "__name__": "__main__",
        })
        import os
        with open(filepath, 'rb') as file:
            exec(compile(file.read(), filepath, 'exec'), globals, locals)

    # create raw_input alias
    raw_input = input
