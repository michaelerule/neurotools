#!/usr/bin/python
# -*- coding: UTF-8 -*-
# The above two lines should appear in all python source files!
# It is good practice to include the lines below
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import print_function

'''
This was an experiment.
Idea was to generate common aliases of functions to mimic case-
insensitivity. It was a terrible idea and should never be used under
any circumstances
'''

def camel2underscore(s):
    '''
    http://stackoverflow.com/questions/1175208/
    elegant-python-function-to-convert-camelcase-to-camel-case
    '''
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def underscore2camel(s):
    '''
    http://stackoverflow.com/questions/4303492/
    how-can-i-simplify-this-conversion-from-underscore-to-camelcase-in-python
    '''
    def camelcase():
        yield str.lower
        while True:
            yield str.capitalize
    c = camelcase()
    return "".join(c.next()(x) if x else '_' for x in s.split("_"))
