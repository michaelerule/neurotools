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

def generateAlternativeNames(name):
    alternatives = set()
    # upper and lower case variants
    alternatives.add(name.lower())
    alternatives.add(name.upper())
    if '_' in name:
        # variants without underscores
        cname = underscore2camel(name)
        alternatives.add(cname)
        alternatives.add(cname.lower())
        alternatives.add(cname.upper())
    if name.lower()!=name:
        # test for capital letters
        # this might be camel case
        uname = camel2underscore(name)
        alternatives.add(uname)
        alternatives.add(uname.lower())
        alternatives.add(uname.upper())
    if 'get'==name.lower()[:3]:
        if len(name)>4 and name[3]=='_':
            alternatives |= set(generateAlternativeNames(name[4:]))
        elif len(name)>3:
            alternatives |= set(generateAlternativeNames(name[3:]))
    return sorted(list(set(alternatives)))

def camelKludge(silent=False):
    if not silent: warn('THIS WILL SMASH THE GLOBALS')
    import types
    to_add = {}
    tocheck = globals().items()
    for name,value in tocheck:
        if type(value) is not types.FunctionType: continue
        if '__IS_AN_ALIAS__' in value.__dict__: continue
        # generate alternatives
        alternatives = generateAlternativeNames(name)
        canAlias = True
        for alt in alternatives:
            if alt in globals() and not globals()[alt] is value:
                if not silent: warn('Aliasing would create conflict. %s %s'%(name,alt))
                canAlias = False
                break
        if canAlias:
            def bindAlias(s,f):
                def aliasf(*a,**k):
                    warn('NO: this is an alias. Please call %s instead!'%s)
                    return f(*a,**k)
                aliasf.__IS_AN_ALIAS__ = True
                return aliasf
            aliasf = bindAlias(name,value)
            for alt in alternatives:
                if not silent: print(name,alt)
                if alt in globals():
                    pass
                else:
                    to_add[alt]=aliasf
    globals().update(to_add)

camelKludge(True)
