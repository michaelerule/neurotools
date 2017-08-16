#!/usr/bin/python
# -*- coding: UTF-8 -*-
# The above two lines should appear in all python source files!
# It is good practice to include the lines below
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

'''
Robust decorators are provided by the decorator package
    http://pythonhosted.org/decorator/documentation.html
'''

from neurotools.ntime import current_milli_time
import os, sys
from collections import defaultdict

import inspect, ast, types
import warnings, traceback, errno
import pickle, json, base64, zlib

import numpy as np

import sys
__PYTHON_2__ = sys.version_info<(3, 0)


try:
    import decorator
    from decorator import decorator as robust_decorator
except:
    print('Error, cannot find decorator module')
try:
    import typedecorator
    from typedecorator import params, returns, setup_typecheck
except:
    print('Error, cannot find typedecorator module')

def listit(t):
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t

def tupleit(t):
    return tuple(map(tupleit, t)) if isinstance(t, (list, tuple)) else t

def sanitize(sig,mode='liberal'):
    '''
    Converts an argument signature into a standard format. Lists will
    be converted to tuples. Non-hashable types will cause an error.

    "strict" mode requires that all data be numeric primitives or
    strings containing "very safe" chracters a-zA-Z0-9 and space
    '''
    if isinstance(sig,np.ndarray):
        if sig.ndim==0:
            sig = sig[()]
        return sanitize(sig)
    if isinstance(sig,np.core.memmap):
        sig = np.array(sig)
        return sanitize(sig)
    if isinstance(sig, (list,tuple)):
        if len(sig)<=0: return ()
        while len(sig)==1: sig=sig[0]
        return tuple(sanitize(s,mode=mode) for s in sig)
    if __PYTHON_2__ and isinstance(sig,(unicode,)):
        return sanitize(str(sig),mode=mode)
    if isinstance(sig, (dict,)):
        keys = sorted(sig.keys())
        vals = (sig[k] for k in keys)
        return sanitize(zip(keys,vals),mode=mode)
    if isinstance(sig, (set,)):
        return tuple(sanitize(s,mode=mode) for s in sorted(list(sig)))
    if hasattr(sig, '__iter__'):
        try:
            return tuple(sanitize(s,mode=mode) for s in list(sig))
        except:
            # Probably a 0 dimensional iterable, not sure how to handle this
            pass

    if mode=='strict':
        if type(sig) in (int,float,bool,None):
            return sig
        if type(sig) is str:
            if any([c not in 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890 ' for c in sig]):
                raise ValueError('Strict mode requires all strings consist of letters, numbers, and spaces')
            return sig
    return sig

def summarize_function(f):
    '''Prints function information, Used for debugging decorators.'''
    print(f)
    for k in dir(f):
        print('\t',k,'\t',end='')
        if not k in ['__globals__','func_globals','func_dict','__dict__']:
            print(f.__getattribute__(k))
        else: print()
    code = f.func_code
    print('\tCODE')
    for k in dir(code):
        if not k[:2]=='__': print('\t\t',k,'\t',code.__getattribute__(k))
    named,vargnames,kwargnames,defaults = inspect.getargspec(f)
    print('\t',named,vargnames,kwargnames,defaults)

def argument_signature(function,*args,**kwargs):
    '''
    Convert the function arguments and values to a unique set.
    Throws ValueError if the provided arguments cannot match argspec.
    '''
    named_store = {} # map from parameter names to values
    named,vargname,kwargname,defaults = inspect.getargspec(function)

    # Pattern matching can give rise to lists in the "named" variable
    # returned here. We need to convert these to something hashable.
    named = sanitize(named)

    available = list(zip(named,args))
    nargs     = len(available)
    ndefault  = len(defaults)   if not defaults is None else 0
    nnamed    = len(named)      if not named    is None else 0
    # All positional arguments must be filled
    nmandatory = nnamed - ndefault
    if nargs<nmandatory: raise ValueError('Not enough positional arguments')
    # Assign available positional arguments to names
    for k,v in available:
        if k in named_store: raise ValueError('Duplicate argument',k)
        named_store[k] = v
    # If not all arguments are provided, check **kwargs and defaults
    ndefaulted   = max(0,nnamed - nargs)
    default_map = dict(zip(named[-ndefault:],defaults)) if ndefault>0 else {}
    if ndefaulted>0:
        for k in named[-ndefaulted:]:
            if k in named_store: raise ValueError('Duplicate argument',k)
            named_store[k] = kwargs[k] if k in kwargs else default_map[k]
            if k in kwargs: del kwargs[k]
    # Store excess positional arguments in *vargs if possible
    vargs = None
    if len(args)>nnamed:
        if vargname is None:
            raise ValueError('Excess positional arguments, but function does not accept *vargs!')
        vargs = args[nnamed:]
    # Store excess keyword arguments if the function accepts **kwargs
    if len(kwargs):
        if kwargname is None:
            raise ValueError("Excess keyword arguments, but function does not accept **kwargs!")
        for k in kwargs:
            if k in named_store: raise ValueError('Duplicate argument',k)
            named_store[k] = kwargs[k]
    # Construct a tuple reflecting argument signature
    keys  = sorted(named_store.keys())
    vals  = tuple(named_store[k] for k in keys)
    return sanitize((tuple(zip(keys,vals)),vargs))



def print_signature(sig):
    '''Formats the argument signature for printing.'''
    named, vargs = sig
    result = ','.join(['%s=%s'%(k,v) for (k,v) in named])
    if not vargs is None: result += ','+','.join(map(str,vargs))
    return result

@robust_decorator
def timed(f,*args,**kwargs):
    '''
    Timer decorator, modifies return type to include runtime
    '''
    t0      = current_milli_time()
    result  = f(*args,**kwargs)
    t1      = current_milli_time()
    return float(t1-t0), result

def memoize(f):
    '''
    Memoization decorator
    '''
    cache = {}
    info  = defaultdict(dict)
    @robust_decorator
    def wrapped(f,*args,**kwargs):
        sig = argument_signature(f,*args,**kwargs)
        if not sig in cache:
            time,result = f(*args,**kwargs)
            info[sig]['density'] = time/sys.getsizeof(result)
            cache[sig] = result
        return cache[sig]
    wrapped.__cache__ = cache
    wrapped.__info__  = info
    return wrapped(timed(f))

def unwrap(f):
    '''
    Strips decorators from a decorated function, provided that the
    decorators were so kind as to set the .__wrapped__ attribute
    '''
    g = f
    while hasattr(g, '__wrapped__'):
        g = g.__wrapped__
    return g

if __name__=="__main__":

    print("Testing the memoize decorator")

    def example_function(a,b,c=1,d=('ok',),*vargs,**kw):
        ''' This docstring should be preserved by the decorator '''
        e,f = vargs if (len(vargs)==2) else (None,None)
        g = kw['k'] if 'k' in kw else None
        print(a,b,c,d,e,f,g)

    f = example_function
    g = memoize(example_function)

    print('Testing example_function')
    fn = f
    fn('a','b','c','d')
    fn('a','b','c','d','e','f')
    fn('a','b',c='c',d='d')
    fn('a','b',**{'c':'c','d':'d'})
    fn('a','b',*['c','d'])
    fn('a','b',d='d',*['c'])
    fn('a','b',*['c'],**{'d':'d'})
    fn('a','b','c','d','e','f')

    print('Testing memoized example_function')
    fn = g
    fn('a','b','c','d')
    fn('a','b','c','d','e','f')
    fn('a','b',c='c',d='d')
    fn('a','b',**{'c':'c','d':'d'})
    fn('a','b',*['c','d'])
    fn('a','b',d='d',*['c'])
    fn('a','b',*['c'],**{'d':'d'})
    fn('a','b','c','d','e','f')
