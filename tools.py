#!/usr/bin/python
# -*- coding: UTF-8 -*-
# BEGIN PYTHON 2/3 COMPATIBILITY BOILERPLATE
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
from neurotools.system import *

import os
import traceback
import inspect
import numpy as np
import datetime
import time as systime
import sys

from matplotlib.cbook          import flatten
from neurotools.jobs.decorator import *
from scipy.io                  import loadmat
import sys
from numbers import Number
from collections import Set, Mapping, deque
import errno

try: 
    # Python 2
    zero_depth_bases = (basestring, Number, xrange, bytearray)
    iteritems = 'iteritems'
except NameError: 
    # Python 3
    zero_depth_bases = (str, bytes, Number, range, bytearray)
    iteritems = 'items'

# this line must come last if we want memoization
#from neurotools.jobs.cache import *

#TODO: make this robust / cross-platform
matlabpath = '/usr/local/bin/matlab -nodesktop -nodisplay -r'

try:
    from decorator import decorator
    from decorator import decorator as robust_decorator
except Exception as e:
    traceback.print_exc()
    print('Importing decorator failed, nothing will work.')
    print('try easy_install decorator')
    print('or  pip  install decorator')
    print('hopefully at least one of those works')

def varexists(varname):
    '''
    '''
    if varname in vars():
        return vars()[varname]!=None
    if varname in globals():
        return globals()[varname]!=None
    return False

def nowarn():
    '''
    TODO: merge warning control with something more standard
    '''
    global SUPPRESS_WARNINGS
    SUPPRESS_WARNINGS=True

def okwarn():
    '''
    TODO: merge warning control with something more standard
    '''
    global SUPPRESS_WARNINGS
    SUPPRESS_WARNINGS=False

def dowarn(*a,**kw):
    '''
    TODO: merge warning control with something more standard
    '''
    if varexists('SUPPRESS_WARNINGS'):
        return not SUPPRESS_WARNINGS
    else: return True

def warn(*a,**kw):
    '''
    TODO: merge warning control with something more standard
    '''
    if dowarn(*a,**kw): print(' '.join(map(str,a)))

# todo: make these separate
# TODO: is debug even used?
debug = warn

def wait(prompt='--- press enter to continue ---'):
    '''
    '''
    print(prompt)
    raw_input()

def matlab(commands):
    '''
    Runs Matlab commands through the shell
    TODO: make matlabpath configurable
    '''
    commands = commands.replace('\n',' ')
    print(commands)
    os.system("""%s "identifyHost(); %s; exit" """%(matlabpath,commands))
    os.system('reset')

def zeroslike(x):
    '''
    Create numpy array of zeros the same shape and type as x
    '''
    return zeros(x.shape,dtype=x.dtype)

def oneslike(x):
    '''
    Create numpy array of ones the same shape and type as x
    '''
    return ones(x.shape,dtype=x.dtype)

def history(n):
    '''
    Return last n lines of shell history
    '''
    print('\n'.join(In[-n:]))

def p2c(p):
    '''
    Convert a point in terms of a length-2 iterable into a complex number
    '''
    return p[0]+1j*p[1]

def c2p(z):
    ''' 
    Convert complex point to tuple
    '''
    return arr([z.real,z.imag])

class emitter():
    '''
    This is a toy example test of a concept used in the piper dectorator
    below. It extends callables so that using them with the logical or
    operator "|" will apply the callable as a side effect before
    returning the original value of the expression. The default side
    effect is printing the object value.
    '''
    '''
    :Example:
    >>> emit = emitter()
    >>> emit | cos(10)
    
    '''
    def __init__(self,operation=None):
        if operation is None:
            def operation(x):
                print(x)
        self.operation=operation
        self._IS_EMITTER_=True
    def __or__(self,other):
        self.operation(other)
        return other
    def __ror__(self,other):
        self.operation(other)
        return other

class piper():
    '''
    Piper extends callables such that they can be called by using 
    infix operators. 
    
    This is dangerous. 
    Do not use it.

    :Example:
    >>> def foo(x):
    >>>     return x+1
    >>> pip = piper(foo)
    >>> 1 + 1 | pip
    >>> @piper
    >>> def zscore(x):
    >>>     return (x-mean(x,0))/std(x,0)
    >>> zscore < rand(10)
    
    '''
    def __init__(self,operation):
        self.operation=operation
        self._IS_EMITTER_=True
    def __or__(self,other):
        return self.operation(other)
    def __lt__(self,other):
        return self.operation(other)
    def __lshift__(self,other):
        return self.operation(other)
    def __rgt__(self,other):
        return self.operation(other)
    def __ror__(self,other):
        return self.operation(other)
    def __call__(self,other):
        return self.operation(other)
    def __pow__(self,other):
        return self.operation(other)

@robust_decorator
def globalize(function,*args,**krgs):
    '''
    globalize allows the positional argument list of a python to be
    truncated. 
    
    Missing arguments will be filled in by name from the
    globals dictionary. This is actually very handy for the functions
    that reference data by session and area. Arguments can be omitted to
    these functions if the session and area are defined in global scope.

    This is dangerous. Do not use it.
    
    '''
    '''
    :Example:
    >>> @globalize
    >>> def fun(r,something='no'):
    >>>     print r,something
    >>> r=8
    >>> fun()
    
    '''
    warn('GLOBALIZE DECORATOR BROKE WHEN MOVING TO @decorator VERSION')
    argsp = inspect.getargspec(function)
    # we dont handle these right now. How to handle?
    # we can't auto-fill *args because we won't know their names
    # we also can't auto-fill kwargs
    # but we can forward them.
    # so it's just args that we tweak.
    # kwargs will be forwarded
    assert argsp.varargs  is None
    assert argsp.keywords is None
    argkeys  = argsp.args
    ndefault = 0 if argsp.defaults is None else len(argsp.defaults)
    ntotargs = 0 if argsp.args     is None else len(argsp.args)
    npositio = ntotargs-ndefault
    newargs = []
    nargs = len(args)
    argnames = []
    # print argkeys
    for i,a in enumerate(argkeys):
        if i>=npositio:
            if i<nargs:
                newargs.append(args[i])
                argnames.append(a)
                continue
            else: break
        if i<len(args):
            newargs.append(args[i])
            argnames.append(a)
        else:
            if a in globals():
                newargs.append(globals()[a])
            else:
                print('warning missing key, setting to None')
                newargs.append(None)
            argnames.append(a)
    warn('globalized ' +' '.join(map(str,argnames)))
    return function(*newargs,**krgs)


# common aliases
exists = varexists
concat = np.concatenate

matfilecache = {}
def metaloadmat(path):
    '''
    Loads a matfile from the provided path, caching it in the global dict
    "matfilecache" using path as the lookup key.

    Parameters
    ----------
    path : string
        unique absolute path to matfile to be loaded
    '''
    global matfilecache
    if path in matfilecache: return matfilecache[path]
    print('caching',path)
    if dowarn(): print('loading data...',)
    data = loadmat(path)
    matfilecache[path]=data
    if dowarn(): print('loaded')
    return data

def find_all_extension(d,ext='png'):
    '''
    Locate all files with a given extension
    
    Parameters
    ----------
    d : string
        String representing a path in the filesystem
    
    Other Parameters
    ----------------
    ext : string
        extension to locate. search is not case-sensitive. 
        
    Returns
    -------
    found : string
        List of files with matching extension
    '''
    found = []
    for root,dirs,files in os.walk(d):
        found.extend([f for f  in files if f.lower().split('.')[-1]==ext])
    return found

# really should make nice datastructures for all this
setinrange = lambda data,a,b: {k for k,v in data.iteritems() if (v>=a) and (v<=b)}
mapdict    = lambda data,function: {k:function(v) for k,v in data.iteritems()}
getdict    = lambda data,index: mapdict(data, lambda v:v[index])

def ensure_dir(dirname):
    """
    Ensure that a named directory exists; if it does not, attempt to create it.
    http://stackoverflow.com/questions/944536/efficient-way-of-creating-recursive-paths-python
    
    Parameters
    ----------
    
    Returns
    -------
    """
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            pass

def getsize(obj):
    """
    Recursively iterate to sum size of object & members.
    http://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python    
    
    Parameters
    ----------
    
    Returns
    -------
    """
    def inner(obj, _seen_ids = set()):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, zero_depth_bases):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Now assume custom object instances
        elif hasattr(obj, '__slots__'):
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        else:
            attr = getattr(obj, '__dict__', None)
            if attr is not None:
                size += inner(attr)
        return size
    return inner(obj)
    

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
    __saved_context__.update(sys.modules[__name__].__dict__)

clear = saveContext

def restoreContext():
    names = sys.modules[__name__].__dict__.keys()
    for n in names:
        if n not in __saved_context__:
            del sys.modules[__name__].__dict__[n]

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

def current_milli_time():
    '''
    Returns the time in milliseconds
    '''
    return int(round(systime.time() * 1000))

now = current_milli_time

def today():
    '''
    Returns
    -------
    `string` : the date in YYMMDD format
    '''
    return datetime.date.today().strftime('%Y%m%d')

__GLOBAL_TIC_TIME__ = None
def tic(st=''):
    ''' 
    Similar to Matlab tic 
    stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    global __GLOBAL_TIC_TIME__
    t = current_milli_time()
    try:
        __GLOBAL_TIC_TIME__
        if not __GLOBAL_TIC_TIME__ is None:
            print('t=%dms'%((t-__GLOBAL_TIC_TIME__)),st)
        else: print("timing...")
    except: print("timing...")
    __GLOBAL_TIC_TIME__ = current_milli_time()
    return t

def toc(st=''):
    ''' 
    Similar to Matlab toc 
    stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    global __GLOBAL_TIC_TIME__
    t = current_milli_time()
    try:
        __GLOBAL_TIC_TIME__
        if not __GLOBAL_TIC_TIME__ is None:
            print('dt=%dms'%((t-__GLOBAL_TIC_TIME__)),st)
        else:
            print("havn't called tic yet?")
    except: print("havn't called tic yet?")
    return t

def waitfor(t):
    '''
    Wait for t milliseconds
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    while current_milli_time()<t:
        pass
    return current_milli_time()
    
    
    
@memoize
def getVariable(path,var):
    '''
    Reads a variable from a .mat or .hdf5 file
    The read result is cached in ram for later access
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    if '.mat' in path:
        return loadmat(path,variable_names=[var])[var]
    elif '.hdf5' in path:
        with h5py.File(path) as f:
            return f[var].value
    raise ValueError('Path is neither .mat nor .hdf5')
