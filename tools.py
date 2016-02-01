import os
import traceback
from itertools import *
from collections import *
import inspect
import numpy as np
from numpy import *
from matplotlib.cbook import flatten


#TODO: make this robust / cross-platform
matlabpath = '/usr/local/bin/matlab -nodesktop -nodisplay -r'



try:
    from decorator import decorator
except Exception, e:
    traceback.print_exc()
    print 'Importing decorator failed, nothing will work.'
    print 'try easy_install decorator'
    print 'or  pip  install decorator'
    print 'hopefully at least one of those works'


print 'WARNING MOST FUNCTIONS HAVE MOVED TO SIGNALTOOLS'

'''
def memoize(f):
    """ Memoization decorator for a function taking one or more arguments. """
    class memodict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)
        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret
    return memodict().__getitem__
'''

# http://stackoverflow.com/questions/6407993/how-to-memoize-kwargs
from functools import wraps
def memoize(fun,*args, **kwargs):
    """A simple memoize decorator for functions supporting positional args."""
    if not fun.func_defaults is None:
        print 'ERROR NO LONGER ACCEPTING KWARGS FOR MEMOIZATION'
        print 'UNIQUE KEY,VALUE ASSOCIATIONS HARD TO DETERMINE SINCE'
        print 'KWARGS MAY BE PASSED IN ANY ORDER, AND MAY DEFINE DEFAULTS'
        print 'THIS CAN BE IMPLMENTED BUT IT WILL TAKE TIME TO CORRECTLY'
        return fun    
    @wraps(fun)
    def wrapper(*args, **kwargs):
        key = (args, frozenset(sorted(kwargs.items())))
        if key in cache:
            #print 'located memo for',fun.__name__,':',key
            return cache[key]
        else:
            #print 'could not find memo for',fun.__name__,':',key
            ret = cache[key] = fun(*args, **kwargs)
        return ret
    wrapper.__name__ = fun.__name__
    wrapper.__doc__ = fun.__doc__
    cache = {}
    return wrapper

#memoize = lambda x:x

def varexists(varname):
    if varname in vars():
        return vars()[varname]!=None
    if varname in globals():
        return globals()[varname]!=None
    return False

def nowarn():
    global SUPPRESS_WARNINGS
    SUPPRESS_WARNINGS=True

def okwarn():
    global SUPPRESS_WARNINGS
    SUPPRESS_WARNINGS=False

def dowarn(*a,**kw):
    if varexists('SUPPRESS_WARNINGS'):
        return not SUPPRESS_WARNINGS
    else: return True
    
def warn(*a,**kw):
    if dowarn(*a,**kw): print ' '.join(map(str,a))

# todo: make these separate
debug = warn

def wait(prompt='--- press enter to continue ---'):
    print prompt
    raw_input()

def matlab(commands):
    commands = commands.replace('\n',' ')
    print commands
    os.system("""%s "identifyHost(); %s; exit" """%(matlabpath,commands))
    os.system('reset')

def zeroslike(x):
    return zeros(shape(x),dtype=x.dtype)

def oneslike(x):
    return ones(shape(x),dtype=x.dtype)

def ensuredir(d):
    os.system('mkdir %s'%d)

def history(n):
    print '\n'.join(In[-n:])
    
def p2c(p):
    return p[0]+1j*p[1]

def c2p(z):
    ''' converts complex point to tuple'''
    return arr([real(z),imag(z)])

class emitter():
    '''
    This is a toy example test of a concept used in the piper dectorator
    below. It extends callables so that using them with the logical or 
    operator "|" will apply the callable as a side effect before
    returning the original value of the expression. The default side 
    effect is printing the object value.
    emit = emitter()
    emit | cos(10)
    '''
    def __init__(self,operation=None):
        if operation is None:
            def operation(x):
                print x         
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
    Piper does some unusual things to the python language. It extends 
    callables such that they can be called by using various infix operators.
    this is slightly dangerous, due to the complexity of operator precedence
    and the nuances of whether the operator definition of the left or right
    operand is used.
       
    #piper test
    def foo(x):
        return x+1
    pip = piper(foo)
    1 + 1 | pip
    @piper
    def zscore(x):
        return (x-mean(x,0))/std(x,0)
    zscore < rand(10)
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

@decorator
def globalize(function,*args,**krgs):
    '''
    globalize allows the positional argument list of a python to be
    truncated. missing arguments will be filled in by name from the 
    globals dictionary. This is actually very handy for the functions
    that reference data by session and area. Arguments can be omitted to 
    these functions if the session and area are defined in global scope.
    
    # globalizer test
    @globalize
    def fun(r,something='no'):
        print r,something
    r=8
    fun()
    '''
    warn('GLOBALIZE DECORATOR BROKE WHEN MOVING TO @decorator VERSION')
    argsp = inspect.getargspec(function)
    # we dont handle these right now. How to handle?
    # we can't auto-fill *args because we won't know their names
    # we also can't auto-fill kwargs
    # but we can forward them?
    # we won't know how.
    # so basically it's just args that we tweak. 
    # kwargs will be forwarded 
    # print argsp
    assert argsp.varargs   is None
    assert argsp.keywords is None
    argkeys  = argsp.args
    ndefault = 0 if argsp.defaults is None else len(argsp.defaults)
    ntotargs = 0 if argsp.args     is None else len(argsp.args)
    npositio = ntotargs-ndefault
    newargs = []
    nargs = len(args)
    argnames = []
    # print argkeys
    for i,a in en|argkeys:
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
                print 'warning missing key, setting to None'
                newargs.append(None)
            argnames.append(a)
    warn('globalized ' +' '.join(map(str,argnames)))
    return function(*newargs,**krgs)


def synonymize(fun):
    '''
    decorator to register likely name variant of a function with globals
    not implemented.
    will not shadow other globals
    '''
    assert 0


# common aliases
exists = varexists
ce = exists
enum = enumerate
enm = enum
en = enm
arr = np.array
nmx = np.max
npx = np.max
nmn = np.min
npn = np.min
npm = np.min
zro = np.zeros
zr = zro
ons = np.ones
on = ons
xr = xrange
iz = izip
concat = np.concatenate
conct = concat
ccat = concat
cct = concat
ct = cct



@piper
def en(x):
    if type(x) is dict: x=x.iteritems()
    return enumerate(x)

zc = piper(lambda x: (x-mean(x,0))/std(x,0))

@piper
def ar(x):
    return array(x)


def enlist(iterable):
    print '\n'.join(map(str,iterable))


def scat(*args,**kwargs):
    if 's' in kwargs:
        return kwargs['s'].join(map(str,args))
    return ' '.join(map(str,args))


from scipy.io import loadmat
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
    print 'caching',path
    if dowarn(): print 'loading data...',
    data = loadmat(path)
    matfilecache[path]=data
    if dowarn(): print 'loaded'
    return data


def shortscientific(x):
    return ('%0.0e'%x).replace('-0','-')
    





