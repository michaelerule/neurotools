#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
`neurotools.util.string`:
Helper functions for string formatting
"""

import numpy as np


def hcat(*args,sep=' ',TABWIDTH=4,prefix='',suffix=''):
    '''
    Horizontally concatenate two string objects that contain newlines
    '''
    S = [str(s).replace('\t',' '*TABWIDTH).split('\n') for s in args]
    L = [np.max(list(map(len,s))) for s in S]
    S = [[ln.ljust(l) for ln in s] for (s,l) in zip(S,L)]
    h = np.max(list(map(len,S)))
    S = [s+list(('',)*(len(s)-h)) for s in S]
    return prefix+(suffix+'\n'+prefix).join([sep.join(z) for z in zip(*S)])+suffix


def wordwrap(text,width=80,sep=' '):
    '''
    Wrap text to a fixed columnd width.
    
    TODO: the implementation of this is a bit weird.
    
    Parameters
    ----------
    words : list
        a list of words
    width : positive int
        Width of column to wrap
        Optional, default is 80
    sep : string
        Optional, default is ' '
        Separator to join words by
    
    Returns
    -------
    lines : list
        List of word-wrapped lines of text
    '''
    lines = []
    line = ''
    words = text.split(sep)
    for t in words:
        if len(line)+len(t)>width:
            lines+=[line[:-len(sep)]]
            line=''
        line += t+sep
    return '\n'.join(lines+[line[:-len(sep)]])

def incolumns(*args,prefix='',suffix='',sep=' ',width=80):
    '''
    Collate collection of strings into parallel columns
    '''
    words  = []
    for a in args:
        words.extend(list(a))
    maxlen = np.max([len(w) for w in words])
    ncols  = (width-len(prefix)-len(suffix)+len(sep))//(maxlen+len(sep))
    lists  = [[] for i in range(ncols)]
    nrows  = int(np.ceil(len(words)/ncols))
    while nrows*ncols>len(words):
        words.append('')
    for i,w in enumerate(words):
        lists[i%ncols].append(w+' '*(maxlen-len(w)))
    result = []
    for group in zip(*lists):
        result.append(prefix+sep.join(group)+suffix)
    return '\n'.join(result)


def percent(n,total):
    '''
    Given `n` observations out of `total`, format
    `n/total` as a percentage to two decimal points
    
    Parameters
    ----------
    n: int
        Number of positive samples
    total: int
        Size of population
    
    Returns
    -------
    str
    '''
    return '%0.2g%%'%(n*100.0/total)
    
    
def shortscientific(p,prec=0,latex=False,sigthr=None):
    '''
    Shortest viable string formatting for scientific
    notation.
    
    Parameters
    ----------
    x: float
    
    Other Parameters
    ----------------
    prec: non-negative integer; default 0
        Extra decimals to add.
        Value of `0` uses only one significant figure. 
    latex: boolean; defaultFalse
        Whether to prepare string for latex rendering
    
    Returns
    -------
    str
    '''
    x = ('%.*e'%(prec,p)).replace('-0','-')
    x = x.replace('+','').replace('e0','e')
    if latex:
        x = '$'+x.replace('e','×10^{')+'}'
        if not sigthr is None and p<sigthr:
            x+='{}^*'
        x+='$'
    else:
        if not sigthr is None and p<sigthr:
            x+='*'    
    return x


def eformat(f, prec, exp_digits):
    '''
    Format a float in scientific notation with fewer 
    characters.
    
    Parameters
    ----------
    f: scalar
        Number
    prec:
        Precision
    exp_digits: int
        Number exponent digits
    
    Returns
    -------
    str:
        Formatted string
    '''
    if not np.isfinite(f):
        return '%e'%f
    s = "%.*e"%(prec, f)
    mantissa, exponent = s.split('e')
    exponent = int(exponent)
    s = mantissa + "e%+0*d"%(exp_digits+1, exponent)
    s = s.replace('+','')
    return s


def v2str(p,sep=','):
    '''
    Format list of numbers as string in short
    scientific notation.
    (see `shortscientific()`)
    
    Parameters
    ----------
    p: iterable of floats
        Numbers to format
        
    Other Parameters
    ----------------
    sep: str
        Item separtor
        
    Returns
    ------
    str
    '''
    return '['+sep.join([shortscientific(x) for x in p])+']'


def v2str_long(p,sep=','):
    '''
    Format list as string with maximum precision.
    
    Parameters
    ----------
    p: iterable of floats
        Numbers to format
        
    Other Parameters
    ----------------
    sep: str
        Item separtor
        
    Returns
    ------
    str
    '''
    return '['+sep.join([
        np.longdouble(x).astype(str) for x in p])+']'


def nicetable(data,format='%4.4f',ncols=8,prefix='',sep=' '):
    '''
    Format a numeric vector as an evenly-spaced table
    '''
    N = len(data)
    nrows = int(np.ceil(N/ncols))
    lines = []
    for r in range(nrows):
        d = data[r*ncols:(r+1)*ncols]
        formatted = [format%i for i in d]
        joined    = sep.join(formatted)
        line      = prefix+joined
        lines    += [line]
    return '\n'.join(lines)
    
    
def isInt(v):
    '''
    Check if a string is an integer
    
    stackoverflow.com/questions/1265665/python-check-if-a-string-represents-an-int-without-using-try-except
    
    Parameters
    ----------
    v: str
    
    Returns
    -------
    boolean
    '''
    v = str(v).strip()
    return v=='0' or (v if v.find('..') > -1 else v.lstrip('-+').rstrip('0').rstrip('.')).isdigit()


def camel_to_snake(s):
    '''
    Convert a ``camelCase`` string to a lower-case
    string with words delimited by underscore ``_``
    ("snake case").
    '''
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def snake_to_camel(s,delimeters=' _'):
    '''
    Convert a string delimited by spaces `` `` or 
    underscores ``_`` to camel case. 
    '''
    for d in delimeters:
        s = s.replace(d,' ')
    s = s.title()
    if len(s)<=0: return ''
    if len(s)<=1: return s[0].lower()
    return s[0].lower() + s[1:].replace(' ','')
    
    

    
    
    

