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

'''
Text and string manipulation routines

'''

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
