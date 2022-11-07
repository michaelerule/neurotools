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

'''
Helper functions associated with time
'''

import numpy as np
import datetime
import time as systime

from . import tools

def current_milli_time():
    '''
    Returns the time in milliseconds
    '''
    return int(round(systime.time() * 1000))

def today():
    '''
    Returns
    -------
    `string` : the date in YYMMDD format
    '''
    return datetime.date.today().strftime('%Y%m%d')

def now():
    '''
    Current date and time as a %Y%m%d_%H%M%S formatted string
    '''
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

__GLOBAL_TIC_TIME__ = None
def tic(doprint=True,prefix=''):
    ''' 
    Similar to Matlab tic 
    stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    
    Parameters
    ----------
    doprint : bool
        if True, print elapsed time. Else, return it.
    
    Returns
    -------
    '''
    global __GLOBAL_TIC_TIME__
    t = current_milli_time()
    try:
        __GLOBAL_TIC_TIME__
        if not __GLOBAL_TIC_TIME__ is None:
            if doprint:
                print(prefix,'t=%dms'%(t-__GLOBAL_TIC_TIME__))
        elif doprint:
            print("timing...")
    except: 
        if doprint: print("timing...")
    __GLOBAL_TIC_TIME__ = current_milli_time()
    return t

def toc(doprint=True,prefix=''):
    ''' 
    Similar to Matlab toc 
    stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    
    Parameters
    ----------
    doprint : bool
        if True, print elapsed time. Else, return it.
    
    Returns
    -------
    t : number
        Current timestamp
    dt : number
        Time since the last call to the tic() or toc() function.
    '''
    global __GLOBAL_TIC_TIME__
    t = current_milli_time()
    try:
        __GLOBAL_TIC_TIME__
        if not __GLOBAL_TIC_TIME__ is None:
            dt = t-__GLOBAL_TIC_TIME__
            if doprint: print(prefix,'dt=%dms'%(dt))
            return t,dt
        elif doprint:
            print("havn't called tic yet?")
    except: 
        if doprint: print("havn't called tic yet?")
    return t,None

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



@tools.piper
def progress_bar(x,N=None):
    '''
    Wraps an iterable to print progress through the loop
    to the terminal.
    
    Parameters
    ----------
    x: iterable
    N: int
        Expected length of iterable (if generator)
    '''
    if N is None:
        x = list(x)
        N = len(x)
    K = int(np.floor(np.log10(N)))+1
    pattern = ' %%%dd/%d'%(K,N)
    wait_til_ms = systime.time()*1000
    for i,x in enumerate(x):
        time_ms = systime.time()*1000
        if time_ms>=wait_til_ms:
            r = i*50/N
            k = int(r)
            q = ' ▏▎▍▌▋▊▉'[int((r-k)*8)]
            print(
                '\r['+
                ('█'*k)+    
                q+
                (' '*(50-k-1))+
                ']%3d%%'%(i*100//N)+
                (pattern%i),
                end='',
                flush=True)
            wait_til_ms = time_ms+1000
        yield x
    print('\r'+' '*70+'\r',end='',flush=True)

pbar = progress_bar
pb   = progress_bar
en   = tools.piper(enumerate)






























