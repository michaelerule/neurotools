#!/usr/bin/python
# -*- coding: UTF-8 -*-
# The above two lines should appear in all python source files!
# It is good practice to include the lines below
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import print_function

import datetime
import time as systime

def current_milli_time():
    '''
    Returns the time in milliseconds
    '''
    return int(round(systime.time() * 1000))

now = current_milli_time

def today():
    '''
    Returns the date in YYMMDD format
    '''
    return datetime.date.today().strftime('%y%m%d')

#crude versions of tic and toc from Matlab
#stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python

__GLOBAL_TIC_TIME__ = None

def tic(st=''):
    '''
    Similar to Matlab tic
    '''
    global __GLOBAL_TIC_TIME__
    t = current_milli_time()
    try:
        __GLOBAL_TIC_TIME__
        if not __GLOBAL_TIC_TIME__ is None:
            print('t=%dms'%((t-__GLOBAL_TIC_TIME__)),st)
        else:
            print("timing...")
    except:
        print("timing...")
    __GLOBAL_TIC_TIME__ = current_milli_time()
    return t

def toc(st=''):
    '''
    Similar to Matlab toc
    '''
    global __GLOBAL_TIC_TIME__
    t = current_milli_time()
    try:
        __GLOBAL_TIC_TIME__
        if not __GLOBAL_TIC_TIME__ is None:
            print('dt=%dms'%((t-__GLOBAL_TIC_TIME__)),st)
        else:
            print("you didn't call tic")
    except:
        print("you didn't call tic")
    return t

def waitfor(t):
    '''
    Wait for t milliseconds
    '''
    while current_milli_time()<t:
        pass
    return current_milli_time()
