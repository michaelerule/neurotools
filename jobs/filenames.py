#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Functions to validate filenames
'''

import sys
from   collections import defaultdict
import numpy as np
import scipy.io
import inspect
import ast
import types
import os
import time
import sys
import subprocess
import warnings
import traceback
import errno
import pickle
import json
import base64
import zlib

def validate_argument_signature(sig):
    '''
    Determines whether a given argument signature can be used to cache
    things to disk. The argument signature must be hashable. It must
    consists of types that can be translated between python, numpy, and
    matlab convenctions.
    
    Parameters
    ----------
    sig : argument signature
        Not implemented
    '''
    raise NotImplementedError('Function not yet implemented');

def is_dangerous_filename(filename):
    '''
    Checks whether a `filename` is safe to use on most modern filesystems.
    Filnames must be shorter than 255 characters, and contain no 
    special characters or escape sequences. Filenames should be ASCII
    
    Parameters
    ----------
    filename : string
        String representing a filename. Should be in ASCII and not unicode
        format
    
    Returns
    -------
    bool : 
        False if the filename is broadly compatible with most modern 
        filesystems.
    '''
    if len(filename)>255:
        return True
    if any([c in filename for c in "\\/<>:\"'|?*,@#{}'&`!%$\n\t\b\r "]):
        return True
    return False

def check_filename(filename):
    '''
    Check if a `filename` is safe to use on most filesystems. More lenient
    than `is_dangerous_filename`. Unicode filenames are permitted. Long 
    filenames (over 255 chars) are ok on many modern filesystems and only 
    trigger a warning. Only special characters that outright break windows 
    will raise an error.
    
    Parameters
    ----------
    filename : string
        String representing a filename.
    '''
    if len(filename)>255:
        warnings.warn('FILE NAME MAY BE TOO LONG ON SOME SYSTEMS')
    if any([c in filename for c in "/?<>\\:*|\"\n\t\b\r"]):
        raise ValueError('Filename contains character forbidden on MS Windows')
    if any([c in filename for c in "\\/<>:\"'|?*,@#{}'&`!%$\n\t\b\r "]):
        warnings.warn('FILE NAME CONTAINS CHARACTER THAT MAY CAUSE ISSUES IN SOME SOFTWARE')

