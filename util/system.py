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
Collection of misc utilities
'''

import sys
__IS_PY3__ = sys.version_info>=(3,0)
__PYTHON_2__ = sys.version_info<(3, 0)


if __IS_PY3__:
    # Add some functionality to python 3
    # 
    def execfile(filepath, globals=None, locals=None):
        '''
        http://stackoverflow.com/questions/
        436198/what-is-an-alternative-to-execfile-in-python-3
        
        Note: this doesn't seem to work with ipython notebooks
        '''
        if globals is None:
            globals = {}
        globals.update({
            "__file__": filepath,
            "__name__": "__main__",
        })
        import os
        with open(filepath, 'rb') as file:
            exec(compile(file.read(), filepath, 'exec'), globals, locals)

    # create raw_input alias
    raw_input = input
    
else:
    # Make functions that return lists in python 2
    # return iterables, like in python 3
    # this will break python2 code, but force us to update to code-base
    # to be python3 compatible
    from itertools import imap as map
    range = xrange
    

def query_yes_no(question, default="yes"):
    """
    Ask a yes/no question via raw_input() and return their answer.
    https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input

    Parameters
    ----------
    question : string
        is a string that is presented to the user.
    default : string
        is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    Returns
    -------
    string:
        The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)
    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
