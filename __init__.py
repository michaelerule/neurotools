#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

#import neurotools
#from   neurotools import *

'''
https://stackoverflow.com/a/67565993/900749

This should make all submodules visible and importable. 

'''

import importlib
import pkgutil

def import_submodules(package, recursive=True):
""" Import all submodules of a module, recursively, including subpackages

:param package: package (name or actual module)
:type package: str | module
:rtype: dict[str, types.ModuleType]
"""
if isinstance(package, str):
    package = importlib.import_module(package)
results = {}
for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
    full_name = package.__name__ + '.' + name
    results[full_name] = importlib.import_module(full_name)
    if recursive and is_pkg:
        results.update(import_submodules(full_name))
return results

