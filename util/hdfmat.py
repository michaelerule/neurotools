#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Routines for handling the new-style `.mat` files, which are secretly `.hdf` files
"""

from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy
import h5py

def getHDFvalue(hdf,d,squeeze=True,detectString=True,detectScalar=True):
    '''
    Unpack a value from a HDF5 file
    
    Parameters
    ----------
    hdf: hdf file or subdirectory object
    path: string
        Subpath to extract within hdf file or subdirectory object
    '''
    try:
        value = d if type(d) is numpy.ndarray else d[()]#.value
    except AttributeError:
        # It's probably a file node
        return d

    # Detect things that are probably strings
    if detectString and len(value.shape)==2 \
        and value.shape[1]==1 \
        and value.shape[0]>0 \
        and value.dtype==numpy.uint16:
        # might be a string
        stringval = ''.join(list(map(chr,value.ravel())))
        return stringval
        
    # Detect and unpack scalars
    if detectScalar and all(s==1 for s in value.shape) or value.shape==():
        return value.ravel()[0]

    if squeeze:
        value = numpy.squeeze(value)
        
    # Try to handle object references if possible
    # This MIGHT create an infinite loop if there are circular references
    # .. so be careful? 
    if value.dtype==numpy.dtype('O'):
    
        # Normal HDF5 references
        if type(value.ravel()[0])==h5py.h5r.Reference:
            # Probably an object reference
            npvalue = numpy.array([
                getHDFvalue(hdf,hdf[r],squeeze,detectString,detectScalar)
                for r in value.ravel()])
        '''
        print('hi',type(value.ravel()[0]))
        # Weird-ass Matlab references?
        if type(value.ravel()[0])==h5py.h5r.Reference:#h5py._hl.dataset.Dataset:
            print('hi')
            npvalue = numpy.array([r.value for r in value.ravel()])
        ''' 
        value = numpy.reshape(npvalue,value.shape+npvalue.shape[1:])
            
    if squeeze:
        value = numpy.squeeze(value)
    
    return value
    
def getHDF(hdf,path,sep=None,squeeze=True,detectString=True,detectScalar=True):
    '''
    Retrieve path from nested dictionary obtained from an HDF5 file.
    Path separator is `/`.
    
    Parameters
    ----------
    hdf: hdf file or subdirectory object
    path: string
        Subpath to extract within hdf file or subdirectory object
    
    Returns
    -------
    extracted hdf file or subdirectory object
    '''
    if not 'items' in dir(hdf):
        raise ValueError('data object is not dictionary-like')
        
    if sep==None:
        separators = "./\\"
        inpath = [s for s in separators if s in path] 
        if len(inpath)>1:
            raise ValueError(\
                'Path seems to contain multiple separators, %s?'\
                %(' '.join(['"%s"'%s for s in inpath])))
        sep = inpath[0] if len(inpath) else '.'
        
    nodes = path.split(sep)
    d = hdf
    for node in nodes:
        if not 'items' in dir(d):
            raise ValueError('data does not contain path '+path)
        d = d[node]
        
    return getHDFvalue(hdf,d,squeeze,detectString,detectScalar)

def hdf2dict(d):
    '''
    Recursively convert HFDF5 Matlab outbut into a nested dict
    '''
    if type(d) is numpy.ndarray:
        return d
    if 'value' in dir(d):
        return d.value#d[()]#.value
    # directory node: recursively convert
    # (Skip the #refs# variable if we encounter it)
    return {k:hdf2dict(v) for (k,v) in d.items() if k!=u'#refs#'}

def printmatHDF5(d):
    '''
    formatted printing for .mat style dicts
    '''
    recursive_printmatHDF5(d)

def recursive_printmatHDF5(d,prefix=' '):
    variables = {}
    dict_vars = {}
    for k,v in d.items():
        if k == u'#refs#':
            continue
            # matlab creates a #refs# variable, what is it?
        if type(v) is numpy.ndarray:
            variables[k] = v
        else:
            try:
                variables[k]=v[()]#.value
            except AttributeError:
                try:
                    dict_vars[k] = dict(v)
                except ValueError as e:
                    traceback.print_exc()
    if len(variables)<=0:
        if len(dict_vars)<=0:
            print(prefix+"(empty)")
    else:
        keys = sorted(variables.keys())
        # format the array dimension, data type for all variables
        content=[keys,[],[],[]]
        for k in keys:
            v = variables[k]
            content[1].append(' x '.join(map(str,v.shape)))
            content[2].append(v.dtype.name)
            content[3].append(str(type(v))[7:-2])  
            if v.shape==(1,1):
                # Scalar variables are special cases!
                # We can show their value.
                content[3][-1] = str(v[0][0])
        # Pad all columns to the same length
        X=[[y+''.join([' ']*(Z-len(y))) for y in Y] 
            for Y,Z in zip(content,[max(map(len,Y)) 
            for Y in content])]
        print( prefix+('\n'+prefix).join([u' | '.join(('',)+x+('',)) for x in zip(*X)]))
    for k,v in dict_vars.items():
        print(prefix+k+':')
        recursive_printmatHDF5(v,prefix+': ')
