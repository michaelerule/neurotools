#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import os,sys
import numpy  as np
import pandas as pd
import scipy
from neurotools.stats.pvalues import nancorrect
from neurotools.util.tools import piper

#[This page](https://jakevdp.github.io/PythonDataScienceHandbook/02.09-structured-data-numpy.html) is educational.

def get(df,col,val,drop=True):
    '''
    Get all rows with column matching value
    (discards the column matched)
    
    Parameters
    ----------
    df: pandas.DataFrame
        
    Returns
    -------
    :pd.DataFrame
    '''
    df = df.sort_index()
    df = df.loc[df[col].values==val]
    if drop:
        df = df.sort_index()
        df = df.drop(col,axis=1)
    return df

def is_empty_col(c):
    '''
    Check if a value is ``NaN``
    
    Parameters
    ----------
    c: object
        tuple of ``str`` or ``nan`` representing multi-
        level column name. 
        
    Returns
    -------
    is_empty:boolean
    '''
    try:
        return np.isnan(c) 
    except TypeError:
        return False

def from_hierarchical_columns(df,delimeter='__',deep=False):
    '''
    Flatten hierarchical columns into strings.
    Inverse of ``to_hierarchical_columns``.
    
    Parameters
    ----------
    df: pandas.DataFrame
    delimeter: str, default '__'
        String delimiter for hierarchical column levels.
    deep: boolean, default False
        Whether retured array is a deep copy.
        
    Returns
    -------
    :pd.DataFrame
    '''
    df = df.copy(deep=deep)
    if all([not isinstance(c,tuple) for c in df.columns]):
        # Not hierarchical
        return df
    newcols = [delimeter.join([str(ci) for ci in c if not is_empty_col(ci)]) for c in df.columns]
    df.columns = newcols
    return df

@piper
def to_hierarchical_columns(df,delimeter='__',deep=False):
    '''
    Convert a flat set of columns given by strings,
    with levels in the hierarchy delimeted by the string
    ``delimeter``, into a dataframe with hierarchical columns.
    Inverse of ``from_hierarchical_columns``.
    ``https://datascience.stackexchange.com/q/9460/20612``
    
    Parameters
    ----------
    df: pandas.DataFrame
    delimeter: str, default '__'
        String delimiter for hierarchical column levels.
    deep: boolean, default False
        Whether retured array is a deep copy.
    Returns
    -------
    :pd.DataFrame
    '''
    df = df.copy(deep=deep)
    newcols = [tuple(c.split(delimeter)) if isinstance(c,str) else c for c in df.columns]
    df.columns = pd.MultiIndex.from_tuples(newcols)
    return df
    

def to_ndarray(df,flatten_columns=True):
    '''
    Convert DataFrame to structured np.ndarray.
    
    Parameters
    ----------
    df: pandas.DataFrame
    flatten_columns: boolean; default True
        Whether to flatten hierarchical columns by
        merging the coulmn-name tuples using the 
        delimeter ``'__'``.
        
    Returns
    -------
    :np.ndarray
        Dataframe as numpy structured array
    '''
    #https://stackoverflow.com/a/75039776/900749
    if flatten_columns:
        df = from_hierarchical_columns(df)
    records = df.to_records(index=False)
    return np.array(records, dtype = records.dtype.descr)

def camelcase_columns(df):
    '''
    Switch column names from Python's 
    ``underscore_convention`` to the Java/Matlab style
    ``lowerCaseCamelCase`` convention. 
    
    Your colleagues working in Matlab may appreciate this.
    This frees up underscore ``_`` to use as a delimeter
    for hierarchical columns.
    Inverse of ``snakecase_columns(df)``.
    
    Parameters
    ----------
    df: pandas.DataFrame
        
    Returns
    -------
    :pd.DataFrame
    '''
    try:
        df = from_hierarchical_columns(df,deep=False)
    except:
        pass
    from neurotools.util.string import snake_to_camel
    newcols = [
        '_'.join(map(snake_to_camel,c.split('__')))
        for c in df.columns]
    df.columns = newcols
    return df

def snakecase_columns(df):
    '''
    Switch column names from
    ``underscoreDelimited_camelCase`` 
    to flattened-hierarical ``snake_case__columns``. 
    Inverse of ``camelcase_columns(df)``.
    
    Parameters
    ----------
    df: pandas.DataFrame
        
    Returns
    -------
    :pd.DataFrame
    '''
    from neurotools.util.string import camel_to_snake
    df = from_hierarchical_columns(df)
    newcols = [
        '__'.join(map(camel_to_snake,c.split('_')))
        for c in df.columns]
    df.columns = newcols
    return df

def ndarray_from_mat(filename,key=None):
    '''
    Load structured ``np.ndarray`` from matfile
    
    Parameters
    ----------
    filename: str
        Matfile path to load
    flatten_columns: boolean; default True
        Whether to flatten hierarchical columns by
        merging the coulmn-name tuples using the 
        delimeter ``'__'``.
        
    Returns
    -------
    data:np.ndarray
        numpy structured array
    readme:str
        ``README`` string from this matfile, if present.
    '''
    data = scipy.io.loadmat(filename,squeeze_me=True)
    readme = [k for k in data.keys() if 'readme' in k.lower()]
    if len(readme)>1:
        warnings.warn((
            'Matfile has multiple README strings: '
            '%s; I am returning %s.')%(readme,readme[0])
            )
    README = data[readme[0]] if len(readme)>0 else ''
    for r in readme: del data[r]
    
    keys = [k for k in data.keys() if not k.startswith('__')]
    if key is None:
        if len(keys)!=1:
            raise UserWarning((
                'I expected a single key, got %s; Set the ``key``'
                ' argument to specify a particular key.')%keys
            )
        key =  keys[0]
    if not key in data:
        raise UserWarning((
            'Key "%s" not present in %s; '
            'Available keys are %s.')%(key,keys)
        )
    return data[key].squeeze(), README

def read_mat(
    filename,
    key=None,
    restore_hierarchical_columns=True,
    delimeter='__'):
    '''
    Load DataFrame from matfile.
    
    Parameters
    ----------
    filename: str
        Matfile path to load
    key: str; default None
        Name of variable to load from matfile. 
        If none, expects matfile containing only a single
        variable.
    restore_hierarchical_columns: boolean; default True
        ``to_mat`` flattens hierarchical columns by
        merging the coulmn-name tuples using the 
        delimeter ``'__'``. If true, we will restore the
        hierarchical column structre in the loaded
        DataFrame.
    delimeter: str, default '__'
        String delimiter for hierarchical column levels.
        
    Returns
    -------
    :pd.DataFrame
    '''
    a, README = ndarray_from_mat(filename,key=key)
    df = pd.DataFrame(a)
    if restore_hierarchical_columns:
        #snakecase_columns(df)
        df = to_hierarchical_columns(df)
    return df, README
    
def to_mat(df,saveas,tablename='data',readme=None):
    '''
    Save dataframe as ``.mat`` matfile.
    
    Parameters
    ----------
    df: pandas.DataFrame
    saveas: str
        File path to save to
    tablename: str; default 'data'
        Variable name to use for dataframe in the matfile.
    readme: str; default None
        Optional string to save in a ``README`` variable
        to document the contents of this ``.mat``
        archive.
    '''
    
    df = from_hierarchical_columns(df)
    #df = camelcase_columns(df)
    
    # Check column name lengths
    cc = df.columns
    lengths = np.int32([*map(len,cc)])
    bad = lengths>31
    if np.sum(bad):
        raise UserWarning((
            'Matlab column names are limited to 31 characters. '
            'I need to flatten hieraricical columns to save them in a '
            'matfile. The following column names are too long: '
            '\n  %s.')%('\n  '.join([
                '%s (length %d)'%(cc[i],lengths[i])
            for i in np.where(bad)[0]
        ])))
        
    df = to_ndarray(df)
    if not saveas.lower().endswith('.mat'):
        saveas += '.mat'
    savedict = {tablename:df}
    if not readme is None:
        savedict['README'] = str(readme)
    scipy.io.savemat(saveas,savedict)
    
    
def conform(x,dtype=np.float32,fill_value=np.NaN):
    '''
    Recursively coerce nested iterables into a numpy
    array with a uniform shape by padding uneven shapes
    with ``np.NaN`` or ``~0``.
    
    Useful for ensuring columns have a uniform
    type signature. 
    
    Parameters
    ----------
    x : iterable
        Hierarichal (nested) iterable structure that 
        we wish to coerce into a single ``np.ndarray``
    dtype: np.dtype; default np.float32
        Desired type of the resulting array. 
        Float and int types are supported.
        Float types will pad with ``np.NaN``.
        Int types will pad with ``~0``.
    fill_value: number; default np.NaN
        
    Returns
    -------
    :np.ndarray
    '''
    try:
        x = [*x]
        x = [conform(xi,dtype,fill_value) for xi in x]
    except TypeError:
        return dtype(x)
    try:
        x = dtype(x)
    except ValueError:
        shapes = [np.shape(q) for q in x]
        length = np.max([*map(len,shapes)])
        shapes = [s+(1,)*(length-len(s)) for s in shapes]
        shapes = np.int32(shapes)
        qq     = np.max(shapes,0)
        coerced = []
        for q in x:
            r = np.full(qq,fill_value,dtype=dtype)
            while len(q.shape)<length:
                q = q[...,None]
            ix = [slice(None,i,None) for i in q.shape]
            r[tuple(ix)] = q
            coerced.append(r)
        x = dtype(coerced)
    return x
    
    
def add_column_from_dict(df,coldata,default,name):
    df[name] = [coldata.get(i,default) for i in df.index]
    return df

def dfhcat(*args):
    args = [*args]
    if len(args)==1 and isinstance(args[0],list):
        args = args[0]
    return pd.concat(args, axis=1)
    
def dfvcat(*args):
    args = [*args]
    if len(args)==1 and isinstance(args[0],list):
        args = args[0]
    return pd.concat(args, axis=0)
    
def check_column_signs(
    df,
    ):
    # A bit silly really
    a = (np.sign(df)+1)/2
    m = a.mode(0).values
    if len(np.shape(m))>1:  m = m[0]
    differ    = np.abs(a - m)
    nmismatch = np.int32(np.nansum(differ,0))
    ntotal    = np.sum(np.isfinite(df),0).values
    return nmismatch, np.int32(ntotal )

@piper
def fdr(d):
    try:
        d['significant'],d['fdr0p05'] = nancorrect(d.pvalue)
        return d
    except AttributeError:
        return fdr(pd.DataFrame(d))

@piper
def colprefix(prefix,df=None):
    if df is None:
        # Curry mode
        return piper(lambda df2:colprefix(prefix,df2))
    # Normal mode
    # Try dataframe
    if isinstance(df,pd.DataFrame) or hasattr(df,'columns'):
        df.columns = [prefix+s for s in df.columns]
        return df
    # Try as dictionary
    if hasattr(df,'items'):
        return {prefix+k:v for k,v in df.items()}
    # Maybe its a named tuple? 
    if hasattr(df,'_asdict'):
        return colprefix(prefix,df._asdict())

@piper
def numeric_columns(x):
    return x[[c for c in x if x[c].dtype.kind in 'biufc']]








