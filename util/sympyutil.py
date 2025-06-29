#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sympy as sp

def vecsym(N,name,latex=None,**kw):
    if latex is None: 
        latex=name
    varnames  = [name+str(i)                            for i in range(1,N+1)]
    variables = [sp.Symbol(r'{%s}_{%d}'%(latex,i),**kw) for i in range(1,N+1)]
    vardict   = dict(zip(varnames,variables))
    vardict[name] = sp.Matrix([variables]).T
    return vardict

def matsym(N,K,name,latex=None,**kw):
    if latex is None: 
        latex=name
    loname = name.lower();
    hiname = name.upper();
    hitex  = latex.title();
    varnames  = [ loname+str(i)+str(j)                        for i in range(1,N+1)  for j in range(1,K+1)]
    variables = [[sp.Symbol(r'{%s}_{%d,%d}'%(latex,i,j),**kw) for i in range(1,N+1)] for j in range(1,K+1)]
    vardict   = dict(zip(varnames,[v for vv in variables for v in vv]))
    vardict[hiname] = sp.Matrix(variables).T
    return vardict
