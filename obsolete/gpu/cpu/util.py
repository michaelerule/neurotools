#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Helper functions
"""
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function



def document(name,docstring):
    '''
    This function adds in documentation to anonymous (lambda) functions. 
    Sphinx autodoc will also accept the first triple-quote string folloing 
    a lambda function declataion as the docstring
    '''
    exec("%s.__doc__=docstring"%name)
    exec("%s.__name__=name"%name)

def flat(arr):
    '''
    A naive flatten functions : concatenates a list of lists via lots of
    copying
    '''
    res = [];
    for a in arr:
        res.extend(a)
    return res

compose = lambda f:lambda g:lambda x:f(g(x))
'''
Curried function composition operator. Use as compose(f)(g) to make a 
composed function.
'''

dot = lambda a,b:sum(lmul(a,b))
'''
dot = lambda a,b:sum(lmul(a,b))
Computes the dot product of two vectors
'''

length = lambda V:sqrt(dot(V,V))
'''
Computes the magnitude of an array, interpreted as an N-vector
'''

idivup = lambda a,b:(a/b,a/b+1)[a%b!=0]
'''
Divides a by b, rounding up to the nearest integer. Eqivalent to ceil(a/b)
'''

fancy  = lambda s:'$\mathrm{%s}$'%re.sub(r" ", r"\ ",s)
'''
This will wap a string into the math markup code for pyplot
'''

elem   = lambda f:lambda a,b:[f(x,y) for (x,y) in zip(a,b)]
'''
Curried function for creating elementwise binary list operators. 
For example, elem(lambda x,y:x+y)(listA)(listB) will create a third list
containing elementwise sum of listA and listB. Equivalently, cmap(lambda x,y:x+y)(zip(listA,listB))
'''

cmap   = lambda f:lambda a:[f(x) for x in a]
'''
This is a curried verion of the map operator. cmap(f) will create a map version of f. For example, cmap(lambda x:x+1) creates a map of the succesor function. cmap(lambda x:x+1)(list) would return a new list with 1 added to every element of the argument list
'''

mmap   = compose(cmap)(cmap)
'''
mmap   = lambda f:cmap(cmap(f))
This is a curried version of map that operates elementwise over lists of lists
'''

ldif   = elem(lambda x,y:x-y)
'''
Element-wise difference of two lists
'''

lsum   = elem(lambda x,y:x+y)
'''
Element-wise sum of two lists
'''

lmul   = elem(lambda x,y:x*y)
'''
Element-wise product of two lists
'''

ldiv   = elem(lambda x,y:x/y)
'''
Element-wise ratio of two lists
'''

lpow   = elem(lambda x,y:x**y)
'''
Element-wise exponentiation of two lists
'''

ldif2  = cmap(lambda x,y:x-y)
'''
Element-wise difference from a list of pairs
'''

lsum2  = cmap(lambda x,y:x+y)
'''
Element-wise sum from a list of pairs
'''

lmul2  = cmap(lambda x,y:x*y)
'''
Element-wise product from a list of pairs
'''

ldiv2  = cmap(lambda x,y:x/y)
'''
Element-wise ratio from a list of pairs
'''

lpow2  = cmap(lambda x,y:x**y)
'''
Element-wise exponentiation from a list of pairs
'''

scale  = lambda x:cmap(lambda y:y*x)
'''
Curried scalar multiplication operator. E.g. scale(2) produces a function that will double each element in a list passed to it
'''

shift  = lambda x:cmap(lambda y:y+x)
'''
Curried scalar shift operator. E.g. shift(2) produces a function that will add 2 to each element in a list passed to it
'''

mu     = lambda v: sum(v)/len(v)
var    = lambda v:mean((lambda mu:cmap(lambda x:(x-mu)**2)(v))(mu(v)))
'''
Population variance of a list
'''

svar   = lambda v:(lambda l:l/(l-1))(len(v))*var(v)
'''
Sample variance of a list
'''

sigma  = lambda v:var(v)**0.5
mean   = mu
'''
Population average of a list
'''

std    = sigma
'''
Population standard deviation of a list
'''

sstd   = lambda v:svar(v)**0.5
'''
Sample standard deviation of a list
'''

sem    = lambda v:(svar(v)/len(v))**0.5
'''
Standard error of mean for a list
'''

norm   = lambda L:(lambda s:cmap(lambda x:x*s)(L))(1.0/ sum(L))
'''
Interprets a list as a vector and normalizes its magnitude to 1
'''

cov    = lambda a,b:mu(lmul(a,b))-mu(a)*mu(b)
'''
Covariance of two lists
'''

corr   = lambda a,b:cov(a,b)/(sigma(a)*sigma(b))
'''
Pearson's correlation coefficient of two lists
'''

mdif   = elem(ldif)
'''
Elementwise difference for a matrix (list of lists)
'''

msum   = elem(lsum)
'''
Elementwise sum for a matrix (list of lists)
'''

mmul   = elem(lmul)
'''
Elementwise product for a matrix (list of lists)
'''

mdiv   = elem(ldiv)
'''
Elementwise ratio for a matrix (list of lists)
'''

mpow   = elem(lpow)
'''
Elementwise exponentiation for a matrix (list of lists)
'''

mmean  = elem(mu)
'''
Returns the population average of each row of a matrix
'''

mstd   = elem(sigma)
'''
Returns the population standard deviation of each row of a matrix
'''

mvar   = elem(var)
'''
Returns the population variance deviation of each row of a matrix
'''

rnorm  = cmap(norm)
'''
Normalizes each row of a matrix independently
'''

cut    = lambda mat,rowlen:[mat[i:i+rowlen] for i in xrange(0,len(mat),rowlen)]
'''
Cuts any subscriptable object into a list of ranges of that object. This can facilitate creation of a matrix like object form a row major packed list. For instance, if you had a 100x100 GPUArray and wanted to represent it as a list of its rows, cut(array,100) would return a list of 100 size 100 slices of your original object. Cut uses range subscripting and the returned object will most likely point to the same underlying section of memory as the argument array
'''
    

