#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Routines for meshes and triangulation
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from neurotools.graphics.plot import force_aspect
from collections import defaultdict

import scipy
import scipy.spatial

def z2xy(z):
    '''
    Converts an array of complex numbers into two arrays
    representing real and imaginary parts, respectively.
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    return np.real(z),np.imag(z)

def uniquerow(x):
    '''
    Removes duplicate rows from a 2D numpy array
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    return np.array(list(set(map(tuple,x))))

def trianglesToEdges(triangles):
    '''
    Accepts Ntriangles x 3 array of triangle indeces, the format
    returned by `scipy.spatial.Delaunay(...).simplices`. Returns a
    Nedges x 2 numpy array of unique edges in the triangulation
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    tedges    = triangles[:,[0,1,1,2,2,0]].reshape((np.size(triangles),2))
    tedges    = uniquerow(np.sort(tedges,axis=1))
    return tedges

def edgesNearby(iz,microd):
    '''
    Returns a dictionary mapping from 
    indecies into point list iz 
    (2d locations passed as x+iy complex)
    to a list of nearby point indices
    
    Computed by thresholding delaunay triangulation
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    points    = np.array([np.real(iz),np.imag(iz)])
    triangles = scipy.spatial.Delaunay(points.T).simplices
    tedges    = trianglesToEdges(triangles)
    edgelen   = np.ravel(np.abs(np.diff(iz[tedges],axis=1)))
    tedges    = tedges[edgelen<microd,:]
    
    #tedges = concatenate([tedges,tedges[:,[1,0]]])
    #coordsparse = scipy.sparse.coo_matrix((ones(tedges.shape[0]),(tedges[:,0],tedges[:,1])))
    #edgelist = scipy.sparse.csr_matrix(coordsparse)
    
    edgelist  = defaultdict(set)
    for i,z in enumerate(iz):
        edgelist[i] = tuple(np.ravel(tedges[(tedges==i)[:,[1,0]]]))   
    
    return edgelist

def coalesce(iz,edgelist):
    '''
    Join connected components as defined in edgelist, and return the centroids
    taken as an average of all point locations in list iz 
    (2d locations passed as x+iy complex)
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    # Make a copy
    remaining  = dict(edgelist)
    components = []
    while len(remaining)>0:
        # as long as there are unexplored components,
        # choose a component to explore
        i,tosearch = list(remaining.items())[0]
        component = {i}
        del remaining[i]
        while len(tosearch)>0:
            # Start with the nodes connected to the first
            # node. Remove each neighbor from the remaining set
            # and explore all nodes connected to that neighbor
            new = set()
            for e in tosearch:
                component |= {e}
                if e in remaining:
                    new |= set(remaining[e])
                    del remaining[e]
            tosearch = new
        components += [component]
    centroids = np.array([np.mean(iz[np.array(list(c))]) for c in components])
    return centroids

def plot_edges(iz,edges,**kwargs):
    '''
    Plots a set of edges given by 2d complex numbers and Nedges x 2 array of edge indices
    keword arguments are forwarded to matplotlib.plot
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    edgelist = np.ravel(np.concatenate([edges[:,:1]*np.NaN,iz[edges]],axis=1))
    plt.plot(*z2xy(np.array(edgelist)),**kwargs)
    force_aspect()
    
def plot_triangles(iz,triangles,**kwargs):
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    edges = trianglesToEdges(triangles)
    edgelist = np.ravel(np.concatenate([edges[:,:1]*np.NaN,iz[edges]],axis=1))
    plt.plot(*z2xy(np.array(edgelist)),**kwargs)
    force_aspect()

def mergeNearby(x,y,radius):
    '''
    Merge nearby points
    
    Parameters
    ----------
    
    Returns
    -------
    '''
    iz = x+ 1j*y
    return z2xy(coalesce(iz,edgesNearby(iz,radius)))
    
