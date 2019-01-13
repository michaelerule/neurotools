#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt

from neurotools.graphics.plot import noaxis,nox,noy

def list_like(tree):
    return hasattr(tree, '__iter__') and not type(tree) is str

def get_depth(tree):
    '''
    Get depth of tree-like structure. 
    The tree can be given as any nested structure of iterables
    (strings are not counted as iterable, however)
    '''
    if list_like(tree):
        return np.max([get_depth(t) for t in tree])+1
    return 1

def count_nodes(tree):
    '''
    Count leaf nodes of tree-like structure. 
    The tree can be given as any nested structure of iterables
    (strings are not counted as iterable, however)
    '''
    if list_like(tree):
        return np.sum([count_nodes(t) for t in tree])
    return 1
    
def inorder_traversal(tree):
    '''
    Return leaves of the tree in order. 
    Can also be used like a "deep flatten" command to 
    flatten nested list and tuple structures.
    The tree can be given as any nested structure of iterables
    (strings are not counted as iterable, however)
    '''
    if list_like(tree):
        order = []
        for t in tree:
            order += inorder_traversal(t)
        return order
    else:
        return [tree]

def plot_brackets(tree,lw=1,color='k'):
    xlevel = [0]
    dx,dy=30,20
    maxdepth = get_depth(tree)
    nnodes   = count_nodes(tree)
    SIZE     = 20
    plt.figure(figsize=(SIZE,SIZE*maxdepth/nnodes))
    def lineplot(x,y):
        plt.plot(x,y,lw=lw,color=color)
    def helper(tree,depth=0,x=0,y=0):
        if list_like(tree):
            tdepth = get_depth(tree)
            xs,ys = np.array([helper(t) for t in tree]).T
            ymax = np.max(ys)
            midx = np.mean(xs)
            ybar = ymax+dy
            for x,y in zip(xs,ys):
                lineplot((x,x),(ybar,y))
            minx = np.min(xs)
            maxx = np.max(xs)
            lineplot((minx,maxx),(ybar,ybar))
            return midx,ybar
        else:
            name = '(%s)'%tree
            x = xlevel[0]
            xlevel[0]+=dx
            y = 0
            plt.text(x,y-0.5*dy,name,
                 horizontalalignment='center',
                 verticalalignment='top',
                 fontsize=9,
                 rotation=90)
            return x,y
    helper(tree)
    plt.axis('equal')
    noaxis(); nox(); noy();

def plot_brackets_polar(tree,
    lw=0.5,
    color='k',
    rotate_labels=True,
    label_offset=3,
    arcspan = 1,
    angle   = 0,
    figsize = 9,
    fontsize= 9,
    line_resolution = 50):
    '''
    '''
    plt.figure(figsize=(figsize,figsize))
    xlevel = [0]
    maxdepth = get_depth(tree)
    nnodes   = count_nodes(tree)
    print(maxdepth,nnodes)
    t = np.linspace(0,1,line_resolution)
    arcspan = arcspan*(nnodes-1)/nnodes
    def topolar(x,y):
        theta= x/(nnodes-1)*np.pi*2*arcspan + 0.5*(1-arcspan)*2*np.pi + angle*2*np.pi
        r=(1+1e-9-y/maxdepth)
        px = r*np.cos(theta)
        py = r*np.sin(theta)
        return r,theta,px,py
    def lineplot(x,y):
        tx = t*np.diff(x)+x[0]
        ty = t*np.diff(y)+y[0]
        r,theta,px,py = topolar(tx,ty)
        plt.plot(px,py,lw=lw,color=color)
    def helper(tree,depth=0,x=0,y=0):
        if list_like(tree):
            tdepth = get_depth(tree)
            xs,ys = np.array([helper(t) for t in tree]).T
            ymax = np.max(ys)
            mi1 = np.mean(xs)
            ybar = ymax+1
            for x,y in zip(xs,ys):
                lineplot((x,x),(ybar,y))
            minx = np.min(xs)
            maxx = np.max(xs)
            lineplot((minx,maxx),(ybar,ybar))
            return mi1,ybar
        else:
            name = str(tree)
            x = xlevel[0]
            xlevel[0]+=1
            y = 0
            if rotate_labels:
                #dr = len(name)//2
                r,theta,px,py = topolar(x,y-label_offset)
                theta = theta*180/np.pi
                theta = (theta+3*360)%360
                if abs(theta-180)<90:
                    theta = (theta+180+360)%360
                plt.text(px,py,name,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=fontsize,
                     rotation=theta)
            else:
                r,theta,px,py = topolar(x,y-label_offset)
                plt.text(px,py,name,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=fontsize,
                     rotation=0)
            return x,y
    helper(tree)
    plt.axis('equal')
    noaxis(); nox(); noy();
