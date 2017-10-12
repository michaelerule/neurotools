#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import print_function

'''
Collected utilities for direct pixel rendering via matplotlib

It is difficult to write pixels directly in python.

One trick is to let Matplotlib do the heavy lifting for you.
At the cost of reverse-engineering how to access pixels in a 
Matplotlib plot, you have Matplotlib handle all th Tk/Qt/Agg 
backend stuff, so you get a common interface for multiple platforms.
'''
import matplotlib.pyplot as plt
import numpy as np
import sys

def hide_toolbar(fig):
    # Command to hide toolabr changes across versions and backends.
    # Without introspecting, try to hide toolbar
    try:
        fig.canvas.toolbar.setVisible(False)
    except AttributeError:
        try:
            fig.canvas.toolbar.pack_forget()
        except AttributeError:
            try:
                fig.canvas.toolbar.hide()
            except AttributeError:
                print('Failed to hide toolbar')

def start(w,h,title='untitled'):
    '''
    http://stackoverflow.com/questions/
    9295026/matplotlib-plots-removing-axis-legends-and-white-spaces
    '''
    
    # Create empty texture to start drawing
    draw = np.zeros((w,h,4),'float32')
    draw[...,-1] = 1.0

    # get image on screen -- unsure how to query dpi before drawing
    # so draw, get dpi, then resize to fit pixels.
    dpi = 80.0
    Win = w/dpi
    Hin = h/dpi
    fig = plt.figure(figsize=(Win,Hin),num=title)
    hide_toolbar(fig)
    dpi = fig.dpi
    Win = w/dpi
    Hin = h/dpi
    fig.set_size_inches((Win,Hin),forward=True)
    
    # draw image    
    ax  = plt.subplot(111)
    fig.subplots_adjust(top=1,bottom=0,left=0,right=1)
    img = ax.imshow(draw,interpolation='nearest',animated=True)
    
    ax.set_xlim(w,-1)
    ax.set_ylim(h,-1)    
    ax.set_axis_off()
    
    return fig,img

def draw_array(screen,rgbdata):
    fig,img = screen
    # prepare image data 
    # clip bytes to 0..255 range
    rgbdata[rgbdata<0]=0
    rgbdata[rgbdata>1]=1
    rgbdata = np.float32(rgbdata)
    # get color dimension
    if len(rgbdata.shape)==3:
        w,h,d = rgbdata.shape
    else:
        w,h = rgbdata.shape
        d=1

    # repack color data in screen format.
    # for matplotlib, colors are float 4 vectors in [0,1], RGBA order
    draw = np.zeros((w,h,4),'float32')
    if d==1:
        draw[...,0]=rgbdata
        draw[...,1]=rgbdata
        draw[...,2]=rgbdata
        draw[...,3]=1 # alpha channel??
    if d==3:
        draw[...,:3]=rgbdata
        draw[...,-1]=1 # alpha channel
    if d==4:
        draw[...,:]=rgbdata

    img.set_data(draw)
    plt.draw()

