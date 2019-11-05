#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import print_function

'''
Collected utilities for pygame

It is difficult to write pixels directly in python.

There's some way to get a framebuffer back from Tk, but it is 
cumberosme. 

The module pygame supports sending pixel buffers, 
which is wrapped for convneinece in this module.

example usage

import neurotools.graphics.pygame as npg
import time
import numpy as np
import pygame
K = 128
screen = npg.start(K,K,'Image data')
dt = 1/20
wait_til = time.time() + dt
print('Animating..')
for i in neurotools.tools.progress_bar(range(100)):
    t = time.time()
    if t<wait_til: time.sleep(wait_til-t)
    wait_til = t + dt
    npg.draw_array(screen, np.random.rand(K,K,3))
pygame.quit()
'''

import sys
import numpy as np

try:
    import pygame as pg
except:
    print('pygame package is missing; it is obsolete so this is not unusual')
    print('pygame graphics will not work')
    pg = None

def enable_vsync():
    if sys.platform != 'darwin':
        return
    try:
        import ctypes
        import ctypes.util
        ogl = ctypes.cdll.LoadLibrary(ctypes.util.find_library("OpenGL"))
        # set v to 1 to enable vsync, 0 to disable vsync
        v = ctypes.c_int(1)
        ogl.CGLSetParameter(ogl.CGLGetCurrentContext(), ctypes.c_int(222), ctypes.pointer(v))
    except:
        print("Unable to set vsync mode, using driver defaults")

def start(W,H,name='untitled'):
    # Get things going
    pg.quit()
    pg.init()
    enable_vsync()
    window  = pg.display.set_mode((W,H))
    pg.display.set_caption(name)
    return window

def draw_array(screen,rgbdata,doshow=True):
    '''
    Send array data to a PyGame window.
    PyGame is BRG order which is unusual -- reorder it.

    Parameters
    ----------
    screen : object
        Object returned by neurotools.graphics.pygame.start
    rgbdata : 
        RGB image data with color values in [0,1]
    '''
    # Cast to int
    rgbdata = np.int32(rgbdata*255)
    # clip bytes to 0..255 range
    rgbdata[rgbdata<0]=0
    rgbdata[rgbdata>255]=255
    # get color dimension
    if len(rgbdata.shape)==3:
        w,h,d = rgbdata.shape
    else:
        w,h = rgbdata.shape
        d=1
    # repack color data in screen format
    draw = np.zeros((w,h,4),'uint8')
    if d==1:
        draw[...,0]=rgbdata
        draw[...,1]=rgbdata
        draw[...,2]=rgbdata
        draw[...,3]=255 # alpha channel
    if d==3:
        draw[...,:3]=rgbdata[...,::-1]
        draw[...,-1]=255 # alpha channel
    if d==4:
        draw[...,:3]=rgbdata[...,-2::-1]
        draw[...,-1]=rgbdata[...,-1]
    # get surface and copy data to sceeen
    surface = pg.Surface((w,h))
    numpy_surface = np.frombuffer(surface.get_buffer())
    numpy_surface[...] = np.frombuffer(draw)
    del numpy_surface
    screen.blit(surface,(0,0))
    if doshow:
        pg.display.update()


