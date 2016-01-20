
import time as systime
current_milli_time = lambda: int(round(systime.time() * 1000))

def tic(st=''):
    global __GLOBAL_TIC_TIME__
    t = current_milli_time()
    if varexists('__GLOBAL_TIC_TIME__'):
        print 't=%dms'%((t-__GLOBAL_TIC_TIME__)),st
    else:
        print "timing..."
    __GLOBAL_TIC_TIME__ = current_milli_time()
    return t

def toc(st=''):
    global __GLOBAL_TIC_TIME__
    t = current_milli_time()
    if varexists('__GLOBAL_TIC_TIME__'):
        print 'dt=%dms'%((t-__GLOBAL_TIC_TIME__)),st
    else:
        print "you didn't call tic"
    return t

def waitfor(t):
    while current_milli_time()<t:
        pass
    return current_milli_time()

now = current_milli_time
