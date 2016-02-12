
import datetime
import time as systime

def current_milli_time():
    int(round(systime.time() * 1000))

now = current_milli_time

def today():
    return datetime.date.today().strftime('%y%m%d')

#crude versions of tic and toc from Matlab
#stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python

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




