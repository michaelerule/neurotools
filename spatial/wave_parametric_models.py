"""
Apply a wave equation fit to the beta oscillations. 
as in Zanos et al 2015

Thy use a 2ms step size
Fit wave equation over 20ms ( not even full beta period? )

they use lsqcurvefit in Matlab to fit the wave equation.
there is a python equivalent scipy.optimize.leastsq

2D wave equation is

u(x,y,t) = A(t) np.sin(Kx(t)*x+Ky(t)*y-w(t)*t+phi(t))

A(t)   is time varying amplitude
Kx(t)  is a time varying contribution from x spatial component
Ky(t)  is a time varying contribution from y spatial component
w(t)   is current wavelength in time
phi(t) is a phase parameter

We don't explicitly model time -- we take short snapshots and fit the wave
equation. So what we're actually fitting is 

u(x,y,t) = A np.sin(a*x+b*y-w*t+phi)

Which has 5 free Parameters amplitude, x/y spatial wavelength, 
time wavelength, and phase offset.
"""

import numpy as np
import matplotlib.pyplot as plt

zscore = lambda x: (x-np.mean(x,0))/std(x,0)

def predict(xys,times,A,B,a,b,w):
    '''
    '''
    nxy = np.shape(xys)[0]
    nt  = np.shape(times)[0]
    predicted = np.zeros((nt,nxy))
    for it in range(nt):
        for ixy in range(nxy):
            x,y = xys[ixy]
            t = times[it]
            phase = a*x+b*y-w*t
            predicted[it,ixy] = A*np.sin(phase)+B*np.cos(phase)
    return predicted

def plotdata(xys,data):
    '''
    '''
    x,y   = xys.T
    scale = 20
    for frame in data:
        plt.clf()
        plt.scatter(x,y,s=frame*scale,color='b')
        plt.scatter(x,y,s=-frame*scale,color='r')
        plt.draw()
        plt.show()


def makeLSQminimizerPolar(xy,time,neuraldata):
    '''
    Generates a suitable function for computing an objective function for
    least-squares minimization of a synchronous wave model
    

    Parameters
    ----------
    xy : 2D numeric array
        locations in space of observations
    time : 1D numeric array
        time-points of observations
    neuraldata:
        experimental observations to which to fit the model
        a Ntimes x NElectrode filtered neural data snippit

    Returns
    -------
    function
        An objective function that can be used with the Numpy leastsq 
        optimizer function
    '''
    nxy  = np.shape(xy)[0]
    nt   = np.shape(time)[0]
    time -= np.mean(time)
    xy   -= np.mean(xy,0)
    window = np.hanning(nt+2)[1:-1]
    def getResiduals(params):
        A,B,a,b,w,xo,yo = params
        residuals = np.zeros((nxy,nt))
        for ixy in range(nxy):
            for it in range(nt):
                x,y = xy[ixy]
                t = time[it]
                x -= xo
                y -= yo
                r = sqrt(x*x+y*y)
                h = arctan2(y,x)
                phase = a*r+b*h-w*t
                prediction = A*np.sin(phase)+B*np.cos(phase)
                residuals[ixy,it] = np.abs(neuraldata[it,ixy] - prediction)*window[it]
        return np.ravel(residuals)
    return getResiduals


def makeLSQminimizerStanding(xy,time,neuraldata):
    '''
    Generates a suitable function for computing an objective function for
    least-squares minimization of a synchronous wave model
    

    Parameters
    ----------
    xy : 2D numeric array
        locations in space of observations
    time : 1D numeric array
        time-points of observations
    neuraldata:
        experimental observations to which to fit the model
        a Ntimes x NElectrode filtered neural data snippit
        
    Returns
    -------
    function
        An objective function that can be used with the Numpy leastsq 
        optimizer function
    '''
    nxy  = np.shape(xy)[0]
    nt   = np.shape(time)[0]
    time -= np.mean(time)
    xy   -= np.mean(xy,0)
    window = np.hanning(nt+2)[1:-1]
    def getResiduals(params):
        A,B,C,D,a,b,w = params
        residuals = np.zeros((nxy,nt))
        for ixy in range(nxy):
            for it in range(nt):
                x,y = xy[ixy]
                t = time[it]
                phase1 = a*x+b*y
                phase2 = w*t
                cp1  = np.cos(phase1)
                sp1  = np.sin(phase1)
                cp2  = np.cos(phase2)
                sp2  = np.sin(phase2)
                prediction = A*sp1*sp2+B*sp1*cp2+C*cp1*sp2+D*cp1*cp2
                residuals[ixy,it] = np.abs(neuraldata[it,ixy] - prediction)*window[it]
        return np.ravel(residuals)
    return getResiduals


def makeLSQminimizerSynchronous(xy,time,neuraldata):
    '''
    Generates a suitable function for computing an objective function for
    least-squares minimization of a synchronous wave model
    

    Parameters
    ----------
    xy : 2D numeric array
        locations in space of observations
    time : 1D numeric array
        time-points of observations
    neuraldata:
        experimental observations to which to fit the model
        a Ntimes x NElectrode filtered neural data snippit
        
    Returns
    -------
    function
        An objective function that can be used with the Numpy leastsq 
        optimizer function
    '''
    nxy   = np.shape(xy)[0]
    nt    = np.shape(time)[0]
    time -= np.mean(time)
    xy   -= np.mean(xy,0)
    window = np.hanning(nt+2)[1:-1]
    def getResiduals(params):
        A,B,w = params
        residuals = np.zeros((nxy,nt))
        for ixy in range(nxy):
            for it in range(nt):
                x,y = xy[ixy]
                t = time[it]
                phase = w*t
                prediction = A*np.cos(phase)+B*np.sin(phase)
                residuals[ixy,it] = np.abs(neuraldata[it,ixy] - prediction)*window[it]
        return np.ravel(residuals)
    return getResiduals


def makeLSQminimizerPlane(xy,time,neuraldata):
    '''
    Generates a suitable function for computing an objective function for
    least-squares minimization of a synchronous wave model

    Parameters
    ----------
    xy : 2D numeric array
        locations in space of observations
    time : 1D numeric array
        time-points of observations
    neuraldata:
        experimental observations to which to fit the model
        a Ntimes x NElectrode filtered neural data snippit

    Returns
    -------
    function
        An objective function that can be used with the Numpy leastsq 
        optimizer function
    '''
    nxy  = np.shape(xy)[0]
    nt   = np.shape(time)[0]
    time -= np.mean(time)
    xy   -= np.mean(xy,0)
    window = np.hanning(nt+2)[1:-1]
    def getResiduals(params):
        A,B,a,b,w = params
        residuals = np.zeros((nxy,nt))
        for ixy in range(nxy):
            for it in range(nt):
                x,y = xy[ixy]
                t = time[it]
                phase = a*x+b*y-w*t
                prediction = A*np.sin(phase)+B*np.cos(phase)
                residuals[ixy,it] = np.abs(neuraldata[it,ixy] - prediction)*window[it]
        return np.ravel(residuals)
    return getResiduals


def makeLSQminimizerDoublePlane(xy,time,neuraldata):
    '''
    Generates a suitable function for computing an objective function for
    least-squares minimization of a synchronous wave model

    Parameters
    ----------
    xy : 2D numeric array
        locations in space of observations
    time : 1D numeric array
        time-points of observations
    neuraldata:
        experimental observations to which to fit the model
        a Ntimes x NElectrode filtered neural data snippit

    Returns
    -------
    function
        An objective function that can be used with the Numpy leastsq 
        optimizer function
    '''
    nxy  = np.shape(xy)[0]
    nt   = np.shape(time)[0]
    time -= np.mean(time)
    xy   -= np.mean(xy,0)
    window = np.hanning(nt+2)[1:-1]
    def getResiduals(params):
        A1,B1,a1,b1,w1,A2,B2,a2,b2,w2 = params
        residuals = np.zeros((nxy,nt))
        for ixy in range(nxy):
            for it in range(nt):
                x,y = xy[ixy]
                t = time[it]
                phase1 = a1*x+b1*y-w1*t
                phase2 = a2*x+b2*y-w2*t
                prediction = A1*np.sin(phase1)+B1*np.cos(phase1)+A2*np.sin(phase2)+B2*np.cos(phase2)
                residuals[ixy,it] = np.abs(neuraldata[it,ixy] - prediction)*window[it]
        return np.ravel(residuals)
    return getResiduals

def phase_gradient(data):
    '''
    Computes 1D linear phase gradient
    '''
    data = np.angle(data)
    phase_gradient = np.diff(data)
    phase_gradient = (phase_gradient+pi)%(2*pi)-pi
    return phase_gradient

def heuristic_B_polar(data,xys,times):
    '''
    Heuristic parameter guess for the polar wave model

    Parameters
    ----------
    xys : 2D numeric array
        locations in space of observations
    timew : 1D numeric array
        time-points of observations
    data:
        experimental observations to which to fit the model
        a Ntimes x NElectrode filtered neural data snippit

    Returns
    -------
    tuple
        Amplitude, ?, ?, ?, frequency
    ''' 
    amplitude_guess = np.max(np.abs(data))
    frequency_guess = np.median(list(map(phase_gradient,data.T))/np.mean(np.diff(times)))
    x,y = xys.T
    return np.array([amplitude_guess,0,0,0,frequency_guess,np.mean(x),np.mean(y)])

def heuristic_B_planar(data,xys,times):
    '''
    Heuristic parameter guess for the planar wave model

    Parameters
    ----------
    xys : 2D numeric array
        locations in space of observations
    timew : 1D numeric array
        time-points of observations
    data:
        experimental observations to which to fit the model
        a Ntimes x NElectrode filtered neural data snippit

    Returns
    -------
    tuple
        Amplitude, ?, ?, ?, frequency
    ''' 
    amplitude_guess = np.max(np.abs(data))
    frequency_guess = np.median(map(phase_gradient,data.T))/np.mean(np.diff(times))
    return np.array([amplitude_guess,0,0,0,frequency_guess])
    
def heuristic_B_standing(data,xys,times):
    '''
    Heuristic parameter guess for the standing wave model

    Parameters
    ----------
    xys : 2D numeric array
        locations in space of observations
    timew : 1D numeric array
        time-points of observations
    data:
        experimental observations to which to fit the model
        a Ntimes x NElectrode filtered neural data snippit

    Returns
    -------
    tuple
        Amplitude, ?, ?, ?,  ?, ?, frequency
    ''' 
    amplitude_guess = np.max(np.abs(data))
    frequency_guess = np.median(list(map(phase_gradient,data.T))/np.mean(np.diff(times)))
    return np.array([amplitude_guess,0,0,0,0,0,frequency_guess])
    
def heuristic_B_synchronous(data,xys,times):
    '''
    Heuristic parameter guess for the spatially synchronous wave model

    Parameters
    ----------
    xys : 2D numeric array
        locations in space of observations
    timew : 1D numeric array
        time-points of observations
    data:
        experimental observations to which to fit the model
        a Ntimes x NElectrode filtered neural data snippit

    Returns
    -------
    tuple
        Amplitude, ?, frequency
    '''
    amplitude_guess = np.max(np.abs(data))
    frequency_guess = np.median(map(phase_gradient,data.T))/np.mean(np.diff(times))
    return np.array([amplitude_guess,0,frequency_guess])

def heuristic_B_double_planar(data,xys,times):
    '''
    Heuristic parameter guess for the double planar wave model

    Parameters
    ----------
    xys : 2D numeric array
        locations in space of observations
    timew : 1D numeric array
        time-points of observations
    data:
        experimental observations to which to fit the model
        a Ntimes x NElectrode filtered neural data snippit

    Returns
    -------
    tuple
        Amplitude, ?, ?, ?, frequency, ? , ampltiude2, 0.1, -0.1, frequency2
    ''' 
    amplitude_guess = np.max(np.abs(data))
    frequency_guess = np.median(list(map(phase_gradient,data.T))/np.mean(np.diff(times)))
    return np.array([amplitude_guess,0,0,0,frequency_guess,0,amplitude_guess,0.1,-0.1,frequency_guess])

def frame_synchrony(frame):
    '''
    Non-Kuromoto synchrony measure
    '''
    return np.abs(np.mean(frame))/np.mean(np.abs(frame))

def synchrony(data):
    '''
    Just maps frame_synchrony(frame) over first dimention of parameter data
    '''
    syn = [frame_synchrony(frame) for frame in data]
    return np.mean(syn)

def pairwise_phase_difference(a,b):
    '''
    Phase difference, compensated for wraparound
    '''
    return (a-b+pi)%(2*pi)-pi

def spatial_phase_gradient(arraymap,chi,frame):
    '''
    Computes phase gradient from electrode positions as opposed to the 
    array-packed representation of data? 
    I think?
    '''
    # PGD = |E(phase)|/E(|phase|)
    frame = np.angle(frame)
    height,width = np.shape(arraymap)
    gradients = []
    for y in range(height-1):
        for x in range(width-1):
            ch0 = arraymap[y][x]
            chx = arraymap[y][x+1]
            chy = arraymap[y+1][x]
            ch3 = arraymap[y+1][x+1]
            if ch0==0: continue
            if chx==0: continue
            if chy==0: continue
            if ch3==0: continue
            if not ch0 in chi: continue
            if not chx in chi: continue
            if not chy in chi: continue
            if not ch3 in chi: continue
            ch0 = np.where(chi==ch0)
            chx = np.where(chi==chx)
            chy = np.where(chi==chy)
            ch3 = np.where(chi==ch3)
            dx = pairwise_phase_difference(frame[ch0],frame[chx])
            dy = pairwise_phase_difference(frame[ch0],frame[chy])
            dx+= pairwise_phase_difference(frame[chy],frame[ch3])
            dy+= pairwise_phase_difference(frame[chx],frame[ch3])
            dz = (dx+1j*dy)*0.5
            gradients.np.append(dz)
    gradients = np.array(gradients)
    return gradients

def directionality_index(arraymap,chi,frame):
    '''
    PGD
    '''
    # PGD = |E(phase)|/E(|phase|)
    frame = np.angle(frame)
    height,width = np.shape(arraymap)
    gradients = spatial_phase_gradient(arraymap,chi,frame)
    return np.abs(np.mean(gradients))/np.mean(np.abs(gradients))

def phase_unwrap(x):
    x = np.angle(x)
    x = np.diff(x)
    x = (x+pi)%(2*pi)-pi
    return np.append(0,np.cumsum(x))+x[0]

def averaged_directionality_index(a,c,x):
    # note: this failes.
    # meanphase = np.array([np.mean(phase_unwrap(x[:,i])) for i in xrange(np.shape(x)[1])])
    # meanphase %= 2*pi
    # return directionality_index(arraymap,chi,exp(1j*meanphase))
    # this is better
    gradients = [spatial_phase_gradient(a,c,f) for f in x]
    # f = np.median(map(phase_gradient,data.T))
    # f is in units of d_phase d_t, can be used to recenter gradients for averaging?
    # wait... there is no need to re-center gradients. 
    # there is no evidence that hatsopoulos averaged PGD in time?
    

def heuristic_solver_double_planar(params):
    '''
    Heuristic fit of data to a wave solution with two plane waves.
    Intended to be used with neurotools.parallel
    

    Parameters
    ----------
    i    : integer
        the job number (will be returned with the result)
    xys  : 2D numeric array
        spatial locations of each channel
    times: 1D numeric array
        the time basis for the observation
    data : 3D numeric array, real valued
        wave data. 
        a Ntimes x NElectrode filtered neural data snippit

    Returns
    -------
    i : 
        job index for tracking parallel jobs
    result[0] : 
        first element of tuple returned from leastsq. presumably the model
        parameters?
    error : 
        norm of the residuals divided by the norm of the data
    '''
    (i,xys,times,data) = params
    objective = makeLSQminimizerDoublePlane(xys,times,real(data))
    result = leastsq(objective,heuristic_B_double_planar(data,xys),full_output=1)
    return i,result[0],norm(result[2]['fvec'])/norm(data)


def heuristic_solver_standing(params):
    '''
    Heuristic fit of data to a planar standing wave solution.
    Intended to be used with neurotools.parallel
    

    Parameters
    ----------
    i    : integer
        the job number (will be returned with the result)
    xys  : 2D numeric array
        spatial locations of each channel
    times: 1D numeric array
        the time basis for the observation
    data : 3D numeric array, real valued
        wave data. 
        a Ntimes x NElectrode filtered neural data snippit

    Returns
    -------
    int
        job index for tracking parallel jobs
    object 
        first element of tuple returned from leastsq. presumably the model
        parameters?
    float 
        norm of the residuals divided by the norm of the data
    '''
    (i,xys,times,data) = params
    objective = makeLSQminimizerStanding(xys,times,real(data))
    result = leastsq(objective,heuristic_B_standing(data,xys),full_output=1)
    return i,result[0],norm(result[2]['fvec'])/norm(data)

def heuristic_solver_planar(params):
    '''
    Heuristic fit of data to a plane wave solution.
    Intended to be used with neurotools.parallel
    

    Parameters
    ----------
    i    : integer
        the job number (will be returned with the result)
    xys  : 2D numeric array
        spatial locations of each channel
    times: 1D numeric array
        the time basis for the observation
    data : 3D numeric array, real valued
        wave data. 
        a Ntimes x NElectrode filtered neural data snippit

    Returns
    -------
    int
        job index for tracking parallel jobs
    object 
        first element of tuple returned from leastsq. presumably the model
        parameters?
    float 
        norm of the residuals divided by the norm of the data
    '''
    (i,xys,times,data) = params
    objective = makeLSQminimizerPlane(xys,times,real(data))
    result = leastsq(objective,heuristic_B_planar(data,xys),full_output=1)
    return i,result[0],norm(result[2]['fvec'])/norm(data)
    
def heuristic_solver_polar(params):
    '''
    Heuristic fit of data to a polar wave solution.
    Polar waves include radial, spiral, and pinwheel rotating waves
    Intended to be used with neurotools.parallel
    

    Parameters
    ----------
    i    : integer
        the job number (will be returned with the result)
    xys  : 2D numeric array
        spatial locations of each channel
    times: 1D numeric array
        the time basis for the observation
    data : 3D numeric array, real valued
        wave data. 
        a Ntimes x NElectrode filtered neural data snippit

    Returns
    -------
    int
        job index for tracking parallel jobs
    object 
        first element of tuple returned from leastsq. presumably the model
        parameters?
    float 
        norm of the residuals divided by the norm of the data
    '''
    (i,xys,times,data) = params
    objective = makeLSQminimizerPolar(xys,times,real(data))
    result = leastsq(objective,heuristic_B_polar(data,xys),full_output=1)
    return i,result[0],norm(result[2]['fvec'])/norm(data)
    
def heuristic_solver_synchronous(params):
    '''
    Heuristic fit of data to a synchronous wave solution.
    Intended to be used with neurotools.parallel
    

    Parameters
    ----------
    i    : integer
        the job number (will be returned with the result)
    xys  : 2D numeric array
        spatial locations of each channel
    times: 1D numeric array
        the time basis for the observation
    data : 3D numeric array, real valued
        wave data. 
        a Ntimes x NElectrode filtered neural data snippit

    Returns
    -------
    int
        job index for tracking parallel jobs
    object 
        first element of tuple returned from leastsq. presumably the model
        parameters?
    float 
        norm of the residuals divided by the norm of the data
    '''
    (i,xys,times,data) = params
    objective = makeLSQminimizerSynchronous(xys,times,real(data))
    result = leastsq(objective,heuristic_B_synchronous(data,xys),full_output=1)
    return i,result[0],norm(result[2]['fvec'])/norm(data)





