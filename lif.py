def euler_integrate_LIF(x,C=1.0,g_L=0.1,g_E=0.00074,E_E=0.0,E_L=-70.0,Thr=-60.0,tau=1.0,Fs=2000,g_S=None):
    '''
    Modeled after the LIF implementation in Baker Pinches Lemon 2003
    This one uses Euler integration though -- it appears similar
    
    C dV/dt = g_L (E_L - V) + g_E (E_E - V)
    
    x is synaptic input in units of "counts" for synaptic activation
    events.
    C   = 1       uF/cm^2  membrane capacitance
    g_L = 0.1     mS/cm^2  leak conductance
    E_L = -70     mV       resting potential
    g_E = 0.00074 mS/cm^2  excitatory conductance
    E_E = 0       mV  
    Thr = -60     mV       spiking threshold
    tau = 1.0     ms       synaptic time constant in seconds
    Fs  = 2000    Hz       sampling rate and integration frequency
    V   =         mV       membrane potential
    
    Note: Expected membrane time constant is 10ms
    Tau = RC = C/g_L = 1 uF / 0.1 mS = 10 ms. Checks out.
    Note: Expected EPSP is 100 microvolts (0.1mV)
    Emperically this checks out
    
    Unit check. 
    Farad Volt / Second = Ampere = Siemen * Volt = Volt / Ohm 
    uF * mV / ms = mS * mV
    1 millisiemen * 1 millivolt = 1 microamp
    1 microamp / 1 microfarad * 1 millisecond = 1 millivolt
       
    # Test code: single EPSP
    x = zeros(2000)
    x[100] = 1
    time, V, spikes, current = euler_integrate_LIF(x)
    clf()
    plot(time,V)
    plot(time,current)
    plot(time,x)
        
    '''
    dt        = 1000.0/Fs         # millisecond
    timescale = dt/C              # kiloohm or 1/microsiemen
    alpha     = 1.0/tau           # 1/millisecond
    if g_S is None:
        time  = arange(len(x))*dt # millisecond
        t     = arange(5*tau/dt)*dt
        alpha_function = t*exp(-alpha*t)
        g_S = g_E * alpha_function / np.max(alpha_function) 
        g_S = convolve(x,g_S,'full')[:len(x)]
        V      = zeros(shape(x))      # millivolts
        spikes = zeros(shape(x))      
    else:
        g_S    = float64(g_S)
        time   = arange(len(g_S))*dt # millisecond
        V      = zeros(shape(g_S))      # millivolts
        spikes = zeros(shape(g_S))
    # precomputing all expressions that can be precomputed
    # this may speed things up
    g_L_E_L = g_L*E_L
    E_E_g_S = E_E*g_S
    g_L_g_S = (g_L+g_S) * timescale
    g_L_E_L_E_E_g_S = (g_L_E_L+E_E_g_S) * timescale
    I_g_L_g_S = 1-g_L_g_S
    for i,t in enumerate(time):
        '''
        v = V[i-1]
        I_leak = g_L   *(E_L-v) # mV*mS = mV/Kohm = uA
        I_syn  = g_S[i]*(E_E-v) # mV*mS = mV/Kohm = uA
        I      = I_leak + I_syn      # uA
        dV     = I / C * dt          # uA/uF*ms = mV 
        '''
        v = g_L_E_L_E_E_g_S[i]+I_g_L_g_S[i]*V[i-1]
        if v>Thr:
            v=E_L
            spikes[i]=1
        V[i] = v
    return time, V, spikes, g_S



def exponential_integrate_LIF(x,C=1.0,g_L=0.1,g_E=0.00074,E_E=0.0,E_L=-70.0,Thr=-60.0,tau=1.0,Fs=2000.0,g_S=None):
    '''
    Modeled after the LIF implementation in Baker Pinches Lemon 2003
    
    C dV/dt = g_L (E_L - V) + g_E (E_E - V)
    
    x is synaptic input in units of "counts" for synaptic activation
    events.
    C   = 1       uF/cm^2  membrane capacitance
    g_L = 0.1     mS/cm^2  leak conductance
    E_L = -70     mV       resting potential
    g_E = 0.00074 mS/cm^2  excitatory conductance
    E_E = 0       mV  
    Thr = -60     mV       spiking threshold
    tau = 1.0     ms       synaptic time constant in seconds
    Fs  = 2000    Hz       sampling rate and integration frequency
    V   =         mV       membrane potential
    
    Note: Expected membrane time constant is 10ms
    Tau = RC = C/g_L = 1 uF / 0.1 mS = 10 ms. Checks out.
    Note: Expected EPSP is 100 microvolts (0.1mV)
    Emperically this checks out
    
    Unit check. 
    Farad Volt / Second = Ampere = Siemen * Volt = Volt / Ohm 
    uF * mV / ms = mS * mV
    1 millisiemen * 1 millivolt = 1 microamp
    1 microamp / 1 microfarad * 1 millisecond = 1 millivolt
       
    # Test code: single EPSP and check against Euler integrator
    x = zeros(2000)
    x[100] = 1
    time, V, spikes, current = euler_integrate_LIF(x)
    clf()
    plot(time,V)
    plot(time,current)
    plot(time,x)
    time, V, spikes, current = exponential_integrate_LIF(x)
    plot(time,V)
    plot(time,current)
    plot(time,x)
    # they appear to match
    '''
    dt        = 1000.0/Fs         # millisecond
    alpha     = 1.0/tau           # 1/millisecond
    if g_S is None:
        time      = arange(len(x))*dt # millisecond
        t     = arange(5*tau/dt)*dt
        alpha_function = t*exp(-alpha*t)
        g_S = g_E * alpha_function / np.max(alpha_function) 
        g_S = convolve(x,g_S,'full')[:len(x)]
        V      = zeros(shape(x))      # millivolts
        spikes = zeros(shape(x))      
    else:
        g_S    = float64(g_S)
        time   = arange(len(g_S))*dt # millisecond
        V      = zeros(shape(g_S))      # millivolts
        spikes = zeros(shape(g_S))      
    for i,t in enumerate(time):
        # exponential integrator
        A = g_L*E_L + g_S[i]*E_E
        B = g_L + g_S[i]
        D = A/B
        V[i] = (V[i-1]-D) * exp(-B/C*dt)+D
        if V[i]>Thr:
            V[i]=E_L
            spikes[i]=1
    return time, V, spikes, g_S







    
def exponential_moving_average(x,tau,Fs=1000):
    '''
    exponential_moving_average(x,tau,Fs=1000)
    x   : data
    tau : time constant in seconds
    Fs  : sampling rate of x in samples per second
    
    Implement exponential moving average as
    Y_{n+1} = (1-alpha) Y_n + alpha X_n
    
    This relates to convolving signal x with decaying exponential
    Y = X * [H(t) exp(-t/tau)]
    Where t is in seconds and H is the heaviside step function
    
    Alpha and tau may be related by considering the differential equation

    tau dY/dT = X-Y
    
    And both solving it as a linear equation and also re-writing it as a
    discrete difference equation
    
    DY = (X-Y) DT/tau
    [Y_{n+1}-Y] = (X-Y) DT/tau
    Y_{n+1} = (X-Y) DT/tau + Y
    Y_{n+1} = (1 - DT/tau) Y + DT/tau X
    
    and we find that alpha = DT/tau for the discrete update
    
    The exact solution to an impulse in X would be
    
    tau dY/dT = -Y, Y_0 = 1
    
    Y(t) = exp(-t/tau) * H(t)
    or Y(t) = H(t) exp(-t*alpha/DT)
    '''
    assert 0 # I feel like this is broken
    DT = 1./Fs
    alpha = DT/tau
    # a bit of a hack here: python intreprets negative indecies as indexing
    # from the end. We exploit this to set the initial conditions by 
    # placing a value in the last position of the output array y
    # this value will eventually be overwritten.
    y = zeros(shape(x))
    y[-1] = 0#x[0]
    for i in xrange(len(x)):
        y[i] = (1-alpha) * y[i-1] + alpha * x[i]
    return y
    
    
    
