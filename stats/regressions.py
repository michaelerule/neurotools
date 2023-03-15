#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Routines for common regression tasks.
'''
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function

import warnings
import numpy as np
from scipy.stats import linregress
from scipy.optimize import minimize

from neurotools.util.functions import npdf
from neurotools.stats.minimize import minimize_retry

'''
Regress on the following model for synchrony
synchrony(x) = np.cos(wx)*np.exp(-x/tau)+b

angular synchrony np.cos(theta_x1-theta_x2) should
decay as a damped cosine, with some constant offset b. Note that
a nonzero constant offset may not indicate uniform synchrony, for
example, the direction of constant phase in a plane wave will contribute
a DC component.

Uses L2 penaly

X: List of distances
W: List of weights
Y: List of average pairwise distances

Model is np.cos(wx)*np.exp(-x/L)+b
Generates predictions Z
error is \sum W*(Z-Y)^2

gradient of the error
dErr/dw np.sum(W*(np.cos(w*x)*np.exp(-x/L)+b-Y)**2)
dErr/dL np.sum(W*(np.cos(w*x)*np.exp(-x/L)+b-Y)**2)
dErr/db np.sum(W*(np.cos(w*x)*np.exp(-x/L)+b-Y)**2)

np.sum(W* dErr/dw (np.cos(w*x)*np.exp(-x/L)+b-Y)**2)
np.sum(W* dErr/dL (np.cos(w*x)*np.exp(-x/L)+b-Y)**2)
np.sum(W* dErr/db (np.cos(w*x)*np.exp(-x/L)+b-Y)**2)

np.sum(W* 2(np.cos(w*x)*np.exp(-x/L)+b-Y) dErr/dw (np.cos(w*x)*np.exp(-x/L)+b-Y))
np.sum(W* 2(np.cos(w*x)*np.exp(-x/L)+b-Y) dErr/dL (np.cos(w*x)*np.exp(-x/L)+b-Y))
np.sum(W* 2(np.cos(w*x)*np.exp(-x/L)+b-Y) dErr/db (np.cos(w*x)*np.exp(-x/L)+b-Y))

np.sum(W* 2(np.cos(w*x)*np.exp(-x/L)+b-Y) * -np.sin(w*x)*np.exp(-x/L) )
np.sum(W* 2(np.cos(w*x)*np.exp(-x/L)+b-Y) *  np.cos(w*x)*(-1/L)*np.exp(-x/L) )
np.sum(W* 2(np.cos(w*x)*np.exp(-x/L)+b-Y))


objective function is

def objective(w,L,b):
    z = np.cos(w*x)*np.exp(-x/L)+b
    error = np.sum( W*(z-Y)**2 )
    return error

def gradient(w,lambda,b):
    z = np.cos(w*x)*np.exp(-x/L)+b
    h = 2*(z-Y)
    dEdw = np.sum(W*h*-np.sin(w*x)*np.exp(-x/L))
    dEdL = np.sum(W*h* np.cos(w*x)*(-1/L)*np.exp(-x/L))
    dEdb = np.sum(W*H)
    return [dEdw,dEdL,dEdb]

Use the minimize function from scipy.optimize.

scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None,
    hessp=None, bounds=None, constraints=(), tol=None, callback=None,
    options=None)

    Minimization of scalar function of one or more variables.

    New in version 0.11.0.
    Parameters:
    fun : callable
        Objective function.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives
        (Jacobian, Hessian).
    method : str or callable, optional
        Type of solver. Should be one of
            'Nelder-Mead'
            'Powell'
            'CG'
            'BFGS'
            'Newton-CG'
            'Anneal (deprecated as of scipy version 0.14.0)'
            'L-BFGS-B'
            'TNC'
            'COBYLA'
            'SLSQP'
            'dogleg'
            'trust-ncg'
            custom - a callable object (added in version 0.14.0)
        If not given, chosen to be one of BFGS, L-BFGS-B, SLSQP, d
        epending if the problem has constraints or bounds.
    jac : bool or callable, optional
        Jacobian (gradient) of objective function. Only for CG, BFGS,
        Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg. If jac is a Boolean
        and is True, fun is assumed to return the gradient along with the
        objective function. If False, the gradient will be estimated
        numerically. jac can also be a callable returning the gradient of the
        objective. In this case, it must accept the same arguments as fun.
    hess, hessp : callable, optional
        Hessian (matrix of second-order derivatives) of objective function or
        Hessian of objective function times an arbitrary vector p. Only for
        Newton-CG, dogleg, trust-ncg. Only one of hessp or hess needs to be
        given. If hess is provided, then hessp will be ignored. If neither hess
        nor hessp is provided, then the Hessian product will be approximated
        using finite differences on jac. hessp must compute the Hessian times
        an arbitrary vector.
    bounds : sequence, optional
        Bounds for variables (only for L-BFGS-B, TNC and SLSQP). (min, max)
        pairs for each element in x, defining the bounds on that parameter. Use
        None for one of min or max when there is no bound in that direction.
    constraints : dict or sequence of dict, optional
        Constraints definition (only for COBYLA and SLSQP). Each constraint is
        defined in a dictionary with fields:
            type : str
                Constraint type: 'eq' for equality, 'ineq' for inequality.
            fun : callable
                The function defining the constraint.
            jac : callable, optional
                The Jacobian of fun (only for SLSQP).
            args : sequence, optional
                Extra arguments to be passed to the function and Jacobian.
        Equality constraint means that the constraint function result is to be
        zero whereas inequality means that it is to be non-negative. Note that
        COBYLA only supports inequality constraints.
    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.
    options : dict, optional
        A dictionary of solver options. All methods accept the following
        generic options:
            maxiter : int
                Maximum number of iterations to perform.
            disp : bool
                Set to True to print convergence messages.
        For method-specific options, see show_options.
    callback : callable, optional
        Called after each iteration, as callback(xk), where xk is the current
        parameter vector.
    Returns:
    res : OptimizeResult
        The optimization result represented as a OptimizeResult object.
        Important attributes are: x the solution array, success a Boolean flag
        indicating if the optimizer exited successfully and message which
        describes the cause of the termination. See OptimizeResult for a
        description of other attributes.

'''

def damped_cosine(X,Y,W):
    '''
    Regress a damped cosine impulse response to point data `X` and `Y` 
    using weighting `W`.
    
    Todo: constrain b, L to be positive

    Parameters
    ----------
    X: 1D array-like
        List of distances
    Y: 1D array-like
        List of average pairwise distances
    W: 1D array-like
        List of weights
    
    Returns
    -------
    result : object 
        Optimization result returned by `scipy.optimize.minimize`.
        See `scipy.optimize` documentation for details.

    Example
    -------
    ::
    
        X = 0.4*arange(9)
        Y = np.exp(-X/4+1)*np.cos(X)
        Z = Y+randn(*shape(X))
        W = ones(shape(X))
        w,L,b = damped_cosine(X,Z,W).x
        plot(X,Y)
        plot(X,Z)
        plot(X,np.cos(w*X)*np.exp(-X/L+b))
    '''
    X = np.float64(X)
    Y = np.float64(Y)
    def objective(wLb):
        (w,L,b) = wLb
        z = np.cos(w*X)*np.exp(-X/L+b)
        error = np.sum( W*(z-Y)**2 )
        return error
    def gradient(wLb):
        (w,L,b) = wLb
        # todo: double check this gradient
        z = np.cos(w*X)*np.exp(-X/L)+b
        h = 2*(z-Y)
        dEdw = np.sum(W*h*-np.sin(w*X)*np.exp(-X/L))
        dEdL = np.sum(W*h* np.cos(w*X)*(-1/L)*np.exp(-X/L))
        dEdb = np.sum(W*h)
        return arr([dEdw,dEdL,dEdb])
    result = minimize(objective,[1,1,0])#,jac=gradient)
    if not result.success:
        print(result.message)
        warnings.warn('Optimization failed: %s'%result.message)
    return result

def weighted_least_squares(X,Y,W):
    '''
    Initialize power law fit
    EPS = 1e-10
    use = (X>EPS)&(Y>EPS)
    weighted_least_squares(np.log(X+EPS)[use],np.log(Y+EPS)[use],1/(EPS+X[use]))
    
    Parameters
    ----------
    X: List of distances
    Y: List of amplitudes
    W: Weights for points 
    
    Returns
    -------
    result : object 
        Optimization result returned by scipy.optimize.minimize. See
        scipy.optimize documentation for details.
    '''
    X = np.float64(X)
    Y = np.float64(Y)
    def objective(ab):
        a,b=(a,b)
        return np.sum( W*(Y-(X*a+b))**2)
    a,b,_,_,_ = linregress(X,Y)
    result = minimize(objective,[a,b])
    if not result.success:
        print(result.message)
        warnings.warn('Optimization failed: %s'%result.message)
    return result


def power_law(X,Y,W):
    '''
    Fit a power law, but with error terms computed by r^2 in
    the original space.
    
    Parameters
    ----------
    X: List of distances
    Y: List of amplitudes
    W: Weights for points   
    
    Returns
    ------- 
    '''
    '''
    power law form is `np.log(y)=a*np.log(x)+b` or `y = b*x^a`

    initial best guess using linear regression.

    result = power_law(X,Y,1/X**16)
    a,b = result.x

    EPS = 1e-10
    use = (X>EPS)&(Y>EPS)
    a,b = weighted_least_squares(np.log(X)[use],np.log(Y)[use],W[use]).x
    plot(sorted(X),b*arr(sorted(X))**a)

    from numpy.polynomial.polynomial import polyfit

    X,Y = ravel(f),ravel(y[:,i])
    a,b = power_law(X,Y,1/X**2)
    plot(sorted(X),b*arr(sorted(X))**a)
    '''
    X = np.float64(X)
    Y = np.float64(Y)
    EPS = 1e-10
    use = (X>EPS)&(Y>EPS)
    a,b = polyfit(np.log(X)[use],np.log(Y)[use],1,w=W[use])
    '''
    def objective(ab):
        a,b = (a,b)
        z = np.exp(b+a*np.log(X))
        obj = np.sum((W*(Y-z)**2)[use])
        print(a,b,obj)
        return obj
    result = minimize(objective,[a,b])
    '''
    return a,np.exp(b)

def gaussian_function(X,Y):
    '''
    
    Parameters
    ----------
    X: List of distances
    Y: List of amplitudes
    
    Returns
    -------
    result : object 
        Optimization result returned by scipy.optimize.minimize. See
        scipy.optimize documentation for details.
        
    '''
    X = np.float64(X)
    Y = np.float64(Y)
    def objective(theta):
        (mu,sigma,scale,dc) = theta
        z = npdf(mu,sigma,X)*scale+dc
        error = np.sum( (z-Y)**2 )
        return error
    result = minimize(objective,[0,1,1])
    return result

def half_gaussian_function(X,Y):
    '''
    
    Parameters
    ----------
    X: List of distances
    Y: List of amplitudes
    
    Returns
    -------
    '''
    X = np.float64(X)
    Y = np.float64(Y)
    def objective(theta):
        (sigma,scale,dc) = theta
        z = npdf(0,sigma,X)*scale+dc
        error = np.sum( (z-Y)**2 )
        return error
    result = minimize(objective,[1,1,0])
    if not result.success:
        print(result.message)
        raise RuntimeError('Optimization failed: %s'%result.message)
    sigma,scale,dc = result.x
    return sigma,scale,dc

def exponential_decay(X,Y):
    '''
    Fit exponential decay from an initial value to a final value with 
    some time (or length, etc) constant.
    
    ``tau,scale,dc = exponential_decay(X,Y)`` 
    fits
    ``z = np.exp(-X/tau)*scale+dc``
    using least squares.
    
    Parameters
    ----------
    X: List of distances
    Y: List of amplitudes
        
    Returns
    -------
    tau : float
        Length constant of exponential fit
    scale: float
        Scale parameter (magnitude at zero) of exponential fit
    dc : float
        DC offset of exponential fit (asymptotic value)
    '''
    X = np.float64(X)
    Y = np.float64(Y)
    def error(theta):
        (tau,scale,dc) = theta
        z = np.exp(np.minimum(50.0,-X/tau))*scale+dc
        return np.mean( (z-Y)**2.0 )
    result = minimize_retry(error,[1,1,0],verbose=False,printerrors=False,show_progress=False)
    #if not result.success:
    #    print(result.message)
    #    warnings.warn('Optimization failed: %s'%result.message)
    # lamb,scale,dc = result.x
    tau,scale,dc = result
    return tau,scale,dc


def robust_line(X,Y):
    '''
    2-variable linear regression with L1 penalty
    returns the tuple (m,b) for line in y = mx+b format
    
    Parameters
    ----------
    X: List of distances
    Y: List of amplitudes
    
    Returns
    -------
    result.x : array-like 
        Optimization result returned by scipy.optimize.minimize. See
        scipy.optimize documentation for details.
        
    '''
    X = np.float64(X)
    Y = np.float64(Y)
    def pldist(x,y,m,b):
        return (-m*x+y-b)/np.sqrt(m**2+1)
    def objective(H):
        m,b = H
        return np.sum([np.abs(pldist(x,y,m,b)) for (x,y) in zip(X,Y)])
    res = scipy.optimize.minimize(objective,[1,0],method = 'Nelder-Mead')
    if not result.success:
        print(result.message)
        warnings.warn('Optimization failed: %s'%result.message)
    return res.x



import neurotools.linalg.matrix as nmatrix
import neurotools.stats.pvalues as npv
def cubic_spline_regression(x,y,
    df = 5,
    NBOOTSTRAP=1000,
    reg=1e-15,
    show_progress=False):
    '''
    Bivariate x→y cubic spline regression with bootstrap
    convidence intervals.
    
    Depends on the `cr()` function in the `patsy` package.
    
    Parametrs
    ---------
    x: length NSAMPLES iterable of scalars
        Independent variable 
    y: length NSAMPLES iterable of scalars
        Dependent variable
    
    Other Parameters
    ----------------
    df: int>0; default 5
        Degrees of freedom for cubic spline regression.
    NBOOTSTRAP: int>0; default 1000
        Number of samples to use for bootstrap
    reg: positive float; default 1e-15
        Regularization for linear least squares
    show_progress: boolean; default False
        Show progress bar while sampling bootstrap
        
    Returns
    -------
    x_hat: np.float32
        Sorted copy of `x`
    y_hat: np.float32
        Smooted curve evaluated at `x_eval`
    samples: np.float32
        NBOOTSTRAP × len(x_eval) bootstrap samples. 
    
    '''
    # Import in function TODO
    # I don't want to create a hard dependency
    # I do want this function to fail with instructions
    # to install patsy if not present (TODO)
    from patsy import cr
    
    x = np.float32([*x])
    y = np.float32([*y])
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    def cubic_spline_smooth(xy,df=5):
        x,y = np.float32(xy).T
        μ  = np.mean(y)
        y -= μ
        B = cr(x,df=df,constraints="center")
        M = nmatrix.reglstsq(B,y,reg=reg)
        return B@M.ravel() + μ

    population = [*zip(x,y)]
    y_hat   = cubic_spline_smooth(population)
    
    samples = np.float64(npv.bootstrap_statistic(
        cubic_spline_smooth,
        population,
        ntrials=NBOOTSTRAP,
        show_progress=show_progress))
    return x, y_hat, samples


import scipy.interpolate
import neurotools.linalg.matrix as nmatrix
import neurotools.stats.pvalues as npv
def cubic_spline_regression(x,y,x_eval,
    df = 5,
    NBOOTSTRAP=1000,
    reg=1e-15,
    show_progress=False):
    '''
    Bivariate x→y cubic spline regression with bootstrap
    convidence intervals.
    
    Splines are spaced according to the points in
    `x_eval`, not `x`.
    
    Depends on the `cr()` function in the `patsy` package.
    
    Parametrs
    ---------
    x: length NSAMPLES iterable of scalars
        Independent variable 
    y: length NSAMPLES iterable of scalars
        Dependent variable
    x_eval: iterable of scalars
        Points at which to evalute the resulting model.
    
    Other Parameters
    ----------------
    df: int>0; default 5
        Degrees of freedom for cubic spline regression.
    NBOOTSTRAP: int>0; default 1000
        Number of samples to use for bootstrap
    reg: positive float; default 1e-15
        Regularization for linear least squares
    show_progress: boolean; default False
        Show progress bar while sampling bootstrap
        
    Returns
    -------
    y_hat: np.float32
        Smooted curve evaluated at `x_eval`
    samples: np.float32
        NBOOTSTRAP × len(x_eval) bootstrap samples. 
    
    '''
    # Import in function TODO
    # I don't want to create a hard dependency
    # I do want this function to fail with instructions
    # to install patsy if not present (TODO)
    from patsy import cr
    
    x = np.float32([*x])
    y = np.float32([*y])
    x_eval = np.float32([*x_eval])
    
    # Basis functions
    B  = cr(x_eval,df=df,constraints="center")
    Bf = scipy.interpolate.interp1d(
        x_eval,
        B,
        axis=0,
        bounds_error=False,
        fill_value='extrapolate',
        assume_sorted=False)

    def cubic_spline_smooth(xy,df=5):
        x,y = np.float32(xy).T
        
        μ  = np.mean(y)
        y -= μ
        M = nmatrix.reglstsq(Bf(x),y,reg=reg)
        return B@M.ravel() + μ

    population = [*zip(x,y)]
    y_hat   = cubic_spline_smooth(population)
    
    samples = np.float64(npv.bootstrap_statistic(
        cubic_spline_smooth,
        population,
        ntrials=NBOOTSTRAP,
        show_progress=show_progress))
    return y_hat, samples
    
    
    
    
    

from neurotools.jobs.parallel import limit_cores, parmap, parcontext
from neurotools.stats.information import betapr
from neurotools.stats.pvalues import bootstrap_statistic
from neurotools.linalg.matrix import reglstsq

class CircregressResult():

    def __init__(
        self,
        theta,
        y,
        nboot  = 1000,
        nshuff = 1000,
        parallel = True,
        show_progress = False,
        #save_samples = True,
        #save_training_data = True
    ):
        # Convert to polar featuress 
        x  = np.float32([
            np.cos(theta),
            np.sin(theta), 
            np.ones(theta.shape)
        ])
        y  = np.float32(y).copy()
        w  = np.squeeze(
            CircregressResult._linregress([*zip(x.T,y)])
        )
        with parcontext():
            wb = np.squeeze(
                parmap(
                    CircregressResult._boothelper ,
                    [(i,(x,y)) for i in range(nboot)],
                    show_progress = show_progress,
                    debug=not parallel)
            ).reshape(nboot ,3)
            ws = np.squeeze(
                parmap(
                    CircregressResult._shuffhelper,
                    [(i,(x,y)) for i in range(nshuff)],
                    show_progress = show_progress,
                    debug=not parallel)
            ).reshape(nshuff,3)
        
        '''
        if save_training_data:
            self.x = x
            self.y = y
        if save_samples:
            self.w_bootstrap = wb
            self.w_shuffle = ws
        '''
        self.w = w
        self.x = x
        self.y = y
        self.a = w[0]
        self.b = w[1]
        self.c = w[2]
        self.w_bootstrap = wb
        self.w_shuffle = ws
        
        # Confidence on magnitude
        r2  = np.linalg.norm(w [  :2],2  )**2
        r2lo, r2hi = np.nanpercentile(self.get_d2_samples(),[2.5,97.5])
        self.d   = r2**.5
        self.dlo = r2lo**.5
        self.dhi = r2hi**.5
        
        # Angle confidence
        θhat    = np.angle(w [  :2]@[1,1j])
        center  = np.exp(1j*θhat)
        θboot   = np.angle( (wb[:,:2]@[1,1j])/center )
        θlo,θhi = np.nanpercentile(θboot,[2.5,97.5])
        θlo += θhat
        θhi += θhat
        self.theta    = θhat
        self.theta_lo = θlo
        self.theta_hi = θhi
        
        # Coefficient of dertermination
        R2     = 1 - np.mean((w@x-y)**2)/np.var(y)
        R2lo, R2hi = np.nanpercentile(self.get_R2_samples(),[2.5,97.5])
        self.R2    = R2
        self.R2lo  = R2lo
        self.R2hi  = R2hi

        # P-value on weight magnitude
        r20    = np.linalg.norm(ws[:,:2],2,1)**2
        pvalue = betapr(sum(r20>r2),len(r20))
        self.pvalue = pvalue
        
    def get_theta_samples(self):
        wb  = self.w_bootstrap
        return np.angle((wb[:,:2]@[1,1j]))
        
    def get_d2_samples(self):
        wb  = self.w_bootstrap
        return np.linalg.norm(wb[:,:2],2,1)**2        
        
    def get_R2_samples(self):
        wb = self.w_bootstrap
        x  = self.x
        y  = self.y
        # Coefficient of dertermination
        return 1 - np.mean((wb@x-y)**2,-1)/np.var(y)

    def __iter__(self):
        yield self.pvalue 
        yield self.a  
        yield self.b  
        yield self.c  
        yield self.d 
        yield self.dlo
        yield self.dhi
        yield self.theta    
        yield self.theta_lo 
        yield self.theta_hi 
        yield self.R2     
        yield self.R2lo   
        yield self.R2hi   
        
    def _asdict(self):
        return {
        'pvalue':self.pvalue,
        'a':self.a,
        'b':self.b,
        'c':self.c,
        'd':self.d,
        'dlo':self.dlo,
        'dhi':self.dhi,
        'theta':self.theta,
        'theta_lo':self.theta_lo,
        'theta_hi':self.theta_hi,
        'R2':self.R2,
        'R2lo':self.R2lo,
        'R2hi':self.R2hi,
        }
    
    def __getitem__(self,s):
        return [*self][s]

    def _linregress(p):
        x,y = zip(*p)
        return reglstsq(np.float32(x),y)

    def _boothelper(p):
        i,(x,t) = p
        np.random.seed(i)
        limit_cores(1)
        xt = [*zip(x.T,t)]
        return i,np.array(bootstrap_statistic(
            CircregressResult._linregress,
            xt,1,show_progress=False))

    def _shuffhelper(p):
        i,(x,t) = p
        np.random.seed(i)
        limit_cores(1)
        ti = np.random.choice(t,len(t),replace=False)
        xt = [*zip(x.T,ti)]
        return i,np.squeeze(
            CircregressResult._linregress(xt))
    
    
def circregress(*args,**kwargs):
    return CircregressResult(*args,**kwargs)
